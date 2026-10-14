"""Compare clean, gray-patched, and trained-patched INRIA test images."""
import argparse
import csv
import json
import os
import re
from pathlib import Path

os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
import torch
from PIL import Image, ImageDraw
from torchvision import transforms
from torchvision.ops import box_iou, nms
from load_data import Darknet, PatchApplier, PatchTransformer
from train_patch import select_device


def prepare(path, annotations, size):
    with Image.open(path) as source:
        image = source.convert('RGB')
    w, h = image.size
    side = max(w, h)
    left, top = (side - w) // 2, (side - h) // 2
    square = Image.new('RGB', (side, side), (127, 127, 127))
    square.paste(image, (left, top))
    text = annotations.read_text(errors='replace')
    boxes = re.findall(r'Bounding box.*?:\s*\((-?\d+),\s*(-?\d+)\)\s*-\s*\((-?\d+),\s*(-?\d+)\)', text)
    if not boxes:
        raise ValueError(f'No ground-truth boxes in {annotations}')
    gt = torch.tensor([[((int(x1) + left) / side), ((int(y1) + top) / side),
                        ((int(x2) + left) / side), ((int(y2) + top) / side)]
                       for x1, y1, x2, y2 in boxes], dtype=torch.float32)
    centers = (gt[:, :2] + gt[:, 2:]) / 2
    labels = torch.cat((torch.zeros((len(gt), 1)), centers, gt[:, 2:] - gt[:, :2]), 1)
    pixels = transforms.ToTensor()(square.resize((size, size), Image.Resampling.BILINEAR))
    return pixels, labels, gt


def decode(output, model, conf, nms_iou):
    # Decode on CPU, using person probability * objectness as the score.
    output = output.detach().cpu()[0]
    _, h, w = output.shape
    raw = output.reshape(model.num_anchors, 5 + model.num_classes, h, w).permute(0, 2, 3, 1)
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    anchors = torch.tensor(model.anchors).reshape(model.num_anchors, -1)[:, :2]
    center = (raw[..., :2].sigmoid() + torch.stack((xx, yy), -1)) / torch.tensor([w, h])
    wh = raw[..., 2:4].exp() * anchors[:, None, None, :] / torch.tensor([w, h])
    probability = raw[..., 5:].softmax(-1)
    scores = raw[..., 4].sigmoid() * probability[..., 0]
    mask = (scores >= conf) & (probability.argmax(-1) == 0)
    boxes = torch.cat((center - wh / 2, center + wh / 2), -1)[mask]
    scores = scores[mask]
    finite = torch.isfinite(boxes).all(1) & torch.isfinite(scores)
    boxes, scores = boxes[finite], scores[finite]
    keep = nms(boxes, scores, nms_iou)
    return boxes[keep], scores[keep]


def match(boxes, scores, gt, threshold):
    matched = set()
    if len(boxes):
        overlaps = box_iou(boxes, gt)
        for index in scores.argsort(descending=True):
            for target in overlaps[index].argsort(descending=True).tolist():
                if target not in matched and overlaps[index, target] >= threshold:
                    matched.add(target)
                    break
    return matched


def render(pixels, boxes, scores, gt, title):
    image = transforms.ToPILImage()(pixels.cpu())
    size = image.width
    draw = ImageDraw.Draw(image)
    for box in gt:
        draw.rectangle(tuple((box * size).tolist()), outline='orange', width=1)
    for box, score in zip(boxes, scores):
        coords = tuple((box * size).tolist())
        draw.rectangle(coords, outline='lime', width=2)
        draw.text((max(0, coords[0]), max(0, coords[1] - 12)), f'person {score:.2f}', fill='lime')
    panel = Image.new('RGB', (size, size + 30), 'white')
    panel.paste(image, (0, 30))
    ImageDraw.Draw(panel).text((8, 8), title, fill='black')
    return panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--patch', default='saved_patches/baseline_10epochs.png')
    parser.add_argument('--device', choices=('auto', 'mps', 'cpu'), default='auto')
    parser.add_argument('--limit', type=int, default=20, help='0 evaluates every test image.')
    parser.add_argument('--conf', type=float, default=0.4)
    parser.add_argument('--nms-iou', type=float, default=0.4)
    parser.add_argument('--match-iou', type=float, default=0.5)
    parser.add_argument('--output', default='testing_mac')
    args = parser.parse_args()
    if args.limit < 0 or not all(0 < x <= 1 for x in (args.conf, args.nms_iou, args.match_iou)):
        parser.error('limit must be nonnegative; thresholds must be in (0, 1].')
    os.chdir(Path(__file__).resolve().parent)
    device = select_device(args.device)
    model = Darknet('cfg/yolo.cfg')
    model.load_weights('weights/yolo.weights')
    model.eval().requires_grad_(False).to(device)
    with Image.open(args.patch) as image:
        patch = transforms.ToTensor()(image.convert('RGB')).to(device)
    if patch.shape[1] != patch.shape[2] or patch.shape[1] > model.height:
        raise ValueError('The patch must be square and no larger than the model input.')
    transformer = PatchTransformer().to(device)
    # Fixed placement, no random rotation, light change, or noise for paired comparison.
    transformer.min_contrast = transformer.max_contrast = 1.0
    transformer.min_brightness = transformer.max_brightness = 0.0
    transformer.noise_factor = 0.0
    applier = PatchApplier().to(device)
    paths = sorted(p for p in Path('inria/Test/pos').iterdir() if p.suffix.lower() in ('.png', '.jpg'))
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        raise ValueError('No test images found.')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    totals = dict.fromkeys(('clean', 'gray', 'trained'), 0)
    suppressed = dict.fromkeys(('gray', 'trained'), 0)
    ground_truth = 0
    records = []
    with torch.no_grad():
        for number, path in enumerate(paths, 1):
            pixels, labels, gt = prepare(path, Path('inria/Test/annotations') / (path.stem + '.txt'), model.height)
            pixels, labels = pixels.to(device), labels.to(device)
            variants = {'clean': pixels}
            for name, candidate in (('gray', torch.full_like(patch, 0.5)), ('trained', patch)):
                transformed = transformer(candidate, labels.unsqueeze(0), model.height, do_rotate=False, rand_loc=False)
                variants[name] = applier(pixels.unsqueeze(0), transformed)[0]
            panels, matches = [], {}
            row = {'image': path.name, 'ground_truth': len(gt)}
            for name, image in variants.items():
                boxes, scores = decode(model(image.unsqueeze(0)), model, args.conf, args.nms_iou)
                matches[name] = match(boxes, scores, gt, args.match_iou)
                count = len(matches[name])
                totals[name] += count
                row[name + '_matched'] = count
                row[name + '_detections'] = len(boxes)
                panels.append(render(image, boxes, scores, gt, f'{name}: matched {count}/{len(gt)}'))
            for name in suppressed:
                suppressed[name] += len(matches['clean'] - matches[name])
            ground_truth += len(gt)
            records.append(row)
            comparison = Image.new('RGB', (model.height * 3, model.height + 30), 'white')
            for index, panel in enumerate(panels):
                comparison.paste(panel, (index * model.height, 0))
            comparison.save(output / (path.stem + '_comparison.png'))
            print(f'{number}/{len(paths)} {row}', flush=True)
    summary = {'images': len(paths), 'ground_truth_people': ground_truth,
               'matched_people': totals, 'recall': {k: v / ground_truth for k, v in totals.items()},
               'clean_detected_people_lost': suppressed,
               'conditional_suppression_rate': {k: v / totals['clean'] if totals['clean'] else None for k, v in suppressed.items()},
               'settings': vars(args),
               'note': 'Fixed digital placement using test annotations; sorted sample; not paper AP or physical-world evaluation.'}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    with (output / 'per_image.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
