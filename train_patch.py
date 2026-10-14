"""Train adversarial patches on MPS, CUDA, or CPU."""
import argparse
import os
from pathlib import Path

# Must be set before importing torch; unsupported MPS operations may use CPU.
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

import patch_config
from load_data import (Darknet, InriaDataset, MaxProbExtractor, NPSCalculator,
                       PatchApplier, PatchTransformer, TotalVariation)


def select_device(name):
    if name == 'auto':
        name = ('mps' if torch.backends.mps.is_available() else
                'cuda' if torch.cuda.is_available() else 'cpu')
    if name == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS is unavailable. Use --device cpu or check your PyTorch installation.')
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Use --device auto or cpu.')
    return torch.device(name)


class PatchTrainer:
    def __init__(self, args):
        self.args = args
        self.config = patch_config.patch_configs[args.mode]()
        self.device = select_device(args.device)
        self.batch_size = args.batch_size or (1 if self.device.type in ('mps', 'cpu')
                                             else self.config.batch_size)
        print(f'Device: {self.device}; batch size: {self.batch_size}', flush=True)
        weights = Path(self.config.weightfile)
        if not weights.is_file() or weights.stat().st_size < 1_000_000:
            raise ValueError(f'{weights}: missing or invalid YOLOv2 weights. Download the binary weights with curl -fL.')
        with weights.open('rb') as handle:
            prefix = handle.read(256).lstrip().lower()
        if prefix.startswith((b'<', b'<!doctype')):
            raise ValueError(f'{weights} contains HTML instead of model weights.')
        self.model = Darknet(self.config.cfgfile)
        self.model.load_weights(str(weights))
        self.model.eval().requires_grad_(False).to(self.device)
        self.transformer = PatchTransformer().to(self.device)
        self.applier = PatchApplier().to(self.device)
        self.extractor = MaxProbExtractor(0, 80, self.config).to(self.device)
        self.nps = NPSCalculator(self.config.printfile, self.config.patch_size).to(self.device)
        self.tv = TotalVariation().to(self.device)
        self.writer = None
        if not args.no_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(comment=f'_{args.mode}_{self.device.type}')

    def train(self):
        dataset = InriaDataset(self.config.img_dir, self.config.lab_dir, 14,
                               self.model.height, shuffle=True)
        if not len(dataset):
            raise ValueError('The training dataset is empty.')
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                                            shuffle=True, num_workers=self.args.workers)
        patch = torch.full((3, self.config.patch_size, self.config.patch_size),
                           0.5, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([patch], lr=self.config.start_learning_rate, amsgrad=True)
        scheduler = self.config.scheduler_factory(optimizer)
        step = 0
        try:
            for epoch in range(self.args.epochs):
                total = 0.0
                batches = 0
                for images, labels in tqdm(loader, desc=f'Epoch {epoch + 1}'):
                    images, labels = images.to(self.device), labels.to(self.device)
                    optimizer.zero_grad(set_to_none=True)
                    transformed = self.transformer(patch, labels, self.model.height,
                                                   do_rotate=True, rand_loc=False)
                    patched = self.applier(images, transformed)
                    patched = F.interpolate(patched, (self.model.height, self.model.width))
                    detection = self.extractor(self.model(patched)).mean()
                    nps_loss = self.nps(patch) * 0.01
                    tv_loss = self.tv(patch) * 2.5
                    loss = detection + nps_loss + tv_loss.clamp_min(0.1)
                    if not torch.isfinite(loss).item():
                        raise RuntimeError('Non-finite training loss.')
                    loss.backward()
                    if patch.grad is None or not torch.isfinite(patch.grad).all().item():
                        raise RuntimeError('Missing or non-finite patch gradients.')
                    optimizer.step()
                    with torch.no_grad():
                        patch.clamp_(0, 1)
                    value = loss.item()
                    total += value
                    batches += 1
                    if self.writer and step % 5 == 0:
                        self.writer.add_scalar('loss/total', value, step)
                        self.writer.add_scalar('loss/detection', detection.item(), step)
                        self.writer.add_image('patch', patch.detach().cpu(), step)
                    step += 1
                    if self.args.max_steps and step >= self.args.max_steps:
                        self.save_patch(patch)
                        print(f'Completed {step} steps; loss={value:.6f}', flush=True)
                        return
                scheduler.step(total / batches)
                self.save_patch(patch)
                print(f'Epoch {epoch + 1}: loss={total / batches:.6f}', flush=True)
        finally:
            if self.writer:
                self.writer.close()

    def save_patch(self, patch):
        output = Path(self.args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        transforms.ToPILImage()(patch.detach().cpu()).save(output)
        print(f'Saved patch: {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=patch_config.patch_configs)
    parser.add_argument('--device', choices=('auto', 'mps', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--max-steps', type=int, default=None, help='Stop after this many training batches.')
    parser.add_argument('--workers', type=int, default=0, help='Use 0 on macOS to avoid multiprocessing issues.')
    parser.add_argument('--no-tensorboard', action='store_true')
    parser.add_argument('--output', default='saved_patches/mac_patch.png')
    args = parser.parse_args()
    if args.epochs < 1 or args.workers < 0 or (args.batch_size is not None and args.batch_size < 1) or (args.max_steps is not None and args.max_steps < 1):
        parser.error('epochs, batch-size, and max-steps must be positive; workers must be nonnegative.')
    os.chdir(Path(__file__).resolve().parent)
    PatchTrainer(args).train()


if __name__ == '__main__':
    main()
