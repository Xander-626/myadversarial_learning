# Adversarial YOLO
This repository is based on the marvis YOLOv2 inplementation: https://github.com/marvis/pytorch-yolo2

This work corresponds to the following paper: https://arxiv.org/abs/1904.08653:
```
@inproceedings{thysvanranst2019,
    title={Fooling automated surveillance cameras: adversarial patches to attack person detection},
    author={Thys, Simen and Van Ranst, Wiebe and Goedem\'e, Toon},
    booktitle={CVPRW: Workshop on The Bright and Dark Sides of Computer Vision: Challenges and Opportunities for Privacy and Security},
    year={2019}
}
```

If you use this work, please cite this paper.

# What you need
We use Python 3.6.
Make sure that you have a working implementation of PyTorch installed, to do this see: https://pytorch.org/
To visualise progress we use tensorboardX which can be installed using pip:
```
pip install tensorboardX tensorboard
```
No installation is necessary, you can simply run the python code straight from this directory.

Make sure you have the YOLOv2 MS COCO weights:
```
mkdir weights; curl https://pjreddie.com/media/files/yolov2.weights -o weights/yolo.weights
```

Get the INRIA dataset:
```
curl ftp://ftp.inrialpes.fr/pub/lear/douze/data/INRIAPerson.tar -o inria.tar
tar xf inria.tar
mv INRIAPerson inria
cp -r yolo-labels inria/Train/pos/
```

# Generating a patch
`patch_config.py` contains configuration of different experiments. You can design your own experiment by inheriting from the base `BaseConfig` class or an existing experiment. `ReproducePaperObj` reproduces the patch that minimizes object score from the paper (With a lower batch size to fit on a desktop GPU).

You can generate this patch by running:
```
python train_patch.py paper_obj
```


## macOS / Apple silicon training

The local training entry point now supports current PyTorch on MPS, CPU, and CUDA.
The original Python 3.6/CUDA setup above is historical. This adaptation has been
smoke-tested with Python 3.13, PyTorch 2.14.1, torchvision 0.29.1, and Apple MPS.

```bash
conda activate adversarial-yolo
cd /Users/ripple/pyproject/adversarial-yolo
python train_patch.py paper_obj --device mps --batch-size 1 --epochs 1 --max-steps 1
```

Remove `--max-steps 1` to train a complete epoch. Set `--epochs 10` (or your
chosen epoch count) for a longer run. `--device auto` selects MPS, then CUDA,
then CPU; `--device cpu` forces CPU. macOS defaults to batch size 1 and
`--workers 0`. `--no-tensorboard` disables logging. View logs separately with:

```bash
tensorboard --logdir runs
```

The patch is saved after each epoch (or at the `--max-steps` limit) to
`saved_patches/mac_patch.png`; use `--output` to choose another destination.
This PNG is the generated patch, not a resumable optimizer checkpoint.
The detector is frozen while gradients still flow through it to the patch.
Grid sampling explicitly uses `align_corners=True` to preserve the historical
PyTorch coordinate convention. Unsupported MPS operations may fall back to CPU.

YOLOv2 weights and the original INRIA images with matching repository labels
are required. Use `curl -fL` when downloading weights to follow redirects.
A short optimization test establishes execution compatibility, not reproduction
of the paper's final attack performance. Legacy evaluation scripts, notebooks,
and the standalone `load_data.py` demo retain their original CUDA assumptions.
