# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeblurGAN-v2 is an ICCV 2019 paper implementation for single image motion deblurring using a relativistic conditional GAN with a Feature Pyramid Network (FPN) generator. It supports multiple backbone architectures (InceptionResNet-v2, MobileNet-v2, DenseNet, SEResNext) and discriminator types (patch, double-scale, multi-scale, or none).

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Train (uses config/config.yaml)
python train.py
python train.py --config_path=config/config.yaml

# Inference - single image
python predict.py test_img/000027.png

# Inference - batch with glob pattern
python predict.py "test_images/*.png" --out_dir results/

# Inference - video deblurring
python predict.py video.mp4 --video --out_dir output/

# Inference - with custom weights
python predict.py image.png --weights_path fpn_mobilenet.h5

# Inference - side-by-side comparison
python predict.py image.png --side_by_side=True

# Run tests
bash test.sh
# or: python -m unittest discover .

# TensorBoard monitoring
tensorboard --logdir=fpn
```

## Architecture

### Generator Flow
Input -> Pretrained backbone (frozen during warmup) -> FPN multi-scale features -> Top-down pathway with lateral connections -> Upsample + concatenate heads -> Residual addition (output = tanh(net_output) + input) -> clamp[-1, 1]

### Key Modules and Their Roles

- **`models/networks.py`**: Central registry. `get_generator()` and `get_discriminator()` instantiate models by name. All models are wrapped in `nn.DataParallel`.
- **`models/fpn_inception.py`**: Default generator (FPN + InceptionResNet-v2). Has `freeze()`/`unfreeze()` for backbone warmup.
- **`models/fpn_mobilenet.py`**, **`fpn_densenet.py`**, **`unet_seresnext.py`**: Alternative generator backbones.
- **`models/losses.py`**: Perceptual loss (VGG19 layer 3-3), L2 loss, and GAN loss variants (ragan-ls, wgan-gp, lsgan, ragan).
- **`adversarial_trainer.py`**: Factory pattern for GAN training modes (NoGAN, SingleGAN, DoubleGAN). DoubleGAN combines patch + full-image discriminators.
- **`dataset.py`**: `PairedDataset` uses hash-based deterministic train/val splitting controlled by `bounds` in config. Supports glob patterns and optional preloading.
- **`aug.py`**: Data augmentation pipeline using albumentations (geometric + corruption augmentations).
- **`train.py`**: `Trainer` class handles warmup (frozen backbone for N epochs), alternating G/D updates, checkpointing (`best_fpn.h5`, `last_fpn.h5`).
- **`metric_counter.py`**: Tracks PSNR, SSIM, G/D losses. Writes to TensorBoard.
- **`predict.py`**: `Predictor` pads input to multiples of 32, normalizes to [-1,1], runs inference. Note: model runs in **train mode** (not eval) to use actual batch statistics in norm layers.
- **`schedulers.py`**: `LinearDecay` and `WarmRestart` LR schedulers.
- **`util/image_pool.py`**: Image history buffer for GAN training stability.

### Configuration (`config/config.yaml`)

All training parameters are in a single YAML file using anchors/references:
- `train.files_a` / `train.files_b`: Dataset paths (glob patterns, YAML anchors shared with val)
- `train.bounds` / `val.bounds`: Hash-based split ratios (default: [0, 0.9] train, [0.9, 1] val)
- `model.g_name`: Generator backbone (`fpn_inception`, `fpn_mobilenet`, `fpn_dense`, `unet_seresnext`, `resnet`)
- `model.d_name`: Discriminator type (`double_gan`, `patch_gan`, `multi_scale`, `no_gan`)
- `model.content_loss`: `perceptual` (VGG19) or `l1`
- `model.learn_residual`: True means output = input + network_output
- `warmup_num`: Epochs with frozen backbone (default 3)

### Training Loop Structure
1. Warmup phase: backbone frozen, only FPN heads trained
2. After warmup: `netG.module.unfreeze()` called, new optimizer created for all params
3. Each epoch: D update -> G content loss + adv loss -> metrics
4. Best model saved by PSNR, last model saved every epoch
5. Checkpoints saved as `{best|last}_{experiment_desc}.h5`

### Weight File Format
`.h5` files are PyTorch state dicts (not HDF5 despite extension):
```python
torch.save({'model': netG.state_dict()}, 'best_fpn.h5')
```
