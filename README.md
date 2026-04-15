# Gender Recognition from Facial Images

Deep Learning and Applications (DLA) course project — University of Cagliari.

## Problem

Binary gender classification from facial images using Convolutional Neural Networks.
We compare a **custom CNN** architecture against **finetuned pretrained models** (MobileNetV2, Xception) on the CelebA dataset.

## Dataset

[CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) — Large-scale CelebFaces Attributes Dataset.

- ~200K celebrity face images
- 40 binary attribute annotations (we use the `Male` attribute)
- Pre-cropped and aligned face images

The dataset is **not** included in this repository. See [Setup](#setup) for download instructions.

## Architecture Overview

```
src/
├── data/            # Dataset loading, preprocessing, transforms
├── models/          # CNN architectures (custom + finetuned)
├── training/        # Training loop, evaluation, metrics
└── utils/           # Config loading, reproducibility, helpers
configs/             # YAML training configurations
scripts/             # CLI entry points (train, evaluate)
notebooks/           # Exploratory analysis and visualization
tests/               # Unit and integration tests
```

### Detection Strategy

1. **Custom CNN**: lightweight architecture designed from scratch to understand baseline performance.
2. **Finetuned MobileNetV2**: efficient pretrained backbone, replace the classification head, finetune on CelebA.
3. **Finetuned Xception**: deeper pretrained backbone for higher accuracy comparison.

All models output a binary prediction (male / female) via a sigmoid-activated head.

## Setup

### Prerequisites

- Python >= 3.10
- macOS (MPS) or Linux (CUDA) for GPU acceleration

### Installation

```bash
git clone https://github.com/Coss02/dla-gender-recognition.git
cd dla-gender-recognition

python -m venv venv
source venv/bin/activate   # Linux/macOS

pip install -r requirements.txt
```

### Download CelebA

The dataset should be placed in `data/celeba/`. You can:

1. Download manually from the [official page](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
2. Use `torchvision.datasets.CelebA` (automatic download, may be unreliable)
3. Use `gdown` to download from Google Drive mirrors

### Weights & Biases

```bash
wandb login
```

## Usage

```bash
# Train with default config
python scripts/train.py --config configs/default.yaml

# Train a specific model
python scripts/train.py --config configs/default.yaml --model mobilenet_v2

# Evaluate a checkpoint
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt
```

## Git Workflow

We follow **Git Flow**:

- `main` — stable, release-ready code
- `develop` — integration branch for features
- `feature/<name>` — individual features and detectors

Branch naming: `feature/custom-cnn`, `feature/mobilenet-finetuning`, `feature/data-pipeline`, etc.

## Trade-offs and Limitations

- **Binary classification only**: CelebA annotates gender as binary (`Male` attribute). This is a simplification that does not reflect real-world gender diversity. We adopt it strictly as a technical benchmark.
- **Dataset bias**: CelebA is biased toward celebrities, which may not generalize to other populations.
- **No face detection**: we assume pre-cropped, aligned faces (as provided by CelebA).

## Potential Improvements

- Multi-task learning (predict multiple CelebA attributes simultaneously)
- Grad-CAM or attention visualization for interpretability
- Cross-dataset evaluation (e.g., LFW, UTKFace)
- Data augmentation ablation study

## Team

- Marco Cosseddu
- [Colleague name]

## License

This project is developed for academic purposes as part of the DLA course at University of Cagliari.
