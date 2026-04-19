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
- Official train/val/test split via `list_eval_partition.txt`

The dataset is **not** included in this repository. See [Setup](#setup) for download instructions.

## Architecture Overview

```
src/
├── data/
│   ├── dataset.py           # CelebAGenderDataset with official splits
│   └── transforms.py        # Train (augmented) and eval (deterministic) pipelines
├── models/
│   ├── factory.py            # Model registry and builder
│   ├── custom_cnn.py         # 4-block CNN baseline
│   └── finetuned.py          # MobileNetV2 and Xception with freeze/unfreeze
├── training/
│   ├── trainer.py            # Training loop, early stopping, checkpointing, wandb
│   └── metrics.py            # Accuracy, precision, recall, F1, confusion matrix
└── utils/
    ├── config.py             # YAML loading and CLI override merging
    └── reproducibility.py    # Seed management for deterministic runs

configs/                      # YAML configs per model + ablation experiments
scripts/                      # CLI entry points (train, evaluate, compare)
notebooks/                    # Data exploration, failure analysis, learning curves
tests/                        # Unit tests (39 tests, all passing)
report/                       # Report template and deliverables
```

### Models

1. **Custom CNN** (`custom_cnn`): 4-block ConvNet (Conv-BN-ReLU-MaxPool) with AdaptiveAvgPool and FC head. ~600K parameters. Trained from scratch as a baseline.
2. **MobileNetV2** (`mobilenet_v2`): torchvision pretrained backbone with replaced classifier head (1280 -> 1). Backbone frozen initially for warmup.
3. **Xception** (`xception`): timm legacy_xception pretrained backbone with replaced fc head (2048 -> 1). Uses 299x299 input. Backbone frozen initially.

All models output raw logits trained with `BCEWithLogitsLoss`.

## Setup

### Prerequisites

- Python >= 3.10
- macOS (MPS) or Linux (CUDA) for GPU acceleration

### Installation

```bash
git clone https://github.com/Coss02/dla-gender-recognition.git
cd dla-gender-recognition

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### Download CelebA

The dataset should be placed in `data/celeba/` with this structure:

```
data/celeba/
├── img_align_celeba/          # face images
├── list_attr_celeba.txt       # attribute annotations
└── list_eval_partition.txt    # official train/val/test split
```

Download options:
1. [Official page](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) (manual)
2. `torchvision.datasets.CelebA` (automatic, may be unreliable)
3. `gdown` for Google Drive mirrors

### Weights & Biases

```bash
wandb login
```

## Usage

```bash
# Train a model
python scripts/train.py --config configs/custom_cnn.yaml
python scripts/train.py --config configs/mobilenet_v2.yaml
python scripts/train.py --config configs/xception.yaml

# Override hyperparameters from CLI
python scripts/train.py --config configs/custom_cnn.yaml --lr 0.0001 --epochs 50

# Evaluate on test set
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --config configs/custom_cnn.yaml

# Compare all evaluated models
python scripts/compare_models.py --results-dir outputs

# Run all experiments sequentially
./scripts/run_all_experiments.sh

# Run tests
python -m pytest tests/ -v
```

### Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_data_exploration.ipynb` | Class distribution, sample images, dataset statistics |
| `02_failure_analysis.ipynb` | Visualize misclassifications, confidence distribution |
| `03_learning_curves.ipynb` | Cross-model training/validation curve comparison |

## Git Workflow

We follow **Git Flow**:

- `main` — stable, release-ready code
- `develop` — integration branch for features
- `feature/<name>` — individual feature branches

Completed feature branches:
- `feature/data-pipeline` — dataset, transforms, config loader
- `feature/training-loop` — trainer, metrics, CLI scripts
- `feature/custom-cnn` — baseline CNN architecture
- `feature/finetuned-models` — MobileNetV2 + Xception
- `feature/experiments` — comparison tools, notebooks, ablation configs
- `feature/report` — report template and final README

## Trade-offs and Design Decisions

- **Official CelebA splits** over random splitting: ensures reproducibility and comparability with published results.
- **ImageNet normalization** for all models: consistent preprocessing, required by pretrained backbones.
- **BCEWithLogitsLoss** over CrossEntropyLoss: numerically more stable for binary classification with a single output neuron.
- **Backbone freezing** for finetuned models: prevents catastrophic forgetting of pretrained features during initial training of the new head.
- **timm for Xception** instead of custom implementation: Xception is not available in torchvision. timm is a widely-used, well-maintained library in the research community.

## Limitations

- **Binary classification only**: CelebA annotates gender as binary (`Male` attribute). This is a simplification that does not reflect real-world gender diversity. We adopt it strictly as a technical benchmark.
- **Dataset bias**: CelebA is biased toward celebrities, which may not generalize to other populations.
- **No face detection**: we assume pre-cropped, aligned faces (as provided by CelebA).
- **Single-phase finetuning**: backbone is frozen but we do not implement the two-phase unfreeze strategy in the Trainer (the `unfreeze_backbone()` method is available for manual use).

## Potential Improvements

- Two-phase finetuning with automated backbone unfreezing after N warmup epochs
- Multi-task learning (predict multiple CelebA attributes simultaneously)
- Grad-CAM or attention visualization for interpretability
- Cross-dataset evaluation (LFW, UTKFace) for generalization assessment
- More aggressive augmentation (mixup, cutout, random erasing)

## Team

- Marco Cosseddu
- [Colleague name]

## License

This project is developed for academic purposes as part of the DLA course at University of Cagliari.
