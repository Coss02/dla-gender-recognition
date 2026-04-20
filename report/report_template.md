# Gender Recognition from Facial Images using CNNs

**Authors**: Marco Cosseddu, [Colleague Name]
**Course**: Deep Learning and Applications — University of Cagliari
**Date**: [Date]

---

## 1. Introduction

<!-- ~1 page -->

Gender recognition from facial images is a well-studied binary classification task
in computer vision. It serves as a fundamental benchmark for evaluating
convolutional neural network architectures on face-related problems.

<!-- TODO: Expand on why this problem matters (demographics analysis, HCI,
surveillance, accessibility). Discuss ethical considerations around binary
gender classification as a simplification. -->

The objective of this project is to compare three approaches:
1. A **custom CNN** trained from scratch (baseline)
2. A **finetuned MobileNetV2** pretrained on ImageNet
3. A **finetuned Xception** pretrained on ImageNet

We evaluate all models on the CelebA dataset using identical data splits,
training procedures, and evaluation metrics.

---

## 2. Data

<!-- ~1-2 pages. Use figures from notebooks/01_data_exploration.ipynb -->

### 2.1 CelebA Dataset

- Source: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
- Total images: ~200K celebrity face images
- Annotations: 40 binary attributes; we use the **Male** attribute
- Images are pre-cropped and aligned (178x218 pixels)

### 2.2 Data Split

We use the **official CelebA partition** (list_eval_partition.txt):

| Split | Samples | Purpose |
|-------|---------|---------|
| Train |         | Model training |
| Val   |         | Hyperparameter selection, early stopping |
| Test  |         | Final evaluation (reported metrics) |

<!-- TODO: Fill in actual sample counts from data exploration notebook -->

### 2.3 Class Distribution

<!-- TODO: Insert class_distribution.png from outputs/ -->
<!-- TODO: Note whether the dataset is balanced or not -->

### 2.4 Preprocessing and Augmentation

**Training transforms**:
- Resize to 224x224 (299x299 for Xception)
- Random horizontal flip (p=0.5)
- Random rotation (±10°)
- Color jitter (brightness=0.2, contrast=0.2, saturation=0.1)
- ImageNet normalization

**Validation/Test transforms**:
- Resize to 224x224 (299x299 for Xception)
- ImageNet normalization

---

## 3. Methods

<!-- ~2-3 pages -->

### 3.1 Custom CNN

<!-- TODO: Insert architecture diagram -->

A lightweight 4-block convolutional network:
- 4 blocks of Conv2d(3x3) → BatchNorm → ReLU → MaxPool(2x2)
- Channel progression: 3 → 32 → 64 → 128 → 256
- AdaptiveAvgPool(1) for spatial-dimension-agnostic pooling
- Classifier: Linear(256, 128) → ReLU → Dropout(0.3) → Linear(128, 1)
- Output: raw logit (trained with BCEWithLogitsLoss)

**Design rationale**: intentionally simple to serve as a lower bound.
BatchNorm for training stability, AdaptiveAvgPool for input-size flexibility.

### 3.2 MobileNetV2 (Finetuned)

- Backbone: torchvision MobileNetV2 pretrained on ImageNet
- Modified classifier: Dropout(0.3) → Linear(1280, 1)
- Finetuning strategy: backbone initially frozen, only classifier head trained

### 3.3 Xception (Finetuned)

- Backbone: timm legacy_xception pretrained on ImageNet
- Modified head: Dropout(0.3) → Linear(2048, 1)
- Input size: 299x299 (native Xception resolution)
- Finetuning strategy: same as MobileNetV2

### 3.4 Training Details

| Hyperparameter | Custom CNN | MobileNetV2 | Xception |
|----------------|-----------|-------------|----------|
| Learning rate  | 0.001     | 0.0001      | 0.0001   |
| Batch size     | 64        | 64          | 32       |
| Optimizer      | Adam      | Adam        | Adam     |
| Scheduler      | Cosine    | Cosine      | Cosine   |
| Max epochs     | 30        | 20          | 20       |
| Early stopping | 5 epochs  | 5 epochs    | 5 epochs |

Loss function: BCEWithLogitsLoss (binary cross-entropy with logits).

---

## 4. Experiments

<!-- ~3-4 pages. This is the core of the report. -->

### 4.1 Model Comparison

<!-- TODO: Insert model_comparison.png from outputs/ -->
<!-- TODO: Insert comparison table from model_comparison.csv -->

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| Custom CNN | | | | |
| MobileNetV2 | | | | |
| Xception | | | | |

### 4.2 Learning Curves

<!-- TODO: Insert learning curve plots generated from outputs/*_history.json -->
<!-- TODO: Discuss convergence speed differences between models -->
<!-- TODO: Note any overfitting (train loss << val loss) -->

### 4.3 Ablation Studies

#### Data Augmentation Impact
<!-- TODO: Compare custom_cnn with vs without augmentation -->

#### Learning Rate Sensitivity
<!-- TODO: Compare lr=0.001 vs lr=0.0001 for custom CNN -->

#### Dropout Sensitivity
<!-- TODO: Compare dropout=0.3 vs dropout=0.5 for custom CNN -->

### 4.4 Failure Case Analysis

<!-- TODO: Insert failure_cases.png from outputs/ -->
<!-- TODO: Discuss common patterns in misclassified images -->
<!-- TODO: Insert confidence_dist.png -->

### 4.5 Confusion Matrices

<!-- TODO: Insert confusion matrix for each model -->

---

## 5. Conclusion

<!-- ~0.5-1 page -->

### 5.1 Key Results

<!-- TODO: Summarize which model performed best and by how much -->
<!-- TODO: Highlight the benefit of transfer learning vs training from scratch -->

### 5.2 Limitations

- Binary gender classification is a simplification that does not reflect
  real-world gender diversity. We adopt it strictly as a technical benchmark.
- CelebA is biased toward celebrities, which may not generalize to other populations.
- No face detection pipeline: we assume pre-cropped, aligned faces.

### 5.3 Future Extensions

- Multi-task learning: predict multiple CelebA attributes simultaneously
- Grad-CAM visualization for model interpretability
- Cross-dataset evaluation (LFW, UTKFace) for generalization assessment
- More aggressive data augmentation (mixup, cutout)
- Two-phase finetuning: unfreeze backbone after warmup

---

## References

1. Liu, Z., Luo, P., Wang, X., & Tang, X. (2015). Deep Learning Face Attributes in the Wild. ICCV.
2. Sandler, M., et al. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. CVPR.
3. Chollet, F. (2017). Xception: Deep Learning with Depthwise Separable Convolutions. CVPR.
