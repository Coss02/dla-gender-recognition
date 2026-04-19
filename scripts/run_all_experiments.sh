#!/bin/bash
# Full experiment pipeline: download data, train all models, evaluate, compare.
#
# Prerequisites:
#   - Virtual environment activated with all dependencies installed
#   - Kaggle API credentials at ~/.kaggle/kaggle.json (for dataset download)
#   - wandb login (for experiment tracking)
#
# Usage:
#   chmod +x scripts/run_all_experiments.sh
#   ./scripts/run_all_experiments.sh

set -e

CHECKPOINT_DIR="checkpoints"
OUTPUT_DIR="outputs"
mkdir -p "$CHECKPOINT_DIR" "$OUTPUT_DIR"

echo "=== Step 1/6: Verifying dataset ==="
python scripts/download_celeba.py

echo ""
echo "=== Step 2/6: Training Custom CNN ==="
python scripts/train.py --config configs/custom_cnn.yaml
cp "$CHECKPOINT_DIR/best_model.pt" "$CHECKPOINT_DIR/custom_cnn_best.pt"

echo ""
echo "=== Step 3/6: Training MobileNetV2 ==="
python scripts/train.py --config configs/mobilenet_v2.yaml
cp "$CHECKPOINT_DIR/best_model.pt" "$CHECKPOINT_DIR/mobilenet_v2_best.pt"

echo ""
echo "=== Step 4/6: Training Xception ==="
python scripts/train.py --config configs/xception.yaml
cp "$CHECKPOINT_DIR/best_model.pt" "$CHECKPOINT_DIR/xception_best.pt"

echo ""
echo "=== Step 5/6: Evaluating all models on test set ==="
python scripts/evaluate.py --checkpoint "$CHECKPOINT_DIR/custom_cnn_best.pt" --config configs/custom_cnn.yaml
python scripts/evaluate.py --checkpoint "$CHECKPOINT_DIR/mobilenet_v2_best.pt" --config configs/mobilenet_v2.yaml
python scripts/evaluate.py --checkpoint "$CHECKPOINT_DIR/xception_best.pt" --config configs/xception.yaml

echo ""
echo "=== Step 6/6: Generating comparison ==="
python scripts/compare_models.py --results-dir "$OUTPUT_DIR"

echo ""
echo "=========================================="
echo "  All experiments completed!"
echo "=========================================="
echo ""
echo "Results:"
echo "  - Checkpoints:       $CHECKPOINT_DIR/"
echo "  - Test metrics:      $OUTPUT_DIR/*_test_results.json"
echo "  - Confusion matrices: $OUTPUT_DIR/*_confusion_matrix.png"
echo "  - Comparison table:  $OUTPUT_DIR/model_comparison.csv"
echo "  - Comparison chart:  $OUTPUT_DIR/model_comparison.png"
echo "  - Training histories: $OUTPUT_DIR/*_history.json"
echo ""
echo "Next steps:"
echo "  1. Open notebooks/02_failure_analysis.ipynb for error analysis"
echo "  2. Open notebooks/03_learning_curves.ipynb for training curves"
echo "  3. Check wandb dashboard for detailed metrics"
