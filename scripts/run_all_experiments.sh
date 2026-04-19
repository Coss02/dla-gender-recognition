#!/bin/bash
# Run all model training experiments sequentially.
# Each model trains with its own config, results are evaluated on the test set.
#
# Usage:
#   chmod +x scripts/run_all_experiments.sh
#   ./scripts/run_all_experiments.sh

set -e

echo "=== Training Custom CNN ==="
python scripts/train.py --config configs/custom_cnn.yaml

echo ""
echo "=== Training MobileNetV2 ==="
python scripts/train.py --config configs/mobilenet_v2.yaml

echo ""
echo "=== Training Xception ==="
python scripts/train.py --config configs/xception.yaml

echo ""
echo "=== Evaluating all models ==="
for config in configs/custom_cnn.yaml configs/mobilenet_v2.yaml configs/xception.yaml; do
    echo "Evaluating with $config ..."
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --config "$config"
done

echo ""
echo "=== Generating comparison ==="
python scripts/compare_models.py --results-dir outputs

echo ""
echo "Done! Check the outputs/ directory for results."
