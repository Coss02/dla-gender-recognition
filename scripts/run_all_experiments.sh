#!/usr/bin/env bash
# Full experiment pipeline: download data, train all models, evaluate, compare.
#
# Prerequisites:
#   - Virtual environment activated with all dependencies installed
#   - Kaggle API credentials at ~/.kaggle/kaggle.json (for dataset download)
#
# Usage:
#   chmod +x scripts/run_all_experiments.sh
#   ./scripts/run_all_experiments.sh
#   ./scripts/run_all_experiments.sh custom_cnn mobilenet_v2
#   MODELS="custom_cnn mobilenet_v2" ./scripts/run_all_experiments.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="$ROOT_DIR/data/celeba"
LEGACY_DATA_DIR="$ROOT_DIR/scripts/data/celeba"
CHECKPOINT_DIR="$ROOT_DIR/checkpoints"
OUTPUT_DIR="$ROOT_DIR/outputs"
RESULTS_DIR="$OUTPUT_DIR"
MPLCONFIGDIR="${MPLCONFIGDIR:-$OUTPUT_DIR/.matplotlib}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-$OUTPUT_DIR/.cache}"

mkdir -p "$ROOT_DIR/data" "$CHECKPOINT_DIR" "$OUTPUT_DIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
export MPLCONFIGDIR
export XDG_CACHE_HOME

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python interpreter not found: $PYTHON_BIN"
    echo "Set PYTHON_BIN to the correct executable and retry."
    exit 1
fi

if [ ! -e "$DATA_DIR" ] && [ -d "$LEGACY_DATA_DIR" ]; then
    echo "Found existing dataset in scripts/data/celeba; linking it into data/celeba"
    ln -s "$LEGACY_DATA_DIR" "$DATA_DIR"
fi

if [ "$#" -gt 0 ]; then
    MODELS=("$@")
elif [ -n "${MODELS:-}" ]; then
    read -r -a MODELS <<< "$MODELS"
else
    MODELS=("custom_cnn" "mobilenet_v2" "xception")
fi

get_config_path() {
    case "$1" in
        custom_cnn) echo "configs/custom_cnn.yaml" ;;
        mobilenet_v2) echo "configs/mobilenet_v2.yaml" ;;
        xception) echo "configs/xception.yaml" ;;
        *)
            echo "Unknown model: $1" >&2
            return 1
            ;;
    esac
}

ensure_checkpoint() {
    local model="$1"
    local config_path
    local model_checkpoint

    config_path="$(get_config_path "$model")"
    model_checkpoint="$CHECKPOINT_DIR/${model}_best.pt"

    if [ -f "$model_checkpoint" ]; then
        echo "=== Training: $model ==="
        echo "Existing checkpoint found at $model_checkpoint; skipping training"
        return 0
    fi

    echo "=== Training: $model ==="
    "$PYTHON_BIN" scripts/train.py --config "$config_path"

    if [ ! -f "$CHECKPOINT_DIR/best_model.pt" ]; then
        echo "Training completed but $CHECKPOINT_DIR/best_model.pt was not created" >&2
        exit 1
    fi

    cp "$CHECKPOINT_DIR/best_model.pt" "$model_checkpoint"
    echo "Saved model-specific checkpoint to $model_checkpoint"
}

evaluate_model() {
    local model="$1"
    local config_path
    local model_checkpoint

    config_path="$(get_config_path "$model")"
    model_checkpoint="$CHECKPOINT_DIR/${model}_best.pt"

    if [ ! -f "$model_checkpoint" ]; then
        echo "Checkpoint missing for $model: $model_checkpoint" >&2
        exit 1
    fi

    echo "=== Evaluating: $model ==="
    "$PYTHON_BIN" scripts/evaluate.py \
        --checkpoint "$model_checkpoint" \
        --config "$config_path" \
        --output-dir "$RESULTS_DIR" \
        --num-workers 0
}

cd "$ROOT_DIR"

echo "Selected models: ${MODELS[*]}"
echo "=== Step 1/5: Verifying dataset ==="
"$PYTHON_BIN" scripts/download_celeba.py --data-dir "$DATA_DIR"

echo ""
echo "=== Step 2/5: Training requested models ==="
for model in "${MODELS[@]}"; do
    ensure_checkpoint "$model"
done

echo ""
echo "=== Step 3/5: Evaluating requested models on test set ==="
for model in "${MODELS[@]}"; do
    evaluate_model "$model"
done

echo ""
echo "=== Step 4/5: Generating evaluation comparison ==="
"$PYTHON_BIN" scripts/compare_models.py --results-dir "$RESULTS_DIR"

echo ""
echo "=== Step 5/5: Generating training plots ==="
"$PYTHON_BIN" scripts/plot_training_curves.py --output-dir "$OUTPUT_DIR" --models "${MODELS[@]}"

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
echo "  - Training plots:    $OUTPUT_DIR/learning_curves_*.png"
echo ""
echo "Next steps:"
echo "  1. Open notebooks/02_failure_analysis.ipynb for error analysis"
echo "  2. Open notebooks/03_learning_curves.ipynb for training curves"
echo "  3. Inspect outputs/*_history.json for per-epoch metrics"
