#!/bin/bash
# =============================================================================
# post_training_pipeline.sh
# Runs after each model training completes:
#   1. Prunes the model (removes optimizer state)
#   2. Evaluates on test set
#   3. Logs results to WandB
# =============================================================================

set -euo pipefail

# =============================================================================
# PARSE ARGUMENTS
# =============================================================================

if [ $# -lt 2 ]; then
    echo "Usage: $0 <best_pt_path> <model_variant_name>"
    echo ""
    echo "Example:"
    echo "  $0 runs/detect/SAR_YOLO26/YOLO26N_AdamW_base/weights/best.pt YOLO26N_AdamW_base"
    exit 1
fi

BEST_PT="$1"
MODEL_VARIANT="$2"
DATA_YAML="${3:-dataset.yaml}"

# =============================================================================
# SETUP
# =============================================================================

PRUNED_DIR="pruned_models"
EVAL_DIR="evaluation_results"

mkdir -p "${PRUNED_DIR}"
mkdir -p "${EVAL_DIR}"

# Generate pruned model name: YOLO26N_AdamW_base_best.pt
PRUNED_MODEL="${PRUNED_DIR}/${MODEL_VARIANT}_best.pt"

echo ""
echo "=========================================================================="
echo "POST-TRAINING PIPELINE: ${MODEL_VARIANT}"
echo "=========================================================================="
echo "Input:  ${BEST_PT}"
echo "Output: ${PRUNED_MODEL}"
echo "=========================================================================="
echo ""

# =============================================================================
# STEP 1: CHECK IF MODEL EXISTS
# =============================================================================

if [ ! -f "${BEST_PT}" ]; then
    echo "❌ ERROR: Model not found: ${BEST_PT}"
    exit 1
fi

echo "✅ Found model: ${BEST_PT}"

# Get original model size
ORIGINAL_SIZE=$(du -h "${BEST_PT}" | cut -f1)
echo "   Original size: ${ORIGINAL_SIZE}"

# =============================================================================
# STEP 2: PRUNE MODEL (Remove Optimizer State)
# =============================================================================

echo ""
echo "🔧 STEP 1/2: Pruning model..."
echo "--------------------------------------------------------------------------"

python3 prune_model.py \
    --model "${BEST_PT}" \
    --output "${PRUNED_MODEL}"

if [ ! -f "${PRUNED_MODEL}" ]; then
    echo "❌ ERROR: Pruning failed - output not created"
    exit 1
fi

PRUNED_SIZE=$(du -h "${PRUNED_MODEL}" | cut -f1)
echo ""
echo "✅ Pruning complete!"
echo "   Original: ${ORIGINAL_SIZE}"
echo "   Pruned:   ${PRUNED_SIZE}"
echo ""

# =============================================================================
# STEP 3: EVALUATE ON TEST SET
# =============================================================================

echo ""
echo "📊 STEP 2/2: Evaluating on test set..."
echo "--------------------------------------------------------------------------"

# Run evaluation with WandB logging
python3 evaluate_YOLO26_v2.py \
    --model "${PRUNED_MODEL}" \
    --data "${DATA_YAML}" \
    --split test \
    --batch 128 \
    --project SAR_YOLO26 \
    --run-name "${MODEL_VARIANT}_test_eval" \
    --model-variant "${MODEL_VARIANT}" \
    --wandb-mode online

EVAL_STATUS=$?

if [ ${EVAL_STATUS} -eq 0 ]; then
    echo ""
    echo "✅ Evaluation complete!"
else
    echo ""
    echo "⚠️  Evaluation completed with warnings (status: ${EVAL_STATUS})"
fi

# =============================================================================
# STEP 4: SUMMARY
# =============================================================================

echo ""
echo "=========================================================================="
echo "POST-TRAINING PIPELINE COMPLETE: ${MODEL_VARIANT}"
echo "=========================================================================="
echo "Pruned Model: ${PRUNED_MODEL}"
echo "Original:     ${ORIGINAL_SIZE} → Pruned: ${PRUNED_SIZE}"
echo "Test Metrics: Logged to WandB project 'SAR_YOLO26'"
echo "=========================================================================="
echo ""
