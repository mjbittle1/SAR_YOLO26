#!/usr/bin/env bash
# =============================================================================
#  run_SAR_YOLO26.sh
#  Wrapper script for SAR-optimized YOLO26 training pipeline
#
#  Usage:
#    chmod +x run_SAR_YOLO26.sh
#    ./run_SAR_YOLO26.sh
#
#  To run in background with logging:
#    nohup ./run_SAR_YOLO26.sh > sar_training.log 2>&1 &
#
#  To resume after crash:
#    Just run the script again - it auto-resumes from last.pt
# =============================================================================

set -euo pipefail

# Configuration
GPUS="0,1,2,3,4,5,6,7"
DATA="dataset.yaml"
EPOCHS=500
BATCH=128  # 16 per GPU on 8 GPUs - optimal for 500 epochs
PATIENCE=50
WARMUP=3.0

export CUDA_VISIBLE_DEVICES="${GPUS}"

# Create log directory
LOG_DIR="logs_SAR_YOLO26"
mkdir -p "${LOG_DIR}"

# Timestamp for this run
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
MAIN_LOG="${LOG_DIR}/pipeline_${TIMESTAMP}.log"

echo "=================================================================="
echo "  SAR-OPTIMIZED YOLO26 TRAINING PIPELINE"
echo "=================================================================="
echo "  Start Time : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  GPUs       : ${GPUS}"
echo "  Dataset    : ${DATA}"
echo "  Epochs     : ${EPOCHS}"
echo "  Batch Size : ${BATCH}"
echo "  Patience   : ${PATIENCE}"
echo "  Warmup     : ${WARMUP}"
echo "  Main Log   : ${MAIN_LOG}"
echo "=================================================================="
echo ""

# Run the training pipeline
python3 train_SAR_YOLO26.py \
    --data "${DATA}" \
    --epochs "${EPOCHS}" \
    --batch "${BATCH}" \
    --patience "${PATIENCE}" \
    --device "${GPUS}" \
    --warmup-epochs "${WARMUP}" \
    2>&1 | tee "${MAIN_LOG}"

echo ""
echo "=================================================================="
echo "  Pipeline Complete : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Check results in  : runs/detect/SAR_YOLO26/"
echo "  Check WandB at    : https://wandb.ai/your-username/SAR_YOLO26"
echo "=================================================================="
