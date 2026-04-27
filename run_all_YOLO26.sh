#!/usr/bin/env bash
# =============================================================================
#  run_all_YOLO26.sh  —  Full pipeline: Train → Tune → Evaluate (per model)
#  Single-GPU mode (one NVIDIA GPU, device 0)
#
#  For each model size (n → s → m → l → x), all three phases run to completion
#  before the next size begins:
#
#    n:  Train → Tune → Eval(trained) → Eval(tuned)
#    s:  Train → Tune → Eval(trained) → Eval(tuned)
#    m:  Train → Tune → Eval(trained) → Eval(tuned)
#    l:  Train → Tune → Eval(trained) → Eval(tuned)
#    x:  Train → Tune → Eval(trained) → Eval(tuned)
#
#  All logs are written to: runs/detect/logs/
#  On completion the entire runs/ folder is archived to: runs_YYYYMMDD_HHMMSS.tar.gz
#
#  Usage:
#    chmod +x run_all_YOLO26.sh
#    nohup ./run_all_YOLO26.sh > runs/detect/logs/pipeline.log 2>&1 &
# =============================================================================

set -euo pipefail          # exit on any error, treat unset vars as errors

GPUS="0"
DATA="dataset.yaml"

# Pin to the single GPU for the entire session
export CUDA_VISIBLE_DEVICES="${GPUS}"

# Root directory where Ultralytics writes run outputs
# Pattern: ${RUNS_DIR}/{NAME}/weights/best.pt
RUNS_DIR="runs/detect/SARDet_100KYOLO26"

# All log files land here
LOG_DIR="runs/detect/logs"
mkdir -p "${LOG_DIR}"

# Model sizes to process — smallest to largest
SIZES=(n s m l x)

# ---------------------------------------------------------------------------
# Helper: run a plain python job (single GPU) and block until it finishes
# ---------------------------------------------------------------------------
run_python() {
    local LOG="${LOG_DIR}/$1"; shift
    echo ""
    echo "=================================================================="
    echo "  START : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  CMD   : python3 $*"
    echo "  LOG   : ${LOG}"
    echo "=================================================================="

    python3 "$@" >> "${LOG}" 2>&1

    echo "=================================================================="
    echo "  END   : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  DONE  : ${LOG}"
    echo "=================================================================="
}

# ===========================================================================
#  MAIN LOOP — one model size at a time: Train → Tune → Eval
# ===========================================================================
for SIZE in "${SIZES[@]}"; do
    SIZE_UPPER="${SIZE^^}"
    TRAIN_WEIGHTS="${RUNS_DIR}/YOLO26${SIZE_UPPER}/weights/best.pt"
    TUNED_WEIGHTS="${RUNS_DIR}/YOLO26${SIZE_UPPER}_tune/weights/best.pt"

    echo ""
    echo "######################################################################"
    echo "#  YOLO26${SIZE_UPPER} — START  ($(date '+%Y-%m-%d %H:%M:%S'))"
    echo "######################################################################"

    # ── 1. TRAIN ──────────────────────────────────────────────────────────────
    echo ""
    echo ">>> [YOLO26${SIZE_UPPER}] Phase 1: Training"
    run_python "YOLO26${SIZE_UPPER}_train.log" \
        train_YOLO26.py \
        --size "${SIZE}" \
        --data "${DATA}" \
        --device "${GPUS}" \
        --cache

    # ── 2. TUNE ───────────────────────────────────────────────────────────────
    echo ""
    echo ">>> [YOLO26${SIZE_UPPER}] Phase 2: Tuning"
    if [[ -f "${TRAIN_WEIGHTS}" ]]; then
        run_python "YOLO26${SIZE_UPPER}_tune.log" \
            tune_YOLO26.py \
            --weights "${TRAIN_WEIGHTS}" \
            --name    "YOLO26${SIZE_UPPER}_tune" \
            --data    "${DATA}" \
            --device  "${GPUS}" \
            --cache
    else
        echo "WARNING: trained checkpoint not found — ${TRAIN_WEIGHTS}. Skipping tune for YOLO26${SIZE_UPPER}."
    fi

    # ── 3. EVALUATE trained model ─────────────────────────────────────────────
    echo ""
    echo ">>> [YOLO26${SIZE_UPPER}] Phase 3a: Evaluating trained model"
    if [[ -f "${TRAIN_WEIGHTS}" ]]; then
        run_python "YOLO26${SIZE_UPPER}_train_eval.log" \
            evaluate_YOLO26.py \
            --model  "${TRAIN_WEIGHTS}" \
            --data   "${DATA}" \
            --split  test
    else
        echo "WARNING: trained checkpoint not found — ${TRAIN_WEIGHTS}. Skipping train eval for YOLO26${SIZE_UPPER}."
    fi

    # ── 4. EVALUATE tuned model ───────────────────────────────────────────────
    echo ""
    echo ">>> [YOLO26${SIZE_UPPER}] Phase 3b: Evaluating tuned model"
    if [[ -f "${TUNED_WEIGHTS}" ]]; then
        run_python "YOLO26${SIZE_UPPER}_tune_eval.log" \
            evaluate_YOLO26.py \
            --model  "${TUNED_WEIGHTS}" \
            --data   "${DATA}" \
            --split  test
    else
        echo "WARNING: tuned checkpoint not found — ${TUNED_WEIGHTS}. Skipping tune eval for YOLO26${SIZE_UPPER}."
    fi

    echo ""
    echo "######################################################################"
    echo "#  YOLO26${SIZE_UPPER} — COMPLETE  ($(date '+%Y-%m-%d %H:%M:%S'))"
    echo "######################################################################"
done

# ===========================================================================
#  ARCHIVE — tar the entire runs/ folder for download
# ===========================================================================
ARCHIVE="runs_$(date '+%Y%m%d_%H%M%S').tar.gz"
echo ""
echo "######################################################################"
echo "#  ALL MODELS COMPLETE  ($(date '+%Y-%m-%d %H:%M:%S'))"
echo "#  Archiving runs/ → ${ARCHIVE} ..."
echo "######################################################################"

tar -czf "${ARCHIVE}" runs/

echo ""
echo ">>> Archive created: ${ARCHIVE}  ($(date '+%Y-%m-%d %H:%M:%S'))"
echo ">>> Done. Download ${ARCHIVE} to retrieve all results and logs."
