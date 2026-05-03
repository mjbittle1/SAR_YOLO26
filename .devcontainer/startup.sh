#!/bin/bash
# =============================================================================
# .devcontainer/startup.sh
# Runs every time the devcontainer starts (including after server restart)
# Automatically resumes training if it was interrupted
# =============================================================================

set -euo pipefail

echo ""
echo "=========================================="
echo "SAR YOLO26 - Container Startup"
echo "=========================================="
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# =============================================================================
# STEP 1: Verify /dev/shm mount
# =============================================================================

echo "🔍 Checking /dev/shm mount..."
SHM_SIZE=$(df -h /dev/shm 2>/dev/null | tail -1 | awk '{print $2}' || echo "unknown")
echo "   /dev/shm size: ${SHM_SIZE}"

if [[ "${SHM_SIZE}" == "64M" ]] || [[ "${SHM_SIZE}" == "65536" ]]; then
    echo "⚠️  /dev/shm is too small (${SHM_SIZE}), attempting remount..."
    
    if mount -t tmpfs -o size=1024g tmpfs /dev/shm 2>/dev/null; then
        echo "✅ Remounted /dev/shm as 1TB"
    else
        echo "⚠️  Could not remount (may already be handled by devcontainer)"
    fi
else
    echo "✅ /dev/shm size looks good: ${SHM_SIZE}"
fi

# Run ldconfig as specified in your original config
ldconfig 2>/dev/null || true

# =============================================================================
# STEP 2: Verify WandB authentication
# =============================================================================

echo ""
echo "🔍 Checking WandB authentication..."
if wandb status &>/dev/null; then
    WANDB_USER=$(wandb status 2>/dev/null | grep "Logged in as" | cut -d: -f2 | xargs || echo "unknown")
    echo "✅ WandB authenticated as: ${WANDB_USER}"
else
    echo "⚠️  WandB not authenticated, attempting login..."
    if [ ! -z "${WANDB_API_KEY:-}" ]; then
        echo "$WANDB_API_KEY" | wandb login --relogin &>/dev/null && \
            echo "✅ WandB login successful" || \
            echo "❌ WandB login failed"
    else
        echo "❌ WANDB_API_KEY not set"
    fi
fi

# =============================================================================
# STEP 3: Verify GPU availability
# =============================================================================

echo ""
echo "🔍 Checking GPU availability..."
if command -v nvidia-smi &>/dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 0)
    echo "✅ Found ${GPU_COUNT} GPUs"
    
    if [[ ${GPU_COUNT} -lt 8 ]] && [[ ${GPU_COUNT} -gt 0 ]]; then
        echo "⚠️  Warning: Expected 8 GPUs, found ${GPU_COUNT}"
    fi
else
    echo "❌ nvidia-smi not available - GPU detection failed"
fi

# =============================================================================
# STEP 4: Check for running training processes
# =============================================================================

echo ""
echo "🔍 Checking for running training processes..."

TRAINING_RUNNING=false
TRAINING_SCRIPT=""

# Check for train_SAR_YOLO26.py (new script)
if pgrep -f "train_SAR_YOLO26.py" > /dev/null 2>&1; then
    echo "✅ train_SAR_YOLO26.py already running"
    TRAINING_RUNNING=true
    TRAINING_SCRIPT="train_SAR_YOLO26.py"
fi

# Check for old script (run_finish_YOLO26_v2.sh)
if pgrep -f "run_finish_YOLO26_v2.sh" > /dev/null 2>&1; then
    echo "✅ run_finish_YOLO26_v2.sh already running"
    TRAINING_RUNNING=true
    TRAINING_SCRIPT="run_finish_YOLO26_v2.sh"
fi

# Check for run_SAR_YOLO26.sh (new script)
if pgrep -f "run_SAR_YOLO26.sh" > /dev/null 2>&1; then
    echo "✅ run_SAR_YOLO26.sh already running"
    TRAINING_RUNNING=true
    TRAINING_SCRIPT="run_SAR_YOLO26.sh"
fi

# =============================================================================
# STEP 5: Auto-resume training if not running
# =============================================================================

if [ "$TRAINING_RUNNING" = false ]; then
    echo ""
    echo "🔍 No training process found. Checking for resumable training..."
    
    # Check if new SAR_YOLO26 training exists
    if [ -f "/workspaces/SAR_YOLO26/run_SAR_YOLO26.sh" ]; then
        echo "📊 Found SAR YOLO26 training script"
        
        # Check for any last.pt checkpoints
        CHECKPOINT_COUNT=$(find /workspaces/SAR_YOLO26/runs/detect/SAR_YOLO26 -name "last.pt" 2>/dev/null | wc -l || echo 0)
        
        if [ ${CHECKPOINT_COUNT} -gt 0 ]; then
            echo "✅ Found ${CHECKPOINT_COUNT} resumable checkpoint(s)"
            echo ""
            echo "🚀 AUTO-RESUMING TRAINING..."
            echo "=========================================="
            
            cd /workspaces/SAR_YOLO26
            chmod +x run_SAR_YOLO26.sh
            
            # Start training in background with setsid
            setsid ./run_SAR_YOLO26.sh > logs_SAR_YOLO26/auto_resume_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
            
            sleep 2
            
            echo "✅ Training resumed successfully"
            echo "   PID: $!"
            echo "   Log: logs_SAR_YOLO26/auto_resume_$(date +%Y%m%d_%H%M%S).log"
        else
            echo "ℹ️  No checkpoints found - training not started"
            echo "   Run './run_SAR_YOLO26.sh' manually to start fresh training"
        fi
    
    # Check for old Bittle training script
    elif [ -f "/workspaces/SAR_YOLO26/SAR_YOLO26/run_finish_YOLO26_v2.sh" ]; then
        echo "📊 Found old Bittle SAR YOLO26 training script"
        
        # Check for checkpoints in old location
        CHECKPOINT_COUNT=$(find /workspaces/SAR_YOLO26/runs/detect/Bittle_SAR_YOLO26 -name "last.pt" 2>/dev/null | wc -l || echo 0)
        
        if [ ${CHECKPOINT_COUNT} -gt 0 ]; then
            echo "✅ Found ${CHECKPOINT_COUNT} resumable checkpoint(s) in old training"
            echo ""
            echo "🚀 AUTO-RESUMING OLD TRAINING..."
            echo "=========================================="
            
            cd /workspaces/SAR_YOLO26/SAR_YOLO26
            chmod +x run_finish_YOLO26_v2.sh
            
            # Start training in background with setsid
            setsid ./run_finish_YOLO26_v2.sh > /workspaces/SAR_YOLO26/runs/detect/logs_Bittle_SAR_YOLO26/auto_resume_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
            
            sleep 2
            
            echo "✅ Old training resumed successfully"
            echo "   PID: $!"
        else
            echo "ℹ️  No old checkpoints found"
        fi
    else
        echo "ℹ️  No training scripts found"
        echo "   Place training script at /workspaces/SAR_YOLO26/run_SAR_YOLO26.sh"
    fi
else
    echo ""
    echo "✅ Training already running: ${TRAINING_SCRIPT}"
    echo ""
    echo "📊 Active training processes:"
    ps aux | grep -E "train_SAR_YOLO26|run_SAR_YOLO26|run_finish_YOLO26" | grep -v grep | head -5 || echo "   (Process details not available)"
fi

# =============================================================================
# STEP 6: Show quick status summary
# =============================================================================

echo ""
echo "=========================================="
echo "📋 STATUS SUMMARY"
echo "=========================================="
echo "WandB:    $(wandb status 2>/dev/null | grep "Logged in" || echo 'Not logged in')"
echo "GPUs:     ${GPU_COUNT:-0} detected"
echo "/dev/shm: ${SHM_SIZE}"
echo "Training: $([ "$TRAINING_RUNNING" = true ] && echo '✅ Running' || echo '⏸️  Not running')"
echo ""

# Show helpful commands
if [ "$TRAINING_RUNNING" = true ]; then
    echo "📝 USEFUL COMMANDS:"
    echo "   Monitor logs:  tail -f logs_SAR_YOLO26/*.log"
    echo "   WandB:         wandb dashboard SAR_YOLO26"
    echo "   Stop training: pkill -f train_SAR_YOLO26"
else
    echo "📝 TO START TRAINING:"
    echo "   cd /workspace"
    echo "   ./run_SAR_YOLO26.sh"
fi

echo "=========================================="
echo ""
