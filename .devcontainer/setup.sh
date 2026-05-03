#!/bin/bash
# =============================================================================
# .devcontainer/setup.sh
# One-time setup script - runs when devcontainer is first created
# =============================================================================

set -euo pipefail

echo "=========================================="
echo "SAR YOLO26 - One-Time Setup"
echo "=========================================="

# Update package list
echo "📦 Updating package list..."
apt-get update -qq

# Install system dependencies (curl and wget)
echo "📦 Installing system packages..."
apt-get install -y -qq curl wget

# Fix OpenCV conflicts as per your requirements
echo "🔧 Fixing OpenCV conflicts..."
pip uninstall -y opencv-python-headless thop || true

# Install required packages (opencv-python, polars, ultralytics-thop)
echo "🐍 Installing required Python packages..."
pip install --no-cache-dir \
    opencv-python \
    polars \
    ultralytics-thop

# Configure YOLO settings
echo "⚙️  Configuring YOLO settings..."
yolo settings wandb=true

# Configure WandB
echo "🔐 Configuring WandB..."
if [ ! -z "${WANDB_API_KEY:-}" ]; then
    echo "$WANDB_API_KEY" | wandb login --relogin
    echo "✅ WandB authentication configured"
else
    echo "⚠️  WANDB_API_KEY not set - manual login required"
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p /workspaces/SAR_YOLO26
mkdir -p /workspaces/SAR_YOLO26/runs/detect/logs_SAR_YOLO26
mkdir -p /workspaces/SAR_YOLO26/runs/detect/SAR_YOLO26
mkdir -p /workspaces/SAR_YOLO26/logs_SAR_YOLO26

# Make all shell scripts executable
echo "🔧 Making scripts executable..."
cd /workspaces/SAR_YOLO26
find . -type f -name "*.sh" -exec chmod +x {} \; 2>/dev/null || true

echo ""
echo "=========================================="
echo "✅ One-time setup complete!"
echo "=========================================="
echo ""
