#!/usr/bin/env bash
set -euo pipefail

echo ""
echo "======================================================"
echo "  Installing Python dependencies..."
echo "======================================================"
pip install --upgrade pip
pip install -r .devcontainer/requirements.txt

echo ""
echo "======================================================"
echo "  Pre-downloading YOLO26 model weights (n/s/m/l/x)..."
echo "======================================================"
python3 - <<'EOF'
from ultralytics import YOLO

for size in ["n", "s", "m", "l", "x"]:
    name = f"yolo26{size}.pt"
    print(f"  → Downloading {name} ...")
    YOLO(name)   # triggers auto-download if not cached
    print(f"    ✓ {name} ready")

print("\nAll YOLO26 weights downloaded successfully.")
EOF

echo ""
echo "======================================================"
echo "  Setup complete. Happy training!"
echo "======================================================"
