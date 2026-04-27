#!/usr/bin/env bash
# postCreateCommand — installs deps and pre-downloads YOLO26 weights

echo ""
echo "======================================================"
echo "  Installing Python dependencies..."
echo "======================================================"
pip install --upgrade pip

# torch/torchvision/torchaudio are already in the base image — skip them.
# Install everything else with verbose output so failures are visible.
pip install -v -r .devcontainer/requirements.txt

echo ""
echo "======================================================"
echo "  Verifying CUDA visibility..."
echo "======================================================"
python3 -c "
import torch
print(f'torch          : {torch.__version__}')
print(f'CUDA available : {torch.cuda.is_available()}')
print(f'GPU count      : {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

echo ""
echo "======================================================"
echo "  Pre-downloading YOLO26 model weights (n/s/m/l/x)..."
echo "======================================================"
python3 - <<'EOF'
from ultralytics import YOLO

for size in ["n", "s", "m", "l", "x"]:
    name = f"yolo26{size}.pt"
    print(f"  → Downloading {name} ...")
    try:
        YOLO(name)
        print(f"    ✓ {name} ready")
    except Exception as e:
        print(f"    ✗ {name} failed: {e}")

print("\nModel download step complete.")
EOF

echo ""
echo "======================================================"
echo "  Setup complete. Happy training!"
echo "======================================================"
