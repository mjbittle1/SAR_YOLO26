# Installation Guide for /workspaces/SAR_YOLO26

## ✅ Corrected File Structure

```
/workspaces/SAR_YOLO26/                  # Your actual project root
│
├── .devcontainer/                       # DevContainer files
│   ├── devcontainer.json                # ✅ UPDATED for correct path
│   ├── Dockerfile                       # ✅ UPDATED for correct path
│   ├── setup.sh                         # ✅ UPDATED for correct path
│   └── startup.sh                       # ✅ UPDATED for correct path
│
├── train_SAR_YOLO26.py                  # Training script
├── run_SAR_YOLO26.sh                    # Wrapper script
├── post_training_pipeline.sh            # Pruning + evaluation
├── evaluate_YOLO26_v2.py                # Enhanced evaluation
├── prune_model.py                       # Your existing file
│
├── dataset.yaml                         # Your existing dataset
│
├── pruned_models/                       # Auto-created
├── logs_SAR_YOLO26/                     # Auto-created
└── runs/                                # Auto-created
    └── detect/
        └── SAR_YOLO26/
```

---

## 📥 Files to Download (All Paths Corrected)

### DevContainer Files (4 files) - ✅ RE-DOWNLOAD UPDATED VERSIONS
1. **devcontainer.json** - Updated with workspaceFolder: `/workspaces/SAR_YOLO26`
2. **Dockerfile** - Updated with WORKDIR: `/workspaces/SAR_YOLO26`
3. **setup.sh** - Updated with correct paths
4. **startup.sh** - Updated with correct paths

### Training Scripts (4 files) - No changes needed
5. **train_SAR_YOLO26.py** - No path changes needed
6. **run_SAR_YOLO26.sh** - No path changes needed
7. **post_training_pipeline.sh** - No path changes needed
8. **evaluate_YOLO26_v2.py** - No path changes needed

---

## 🚀 Quick Installation

### Step 1: Place Files

```bash
cd /workspaces/SAR_YOLO26

# DevContainer files
mkdir -p .devcontainer
# Upload these to .devcontainer/:
#   - devcontainer.json (RE-DOWNLOAD)
#   - Dockerfile (RE-DOWNLOAD)
#   - setup.sh (RE-DOWNLOAD)
#   - startup.sh (RE-DOWNLOAD)

# Training scripts (place in root)
# Upload these to /workspaces/SAR_YOLO26/:
#   - train_SAR_YOLO26.py
#   - run_SAR_YOLO26.sh
#   - post_training_pipeline.sh
#   - evaluate_YOLO26_v2.py
```

### Step 2: Make Scripts Executable

```bash
cd /workspaces/SAR_YOLO26

chmod +x run_SAR_YOLO26.sh
chmod +x post_training_pipeline.sh
chmod +x .devcontainer/setup.sh
chmod +x .devcontainer/startup.sh
```

### Step 3: Rebuild Container

In VS Code:
1. **Ctrl+Shift+P**
2. "Dev Containers: Rebuild Container"
3. Wait for build completion

### Step 4: Verify Setup

After container rebuilds, check terminal output:

```
==========================================
SAR YOLO26 - Container Startup
==========================================
...
ℹ️  No training scripts found
   Place training script at /workspaces/SAR_YOLO26/run_SAR_YOLO26.sh
==========================================
```

If you see the correct path (`/workspaces/SAR_YOLO26/`), you're good!

### Step 5: Start Training

```bash
cd /workspaces/SAR_YOLO26
./run_SAR_YOLO26.sh
```

---

## ✅ Verification Script

Run this to verify everything:

```bash
#!/bin/bash
cd /workspaces/SAR_YOLO26

echo "=========================================="
echo "File Verification for /workspaces/SAR_YOLO26"
echo "=========================================="

files=(
    ".devcontainer/devcontainer.json"
    ".devcontainer/Dockerfile"
    ".devcontainer/setup.sh"
    ".devcontainer/startup.sh"
    "train_SAR_YOLO26.py"
    "run_SAR_YOLO26.sh"
    "post_training_pipeline.sh"
    "evaluate_YOLO26_v2.py"
    "prune_model.py"
    "dataset.yaml"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ MISSING: $file"
    fi
done

echo ""
echo "Checking executables..."
for script in run_SAR_YOLO26.sh post_training_pipeline.sh .devcontainer/setup.sh .devcontainer/startup.sh; do
    if [ -x "$script" ]; then
        echo "✅ $script (executable)"
    else
        echo "⚠️  $script (run: chmod +x $script)"
    fi
done

echo "=========================================="
```

---

## 🔍 What Changed

**Updated these 4 files to use `/workspaces/SAR_YOLO26`:**

1. **devcontainer.json**
   - Added `"workspaceFolder": "/workspaces/SAR_YOLO26"`

2. **Dockerfile**
   - Changed `WORKDIR /workspace` → `WORKDIR /workspaces/SAR_YOLO26`

3. **setup.sh**
   - Changed all `/workspace/` paths → `/workspaces/SAR_YOLO26/`

4. **startup.sh**
   - Changed all `/workspace/` paths → `/workspaces/SAR_YOLO26/`

**No changes needed for training scripts** - they use relative paths.

---

## ⚠️ Important

**RE-DOWNLOAD the 4 DevContainer files** - they've been updated with correct paths!

The training scripts (train_SAR_YOLO26.py, run_SAR_YOLO26.sh, etc.) don't need updates - they work from any directory.
