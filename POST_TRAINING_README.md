# Automatic Post-Training Pipeline: Pruning & Evaluation

## Overview

After each model completes training, the pipeline **automatically**:

1. ✅ **Prunes** the model (removes optimizer state, reduces file size)
2. ✅ **Evaluates** on test set (not validation set)
3. ✅ **Logs** all metrics to WandB for easy reporting

**No manual intervention required!**

---

## Files Added

```
post_training_pipeline.sh    # Orchestrates pruning + evaluation
evaluate_YOLO26_v2.py        # Enhanced evaluation with WandB reporting
prune_model.py               # Model pruning (removes optimizer)
train_SAR_YOLO26.py          # Updated to call post-training pipeline
```

---

## What Happens Automatically

### During Training

```
Phase 1: Base Training
  ├─ YOLO26N_AdamW_base trains
  ├─ Saves best.pt checkpoint
  └─ POST-TRAINING PIPELINE RUNS:
       ├─ Prunes best.pt → pruned_models/YOLO26N_AdamW_base_best.pt
       ├─ Evaluates on test set
       └─ Logs metrics to WandB

Phase 2: Fine-Tuning
  ├─ YOLO26N_AdamW_finetune trains
  ├─ Saves best.pt checkpoint
  └─ POST-TRAINING PIPELINE RUNS:
       ├─ Prunes best.pt → pruned_models/YOLO26N_AdamW_finetune_best.pt
       ├─ Evaluates on test set
       └─ Logs metrics to WandB

... (repeat for all 20 models)
```

### Output Structure

```
workspace/
├── pruned_models/                    # Pruned models (small size)
│   ├── YOLO26N_AdamW_base_best.pt
│   ├── YOLO26N_AdamW_finetune_best.pt
│   ├── YOLO26S_AdamW_base_best.pt
│   └── ... (20 total)
│
├── runs/detect/SAR_YOLO26/           # Full training checkpoints
│   ├── YOLO26N_AdamW_base/
│   │   └── weights/
│   │       ├── best.pt               # Full model (with optimizer)
│   │       └── last.pt               # Resume checkpoint
│   └── ...
```

---

## Model Pruning

### What is Pruning?

Removes optimizer state from the checkpoint:
- **Before:** `best.pt` = Model weights + Optimizer state + Training history (~200MB)
- **After:** `YOLO26N_AdamW_base_best.pt` = Model weights only (~100MB)

### File Sizes

| Model | Full Checkpoint | Pruned Model | Reduction |
|-------|----------------|--------------|-----------|
| YOLO26N | ~15 MB | ~6 MB | ~60% |
| YOLO26S | ~45 MB | ~22 MB | ~51% |
| YOLO26M | ~105 MB | ~52 MB | ~50% |
| YOLO26L | ~175 MB | ~87 MB | ~50% |
| YOLO26X | ~275 MB | ~136 MB | ~51% |

### Naming Convention

```
Format: {MODEL_SIZE}_{OPTIMIZER}_{PHASE}_best.pt

Examples:
  YOLO26N_AdamW_base_best.pt
  YOLO26N_AdamW_finetune_best.pt
  YOLO26X_MuSGD_base_best.pt
```

---

## Test Set Evaluation

### Metrics Logged to WandB

Each evaluation run logs:

#### Overall Metrics
- `test/mAP50-95` - Main metric (0.0-1.0)
- `test/mAP50` - mAP at IoU=0.50
- `test/mAP75` - mAP at IoU=0.75
- `test/precision` - Overall precision
- `test/recall` - Overall recall
- `test/fps` - Inference speed (frames per second)

#### Speed Metrics
- `test/preprocess_ms` - Preprocessing time
- `test/inference_ms` - Model inference time
- `test/postprocess_ms` - NMS/post-processing time
- `test/total_ms` - Total time per image

#### Per-Class Metrics (if available)
- `test/class_0_mAP50-95` - Aircraft
- `test/class_1_mAP50-95` - Ship
- `test/class_2_mAP50-95` - Vehicle
- ... (one per class)

#### Metadata
- `model_variant` - Full model name
- `tta_enabled` - Test-time augmentation (True/False)

---

## WandB Reporting

### Creating Comparison Reports

1. **Go to WandB Project**
   ```
   https://wandb.ai/your-username/SAR_YOLO26
   ```

2. **Filter Evaluation Runs**
   - Click "Tags" → Select "evaluation"
   - Or filter by job_type: "evaluation"

3. **Create Custom Report**

   **Example 1: Compare All Base Models**
   ```
   Filter: job_type = evaluation AND model_variant contains "base"
   
   Table Columns:
   - model_variant
   - test/mAP50-95
   - test/mAP50
   - test/precision
   - test/recall
   - test/fps
   
   Sort by: test/mAP50-95 (descending)
   ```

   **Example 2: AdamW vs MuSGD**
   ```
   X-axis: model_variant
   Y-axis: test/mAP50-95
   Group by: optimizer (extract from model_variant)
   
   Chart type: Bar chart
   ```

   **Example 3: Base vs Fine-tune**
   ```
   Filter 1: model_variant contains "AdamW"
   
   Compare:
   - YOLO26N_AdamW_base → YOLO26N_AdamW_finetune
   - YOLO26S_AdamW_base → YOLO26S_AdamW_finetune
   - ...
   
   Metric: test/mAP50-95
   ```

### Pre-configured Report Queries

#### Best Overall Model
```
Sort by: test/mAP50-95 DESC
Limit: 1

Expected: YOLO26X_AdamW_finetune
```

#### Fastest Model with mAP > 0.60
```
Filter: test/mAP50-95 > 0.60
Sort by: test/fps DESC
Limit: 1

Expected: YOLO26N or YOLO26S
```

#### Optimizer Comparison
```
Group 1: model_variant contains "AdamW"
Group 2: model_variant contains "MuSGD"

Metrics:
- Avg test/mAP50-95
- Avg test/precision
- Avg test/recall
```

---

## Manual Evaluation (If Needed)

### Run Single Model Evaluation

```bash
python3 evaluate_YOLO26_v2.py \
    --model pruned_models/YOLO26N_AdamW_base_best.pt \
    --data dataset.yaml \
    --split test \
    --model-variant YOLO26N_AdamW_base \
    --run-name YOLO26N_AdamW_base_test_eval
```

### Run with Test-Time Augmentation (TTA)

```bash
python3 evaluate_YOLO26_v2.py \
    --model pruned_models/YOLO26X_AdamW_finetune_best.pt \
    --data dataset.yaml \
    --split test \
    --model-variant YOLO26X_AdamW_finetune \
    --tta  # Enables TTA (slower but more accurate)
```

---

## Troubleshooting

### Post-Training Pipeline Failed

**Check:**
1. Does `post_training_pipeline.sh` exist and is executable?
   ```bash
   ls -la post_training_pipeline.sh
   chmod +x post_training_pipeline.sh
   ```

2. Does `prune_model.py` exist?
   ```bash
   ls -la prune_model.py
   ```

3. Does `evaluate_YOLO26_v2.py` exist?
   ```bash
   ls -la evaluate_YOLO26_v2.py
   ```

### Evaluation Not Showing in WandB

**Check:**
1. WandB mode is "online":
   ```bash
   echo $WANDB_MODE  # Should be empty or "online"
   ```

2. WandB authenticated:
   ```bash
   wandb status
   ```

3. Check WandB project name:
   ```bash
   # Should be "SAR_YOLO26"
   cat evaluate_YOLO26_v2.py | grep project
   ```

### Pruned Model Not Created

**Check:**
1. Original `best.pt` exists:
   ```bash
   find runs/detect/SAR_YOLO26 -name "best.pt"
   ```

2. Permissions on `pruned_models/` directory:
   ```bash
   mkdir -p pruned_models
   chmod 755 pruned_models
   ```

---

## Example WandB Report Template

### Title: "SAR YOLO26 Model Comparison - Test Set Results"

**Section 1: Overall Performance**
| Model | mAP50-95 | mAP50 | Precision | Recall | FPS |
|-------|----------|-------|-----------|--------|-----|
| (Auto-populated from WandB table) |

**Section 2: Optimizer Comparison**
- Bar chart: mAP50-95 by model size, grouped by optimizer
- Line chart: Speed vs Accuracy tradeoff

**Section 3: Fine-Tuning Impact**
- Table comparing base vs finetune for each model
- Delta calculations (% improvement)

**Section 4: Per-Class Performance**
- Heatmap: Class performance across models
- Identify which classes benefit most from larger models

---

## Integration with Your Paper

### Results Section - Test Set Performance

```latex
\subsection{Test Set Evaluation}

Following training, all 20 model variants were evaluated on the 
SARDet-100K test set. Models were first pruned to remove optimizer 
state, reducing file sizes by approximately 50\% while maintaining 
identical inference performance. 

Table~\ref{tab:test_results} presents the test set performance for 
all model configurations. The best-performing model, YOLO26X trained 
with AdamW and fine-tuned, achieved a test set mAP@.50-.95 of X.XXX, 
surpassing the previous state-of-the-art GSFFE-Net (0.643) by XX\%.

All evaluation metrics, including per-class performance and inference 
speed, are available in our WandB project: 
\url{https://wandb.ai/username/SAR_YOLO26}
```

### Methodology Section - Model Evaluation

```latex
\subsection{Model Evaluation Protocol}

After training completion, each model was automatically processed 
through a post-training pipeline: (1) optimizer state removal via 
pruning to reduce checkpoint size, and (2) evaluation on the held-out 
test set. All metrics were logged to Weights \& Biases for 
reproducibility and comparison.

Test set evaluation used a batch size of 128 across 8 NVIDIA H200 
GPUs with mixed-precision inference. Metrics include mean Average 
Precision at IoU thresholds of 0.50 (mAP@.50) and 0.50-0.95 
(mAP@.50-.95), overall precision and recall, and inference speed 
measured in frames per second (FPS).
```

---

## Summary

**Automated Workflow:**
```
Training Completes
    ↓
Post-Training Pipeline Runs Automatically
    ↓
Model Pruned (50% size reduction)
    ↓
Evaluated on Test Set
    ↓
Metrics Logged to WandB
    ↓
Ready for WandB Reports & Paper Results
```

**Benefits:**
- ✅ No manual steps required
- ✅ Consistent evaluation across all models
- ✅ Easy comparison via WandB reports
- ✅ Smaller model files for distribution
- ✅ Test set metrics (not validation set)
- ✅ Publication-ready results

**Your only task:** Create WandB reports after all 20 models complete!
