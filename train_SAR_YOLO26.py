"""
SAR-Optimized YOLO26 Training Script
====================================

Trains all 5 YOLO26 models with SAR-specific augmentation configuration.
Includes automatic crash recovery and WandB best metric tracking.

Training Strategy:
- Phase 1: Base training (10 models: 5 AdamW + 5 MuSGD)
- Phase 2: Fine-tuning from best.pt (10 models)
- Total: 20 models

Author: Bittle SAR Research
Date: 2026
"""

import os
import sys
import argparse
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.callbacks import default_callbacks
import wandb


# =============================================================================
# WANDB CUSTOM CALLBACK FOR BEST METRIC TRACKING
# =============================================================================

class BestMetricTracker:
    """
    Custom callback to continuously update WandB with best mAP50-95 and epoch.
    This ensures you can always see the best performance achieved so far.
    """
    def __init__(self):
        self.best_map50_95 = 0.0
        self.best_epoch = 0
        
    def on_fit_epoch_end(self, trainer):
        """Called at the end of each training epoch"""
        try:
            # Get current metrics
            metrics = trainer.metrics
            if metrics:
                current_map = metrics.get('metrics/mAP50-95(B)', 0.0)
                current_epoch = trainer.epoch
                
                # Update best if current is better
                if current_map > self.best_map50_95:
                    self.best_map50_95 = current_map
                    self.best_epoch = current_epoch
                
                # Log to WandB continuously
                if wandb.run:
                    wandb.log({
                        'best/mAP50-95': self.best_map50_95,
                        'best/epoch': self.best_epoch,
                        'current/mAP50-95': current_map,
                    }, step=current_epoch)
                    
        except Exception as e:
            print(f"Warning: BestMetricTracker callback failed: {e}")


# =============================================================================
# SAR-OPTIMIZED TRAINING CONFIGURATION
# =============================================================================

def get_sar_optimized_config(
    data,
    epochs,
    batch,
    imgsz,
    project,
    name,
    device,
    patience,
    optimizer,
    lr0,
    warmup_epochs,
    cache="ram"
):
    """
    Returns SAR-optimized training configuration.
    
    Key SAR-Specific Modifications:
    - HSV augmentations DISABLED (SAR has no color information)
    - Auto-augment DISABLED (solarize/posterize break backscatter physics)
    - Perspective DISABLED (SAR is orthographic, not perspective)
    - Conservative geometric augmentations only
    - No backbone freezing (proven ineffective)
    
    Args:
        Standard YOLO training arguments
        
    Returns:
        dict: Complete training configuration
    """
    
    config = {
        # =================================================================
        # CORE SETTINGS
        # =================================================================
        "data": data,
        "epochs": epochs,
        "batch": batch,
        "imgsz": imgsz,
        "project": project,
        "name": name,
        "device": device,
        "patience": patience,
        "workers": 8,
        "cache": cache if cache != "False" else False,  # RAM caching for speed
        "exist_ok": True,
        "pretrained": True,
        "verbose": True,
        "seed": 42,
        
        # =================================================================
        # OPTIMIZER SETTINGS
        # =================================================================
        "optimizer": optimizer,
        "lr0": lr0,
        "lrf": 0.01,              # Final LR = lr0 * lrf
        "momentum": 0.937,        # SGD momentum / Adam beta1
        "weight_decay": 0.0005,   # L2 regularization
        "warmup_epochs": warmup_epochs,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        
        # =================================================================
        # TRAINING DYNAMICS
        # =================================================================
        "cos_lr": True,           # Cosine LR scheduler
        "amp": True,              # Automatic Mixed Precision (H200)
        "fraction": 1.0,          # Use 100% of dataset
        
        # =================================================================
        # LOSS WEIGHTS (DEFAULTS)
        # =================================================================
        "box": 7.5,               # Box regression loss weight
        "cls": 0.5,               # Classification loss weight
        "dfl": 1.5,               # DFL loss (ignored in YOLO26, included for completeness)
        
        # =================================================================
        # SAR-SPECIFIC: DISABLE OPTICAL AUGMENTATIONS
        # =================================================================
        # CRITICAL: SAR is radar backscatter, not RGB color.
        # These augmentations destroy physical meaning and MUST be disabled.
        
        "hsv_h": 0.0,             # Hue shift DISABLED
        "hsv_s": 0.0,             # Saturation shift DISABLED  
        "hsv_v": 0.0,             # Value/brightness shift DISABLED
        "auto_augment": False,    # AutoAugment DISABLED (solarize, posterize, etc.)
        
        # =================================================================
        # SAR-SPECIFIC: GEOMETRIC AUGMENTATIONS
        # =================================================================
        # SAR targets appear at arbitrary orientations in satellite imagery.
        # Use rotational invariance and overhead-appropriate transforms.
        
        "degrees": 180.0,         # ±180° rotation (full coverage)
        "translate": 0.1,         # ±10% translation
        "scale": 0.4,             # ±40% scale (reduced for small objects)
        "shear": 0.0,             # Shear DISABLED
        "perspective": 0.0,       # Perspective DISABLED (SAR is orthographic)
        "flipud": 0.5,            # Vertical flip 50%
        "fliplr": 0.5,            # Horizontal flip 50%
        
        # =================================================================
        # SAR-SPECIFIC: CONSERVATIVE AUGMENTATION BASELINE
        # =================================================================
        # Advanced augmentations disabled based on SAR research showing
        # they can introduce physically unrealistic artifacts.
        
        "mosaic": 0.0,            # Mosaic DISABLED (can break shadow continuity)
        "mixup": 0.0,             # Mixup DISABLED (unrealistic backscatter blending)
        "copy_paste": 0.0,        # Copy-paste DISABLED (breaks physics)
        "erasing": 0.0,           # Random erasing DISABLED (can remove small targets)
        "crop_fraction": 1.0,     # No cropping (preserves context)
        
        # =================================================================
        # REGULARIZATION
        # =================================================================
        "label_smoothing": 0.1,   # Anti-overfitting for noisy SAR
        "dropout": 0.0,           # Dropout disabled
        
        # =================================================================
        # VALIDATION & CHECKPOINTING
        # =================================================================
        "val": True,
        "plots": True,
        "save": True,
        "save_period": -1,        # Save only last + best
        
        # =================================================================
        # MULTI-SCALE TRAINING
        # =================================================================
        "multi_scale": True,      # Critical for small object detection
        "close_mosaic": 10,       # Disable mosaic last 10 epochs (already 0, but kept for clarity)
    }
    
    return config


# =============================================================================
# TRAINING EXECUTION
# =============================================================================

def train_model(
    model_size,
    optimizer,
    lr0,
    phase,
    data_yaml,
    epochs,
    batch,
    patience,
    device,
    warmup_epochs,
    resume_from=None
):
    """
    Train a single YOLO26 model with SAR-optimized configuration.
    
    Args:
        model_size: 'n', 's', 'm', 'l', or 'x'
        optimizer: 'AdamW' or 'MuSGD'
        lr0: Initial learning rate
        phase: 'base' or 'finetune'
        resume_from: Path to weights (for fine-tuning) or None (for base training)
        
    Returns:
        Path to best.pt checkpoint
    """
    
    # Build experiment name
    model_name = f"YOLO26{model_size.upper()}_{optimizer}_{phase}"
    
    # Determine weights to load
    if resume_from:
        weights = resume_from
        print(f"\n{'='*80}")
        print(f"🔄 FINE-TUNING: {model_name}")
        print(f"Loading from: {weights}")
    else:
        weights = f"yolo26{model_size}.pt"
        print(f"\n{'='*80}")
        print(f"🚀 BASE TRAINING: {model_name}")
        print(f"Starting from: {weights}")
    
    print(f"{'='*80}\n")
    
    # Check for existing checkpoint to resume from
    project_dir = "SAR_YOLO26"
    run_dir = Path("runs/detect") / project_dir / model_name
    last_pt = run_dir / "weights" / "last.pt"
    
    if last_pt.exists():
        print(f"✅ Found existing checkpoint: {last_pt}")
        print(f"🔄 Resuming training from last.pt\n")
        weights = str(last_pt)
        resume = True
    else:
        resume = False
    
    # Set WandB environment variables for this run
    os.environ["WANDB_PROJECT"] = project_dir
    os.environ["WANDB_NAME"] = model_name
    
    # Initialize model
    model = YOLO(weights)
    
    # Add custom callback for best metric tracking
    best_tracker = BestMetricTracker()
    
    # Get SAR-optimized configuration
    train_args = get_sar_optimized_config(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=640,
        project=project_dir,
        name=model_name,
        device=device,
        patience=patience,
        optimizer=optimizer,
        lr0=lr0,
        warmup_epochs=warmup_epochs,
        cache="ram"  # Enable RAM caching by default
    )
    
    # Add resume flag if needed
    if resume:
        train_args["resume"] = True
    
    # Print configuration summary
    print(f"📊 CONFIGURATION SUMMARY:")
    print(f"  Optimizer:      {optimizer}")
    print(f"  Learning Rate:  {lr0}")
    print(f"  Warmup Epochs:  {warmup_epochs}")
    print(f"  Batch Size:     {batch} total ({batch//8} per GPU)")
    print(f"  Epochs:         {epochs}")
    print(f"  Patience:       {patience}")
    print(f"  Device:         {device}")
    print(f"  SAR-Optimized:  ✅ (HSV disabled, conservative augmentations)")
    print(f"\n{'='*80}\n")
    
    # Train with custom callback
    try:
        # Register the callback
        model.add_callback("on_fit_epoch_end", best_tracker.on_fit_epoch_end)
        
        # Start training
        results = model.train(**train_args)
        
        print(f"\n{'='*80}")
        print(f"✅ TRAINING COMPLETE: {model_name}")
        print(f"Best mAP50-95: {best_tracker.best_map50_95:.4f} @ epoch {best_tracker.best_epoch}")
        print(f"{'='*80}\n")
        
        best_pt_path = run_dir / "weights" / "best.pt"
        
        # Run post-training pipeline: pruning + evaluation
        print(f"\n{'='*80}")
        print(f"🔄 POST-TRAINING PIPELINE: {model_name}")
        print(f"{'='*80}\n")
        
        import subprocess
        
        try:
            # Run pruning and evaluation
            result = subprocess.run(
                [
                    "bash",
                    "post_training_pipeline.sh",
                    str(best_pt_path),
                    model_name,
                    data_yaml
                ],
                check=True,
                capture_output=False,
                text=True
            )
            
            print(f"\n{'='*80}")
            print(f"✅ POST-TRAINING PIPELINE COMPLETE: {model_name}")
            print(f"{'='*80}\n")
            
        except subprocess.CalledProcessError as e:
            print(f"\n{'='*80}")
            print(f"⚠️  POST-TRAINING PIPELINE FAILED: {model_name}")
            print(f"Error: {e}")
            print(f"Continuing to next model...")
            print(f"{'='*80}\n")
        
        return best_pt_path
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"❌ ERROR during training: {model_name}")
        print(f"Error: {e}")
        print(f"{'='*80}\n")
        raise


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main training pipeline"""
    
    parser = argparse.ArgumentParser(
        description="SAR-Optimized YOLO26 Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--data",
        type=str,
        default="dataset.yaml",
        help="Path to dataset YAML file"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="Training epochs (default: 500)"
    )
    
    parser.add_argument(
        "--batch",
        type=int,
        default=128,
        help="Total batch size. Recommended: 128 for 500 epochs (16 per GPU on 8 GPUs)"
    )
    
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early stopping patience (default: 50)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="0,1,2,3,4,5,6,7",
        help="GPU devices (default: all 8 GPUs)"
    )
    
    parser.add_argument(
        "--warmup-epochs",
        type=float,
        default=3.0,
        help="Warmup epochs (default: 3.0 - well-documented best practice)"
    )
    
    parser.add_argument(
        "--cache",
        type=str,
        default="ram",
        choices=["ram", "disk", "False"],
        help="Cache images for faster training: 'ram' (fastest, needs ~80GB), 'disk' (medium), 'False' (slowest)"
    )
    
    parser.add_argument(
        "--skip-base",
        action="store_true",
        help="Skip base training (only run fine-tuning from existing checkpoints)"
    )
    
    parser.add_argument(
        "--skip-finetune",
        action="store_true",
        help="Skip fine-tuning (only run base training)"
    )
    
    args = parser.parse_args()
    
    # Model sizes and configurations
    MODEL_SIZES = ['n', 's', 'm', 'l', 'x']
    
    # Optimizer configurations
    ADAMW_CONFIG = {
        'n': 0.001,
        's': 0.001,
        'm': 0.001,
        'l': 0.001,
        'x': 0.0005  # Lower LR for largest model
    }
    
    MUSGD_CONFIG = {
        'n': 0.01,
        's': 0.01,
        'm': 0.01,
        'l': 0.01,
        'x': 0.01
    }
    
    print("\n" + "="*80)
    print("SAR-OPTIMIZED YOLO26 TRAINING PIPELINE")
    print("="*80)
    print(f"\n📋 CONFIGURATION:")
    print(f"  Dataset:        {args.data}")
    print(f"  Epochs:         {args.epochs}")
    print(f"  Batch Size:     {args.batch} total ({args.batch//8} per GPU)")
    print(f"  Patience:       {args.patience}")
    print(f"  Warmup Epochs:  {args.warmup_epochs}")
    print(f"  Device:         {args.device}")
    print(f"\n🎯 TRAINING STRATEGY:")
    print(f"  Phase 1: Base Training (AdamW → MuSGD)")
    print(f"  Phase 2: Fine-Tuning (from best.pt)")
    print(f"  Total Models: {len(MODEL_SIZES) * 2 * 2} = 20")
    print(f"\n⚙️  SAR-SPECIFIC OPTIMIZATIONS:")
    print(f"  ✅ HSV augmentations DISABLED")
    print(f"  ✅ Auto-augment DISABLED")
    print(f"  ✅ Perspective transform DISABLED")
    print(f"  ✅ Conservative geometric augmentations")
    print(f"  ✅ No backbone freezing")
    print("="*80 + "\n")
    
    # Storage for best checkpoints
    base_checkpoints = {}
    
    # =================================================================
    # PHASE 1: BASE TRAINING
    # =================================================================
    
    if not args.skip_base:
        print("\n" + "="*80)
        print("PHASE 1: BASE TRAINING")
        print("="*80 + "\n")
        
        # Train all AdamW models first
        print("\n" + "-"*80)
        print("OPTIMIZER: AdamW")
        print("-"*80 + "\n")
        
        for size in MODEL_SIZES:
            lr = ADAMW_CONFIG[size]
            checkpoint = train_model(
                model_size=size,
                optimizer="AdamW",
                lr0=lr,
                phase="base",
                data_yaml=args.data,
                epochs=args.epochs,
                batch=args.batch,
                patience=args.patience,
                device=args.device,
                warmup_epochs=args.warmup_epochs,
                resume_from=None
            )
            base_checkpoints[f"{size}_AdamW"] = checkpoint
        
        # Train all MuSGD models
        print("\n" + "-"*80)
        print("OPTIMIZER: MuSGD")
        print("-"*80 + "\n")
        
        for size in MODEL_SIZES:
            lr = MUSGD_CONFIG[size]
            checkpoint = train_model(
                model_size=size,
                optimizer="MuSGD",
                lr0=lr,
                phase="base",
                data_yaml=args.data,
                epochs=args.epochs,
                batch=args.batch,
                patience=args.patience,
                device=args.device,
                warmup_epochs=args.warmup_epochs,
                resume_from=None
            )
            base_checkpoints[f"{size}_MuSGD"] = checkpoint
    
    else:
        print("\n⏭️  Skipping base training (--skip-base flag)")
        print("Loading existing checkpoints for fine-tuning...\n")
        
        # Try to find existing checkpoints
        project_dir = "SAR_YOLO26"
        for size in MODEL_SIZES:
            for opt in ["AdamW", "MuSGD"]:
                model_name = f"YOLO26{size.upper()}_{opt}_base"
                checkpoint_path = Path("runs/detect") / project_dir / model_name / "weights" / "best.pt"
                
                if checkpoint_path.exists():
                    base_checkpoints[f"{size}_{opt}"] = checkpoint_path
                    print(f"✅ Found: {checkpoint_path}")
                else:
                    print(f"⚠️  Missing: {checkpoint_path}")
    
    # =================================================================
    # PHASE 2: FINE-TUNING
    # =================================================================
    
    if not args.skip_finetune:
        print("\n" + "="*80)
        print("PHASE 2: FINE-TUNING FROM BEST CHECKPOINTS")
        print("="*80 + "\n")
        
        # Fine-tune all AdamW models
        print("\n" + "-"*80)
        print("OPTIMIZER: AdamW")
        print("-"*80 + "\n")
        
        for size in MODEL_SIZES:
            lr = ADAMW_CONFIG[size]
            base_checkpoint = base_checkpoints.get(f"{size}_AdamW")
            
            if base_checkpoint and base_checkpoint.exists():
                train_model(
                    model_size=size,
                    optimizer="AdamW",
                    lr0=lr,
                    phase="finetune",
                    data_yaml=args.data,
                    epochs=args.epochs,
                    batch=args.batch,
                    patience=args.patience,
                    device=args.device,
                    warmup_epochs=args.warmup_epochs,
                    resume_from=str(base_checkpoint)
                )
            else:
                print(f"⚠️  Skipping fine-tune for YOLO26{size.upper()}_AdamW (base checkpoint missing)")
        
        # Fine-tune all MuSGD models
        print("\n" + "-"*80)
        print("OPTIMIZER: MuSGD")
        print("-"*80 + "\n")
        
        for size in MODEL_SIZES:
            lr = MUSGD_CONFIG[size]
            base_checkpoint = base_checkpoints.get(f"{size}_MuSGD")
            
            if base_checkpoint and base_checkpoint.exists():
                train_model(
                    model_size=size,
                    optimizer="MuSGD",
                    lr0=lr,
                    phase="finetune",
                    data_yaml=args.data,
                    epochs=args.epochs,
                    batch=args.batch,
                    patience=args.patience,
                    device=args.device,
                    warmup_epochs=args.warmup_epochs,
                    resume_from=str(base_checkpoint)
                )
            else:
                print(f"⚠️  Skipping fine-tune for YOLO26{size.upper()}_MuSGD (base checkpoint missing)")
    
    else:
        print("\n⏭️  Skipping fine-tuning (--skip-finetune flag)")
    
    # =================================================================
    # COMPLETION SUMMARY
    # =================================================================
    
    print("\n" + "="*80)
    print("🎉 TRAINING PIPELINE COMPLETE!")
    print("="*80)
    print(f"\n📊 All models saved to: runs/detect/SAR_YOLO26/")
    print(f"📈 WandB project: SAR_YOLO26")
    print(f"\n✅ Check WandB for:")
    print(f"  - best/mAP50-95: Best mAP50-95 achieved")
    print(f"  - best/epoch: Epoch where best mAP occurred")
    print(f"  - current/mAP50-95: Current epoch mAP50-95")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
