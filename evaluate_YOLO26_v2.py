import os
import argparse
import wandb
import torch
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser("Evaluate YOLO26 on SARDet-100K Test Set")
    parser.add_argument("--model", type=str, required=True, 
                       help="Path to trained model weights (e.g., best.pt)")
    parser.add_argument("--data", type=str, default="dataset.yaml", 
                       help="Path to YOLO dataset config yaml")
    parser.add_argument("--split", type=str, default="test", 
                       help="Which split to evaluate on (val or test)")
    parser.add_argument("--imgsz", type=int, default=640, 
                       help="Image size")
    parser.add_argument("--batch", type=int, default=128, 
                       help="Batch size for evaluation")
    parser.add_argument("--project", type=str, default="SAR_YOLO26", 
                       help="W&B project name")
    parser.add_argument("--run-name", type=str, default=None,
                       help="W&B run name (auto-generated if not provided)")
    parser.add_argument("--wandb-mode", type=str, default="online",
                       choices=["online", "offline", "disabled"],
                       help="W&B logging mode")
    parser.add_argument("--tta", action="store_true", 
                       help="Enable Test-Time Augmentation (TTA)")
    parser.add_argument("--model-variant", type=str, default="unknown",
                       help="Model variant name (e.g., YOLO26N_AdamW_base)")
    args = parser.parse_args()

    # Auto-generate run name if not provided
    if args.run_name is None:
        # Extract model info from path
        model_dir = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(args.model))))
        args.run_name = f"{model_dir}_eval_{args.split}"
    
    print(f"\n{'='*80}")
    print(f"EVALUATING: {args.model}")
    print(f"Run Name:   {args.run_name}")
    print(f"Split:      {args.split}")
    print(f"TTA:        {args.tta}")
    print(f"{'='*80}\n")

    # Initialize WandB with detailed config
    os.environ["WANDB_MODE"] = args.wandb_mode
    
    wandb_config = {
        "model_path": args.model,
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "tta": args.tta,
        "model_variant": args.model_variant,
    }
    
    wandb.init(
        project=args.project,
        name=args.run_name,
        config=wandb_config,
        job_type="evaluation",
        tags=["test_set", "evaluation", args.model_variant]
    )

    # Load model
    model = YOLO(args.model)

    # Perform evaluation
    print("Running evaluation...")
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        plots=True,
        workers=8,
        augment=args.tta
    )

    # Calculate FPS
    total_time_ms = sum(metrics.speed.values())
    fps = 1000.0 / total_time_ms if total_time_ms > 0 else 0

    # Extract all metrics for WandB logging
    test_metrics = {
        # Overall metrics
        "test/mAP50-95": metrics.box.map,
        "test/mAP50": metrics.box.map50,
        "test/mAP75": metrics.box.map75,
        "test/precision": metrics.box.mp,
        "test/recall": metrics.box.mr,
        "test/fps": fps,
        
        # Per-class mAP50-95 (if available)
        "test/class_map": metrics.box.maps.tolist() if hasattr(metrics.box, 'maps') else [],
        
        # Speed metrics
        "test/preprocess_ms": metrics.speed.get('preprocess', 0),
        "test/inference_ms": metrics.speed.get('inference', 0),
        "test/postprocess_ms": metrics.speed.get('postprocess', 0),
        "test/total_ms": total_time_ms,
        
        # Model info
        "model_variant": args.model_variant,
        "tta_enabled": args.tta,
    }
    
    # Add per-class metrics if available
    if hasattr(metrics.box, 'class_result'):
        class_results = metrics.box.class_result
        for i, class_map in enumerate(class_results):
            test_metrics[f"test/class_{i}_mAP50-95"] = class_map
    
    # Log to WandB
    wandb.log(test_metrics)
    
    # Create summary table for WandB reports
    summary_table = wandb.Table(
        columns=[
            "Model Variant", "Split", "TTA", 
            "mAP50-95", "mAP50", "mAP75", 
            "Precision", "Recall", "FPS"
        ],
        data=[[
            args.model_variant,
            args.split,
            "Yes" if args.tta else "No",
            f"{metrics.box.map:.4f}",
            f"{metrics.box.map50:.4f}",
            f"{metrics.box.map75:.4f}",
            f"{metrics.box.mp:.4f}",
            f"{metrics.box.mr:.4f}",
            f"{fps:.2f}"
        ]]
    )
    
    wandb.log({"test/summary_table": summary_table})
    
    # Print results
    print(f"\n{'='*80}")
    print(f"EVALUATION RESULTS: {args.run_name}")
    print(f"{'='*80}")
    print(f"  Precision:  {metrics.box.mp:.4f}")
    print(f"  Recall:     {metrics.box.mr:.4f}")
    print(f"  mAP50:      {metrics.box.map50:.4f}")
    print(f"  mAP50-95:   {metrics.box.map:.4f}")
    print(f"  FPS:        {fps:.2f}")
    print(f"{'='*80}\n")
    
    # Close WandB run
    wandb.finish()
    
    return test_metrics


if __name__ == "__main__":
    main()
