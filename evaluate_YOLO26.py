import os
import argparse
import wandb
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser("Evaluate YOLO26X on SARDet-100K")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model weights (e.g., best.pt)")
    parser.add_argument("--data", type=str, default="dataset.yaml", help="Path to YOLO dataset config yaml")
    parser.add_argument("--split", type=str, default="test", help="Which split to evaluate on (val or test)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--project", type=str, default="SARDet_100K_YOLO26", help="W&B project name")
    parser.add_argument("--wandb_mode", type=str, default="online",
                        choices=["online", "offline", "disabled"],
                        help="W&B logging mode")
    args = parser.parse_args()

    # Derive run name from model path: .../YOLO26N/weights/best.pt → YOLO26N_eval_test
    model_dir = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(args.model))))
    run_name = f"{model_dir}_eval_{args.split}"

    os.environ["WANDB_MODE"] = args.wandb_mode
    wandb.init(project=args.project, name=run_name, config=vars(args))

    model = YOLO(args.model)

    # Perform evaluation
    metrics = model.val(data=args.data, split=args.split, imgsz=args.imgsz, batch=args.batch, plots=True, workers=8)#, half=False, iou=0.55)

    # half=False Forces FP32 precision to stop H200 rounding errors
    # iou=0.65,      # Tweaks NMS threshold to help the Aircraft class
        
    # Calculate FPS
    total_time_ms = sum(metrics.speed.values())
    fps = 1000.0 / total_time_ms if total_time_ms > 0 else 0

    print(f"Evaluation completed. P: {metrics.box.mp:.4f}, R: {metrics.box.mr:.4f}, mAP50: {metrics.box.map50:.4f}, mAP50-95: {metrics.box.map:.4f}, FPS: {fps:.2f}")

if __name__ == "__main__":
    main()

# python evaluate_YOLO26.py --model best.pt --data dataset.yaml