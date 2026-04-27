import os
import argparse
import wandb
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Fine-tune an existing YOLO26 checkpoint on a single GPU.")
    parser.add_argument("--weights",       type=str, required=True,
                        help="Path to existing trained .pt file (e.g. Best/best_YOLO26x.pt)")
    parser.add_argument("--data",          type=str, default="dataset.yaml",
                        help="Path to YOLO dataset config yaml")
    parser.add_argument("--epochs",        type=int, default=300)
    parser.add_argument("--batch",         type=int, default=16,
                        help="Total batch size for the single GPU.")
    parser.add_argument("--imgsz",         type=int, default=640)
    parser.add_argument("--optimizer",     type=str, default="AdamW",
                        choices=["SGD", "Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "auto"])
    parser.add_argument("--project",       type=str, default="SARDet_100K_YOLO26")
    parser.add_argument("--name",          type=str, default="",
                        help="Run name. Auto-derived from weights filename if not set.")
    parser.add_argument("--patience",      type=int, default=50)
    parser.add_argument("--device",        type=str, default="0")
    parser.add_argument("--freeze_layers", type=int, default=10,
                        help="Freeze the first N layers (default 10 = full YOLO26 backbone).")
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--lr",            type=float, default=2e-4,
                        help="Initial learning rate (lr0).")
    parser.add_argument("--cfg",           type=str, default=None,
                        help="Optional path to hyperparameter yaml override.")
    parser.add_argument("--cache",         action="store_true")

    args = parser.parse_args()

    model_name = args.name if args.name else os.path.splitext(os.path.basename(args.weights))[0] + "_tuned"

    print(f"\n{'='*55}")
    print(f"Fine-tuning : {args.weights}")
    print(f"Run name    : {model_name}")
    print(f"Device      : GPU {args.device}  |  batch={args.batch}  |  lr={args.lr}")
    print(f"Freeze      : first {args.freeze_layers} layers (backbone)")
    print(f"{'='*55}\n")

    # Initialize W&B
    wandb.init(project=args.project, name=model_name, config=vars(args))

    model = YOLO(args.weights)

    train_args = {
        "data":          args.data,
        "epochs":        args.epochs,
        "batch":         args.batch,
        "imgsz":         args.imgsz,
        "project":       args.project,
        "name":          model_name,
        "device":        args.device,
        "patience":      args.patience,
        "cache":         args.cache,
        "optimizer":     args.optimizer,
        "workers":       8,        # Single-GPU: use background workers to keep the GPU fed
        "warmup_epochs": args.warmup_epochs,
        "lr0":           args.lr,
    }

    if args.freeze_layers > 0:
        train_args["freeze"] = args.freeze_layers
        print(f"Freezing first {args.freeze_layers} layers (backbone protection).")

    if args.cfg:
        train_args["cfg"] = args.cfg

    model.train(**train_args)


if __name__ == "__main__":
    main()