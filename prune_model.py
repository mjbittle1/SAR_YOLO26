from ultralytics.utils.torch_utils import strip_optimizer
import argparse

def main():
    parser = argparse.ArgumentParser("Prune/Strip Optimizer from YOLO Model")
    parser.add_argument("--model", type=str, default="best.pt", help="Path to input model")
    parser.add_argument("--output", type=str, default="best_prune.pt", help="Path to output pruned model")
    args = parser.parse_args()

    print(f"Stripping optimizer from {args.model} and saving to {args.output}...")
    strip_optimizer(f=args.model, s=args.output)
    print("Done! Model sizes should be significantly reduced.")

if __name__ == "__main__":
    main()
# Command: python prune_model.py --model best.pt --output best_prune.pt    