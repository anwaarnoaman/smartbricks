# main.py
import argparse
from train import train_and_save_models

def main():
    parser = argparse.ArgumentParser(description="Training script for rent prediction.")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run training pipeline and save models.",
    )
    args = parser.parse_args()

    if args.train:
        print("Starting training pipeline...")
        train_and_save_models()
        print("Training pipeline completed.")

if __name__ == "__main__":
    main()
