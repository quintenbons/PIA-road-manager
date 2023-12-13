#!/usr/bin/python3
from ai.dataset import NodeDataset
from ai.model import CrossRoadModel
import argparse
import os

from ai.training import train

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-scores", action="store_true")
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--dataset-path", type=str, default="dataset.pt")
    parser.add_argument("--model-path", type=str, default="model.pt")
    args = parser.parse_args()

    if not os.path.exists(args.dataset_path):
        print("No dataset found please generate one with gen_dataset.py")
        exit(1)

    # Create model
    if not os.path.exists(args.model_path):
        print("====== Generating model:")
        model = CrossRoadModel(15) # 15 strategies for 4 roads
        model.save(args.model_path)

    train(args.dataset_path, args.model_path, args.epoch)

if __name__ == "__main__":
    main()
