#!/usr/bin/python3
import torch
import argparse
import os

from torch.utils.data import DataLoader
from ai.dataset import NodeDataset
from ai.model import CrossRoadModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="model.pt")
    parser.add_argument("--dataset-path", type=str, default="dataset.pt")
    args = parser.parse_args()

    if not os.path.exists(args.model_path) or not os.path.exists(args.dataset_path):
        print(f"Dataset or model missing {args.model_path} {args.dataset_path}")
        exit(1)

    model = CrossRoadModel.load(args.model_path)
    dataset = NodeDataset.load(args.dataset_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    print("+++++++++++++++++++ This is not totally working ++++++++++++++++++++++")

    hit = 0
    miss = 0

    with torch.no_grad():
        for batch_idx, (inputs, expected) in enumerate(dataloader):
            results = model(inputs)
            for output, target in zip(results, expected):
                hitormiss = "\033[92mHIT \033[0m" if output.argmax().item() == target.argmax().item() else "\033[91mMISS\033[0m"
                print(f"{hitormiss} Got: {output.argmax().item():2d} Expected: {target.argmax().item():2d}")
                if output.argmax().item() == target.argmax().item():
                    hit += 1
                else:
                    miss += 1
        print()
        print(f"HIT rate: {hit / (hit + miss) * 100:.1f}% ({hit}/{hit+miss})")

if __name__ == "__main__":
    main()
