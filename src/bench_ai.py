#!/usr/bin/python3
import torch
import argparse
import os

from torch.utils.data import DataLoader
from ai.training import train

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="model.pt")
    parser.add_argument("--dataset-path", type=str, default="dataset.pt")
    args = parser.parse_args()

    if not os.path.exists(args.model_path) or not os.path.exists(args.dataset_path):
        print(f"Dataset or model missing {args.model_path} {args.dataset_path}}")
        exit(1)

    model = torch.load(args.model_path)
    dataset = torch.load(args.dataset_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    print("+++++++++++++++++++ This is not totally working ++++++++++++++++++++++")

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            print(model(inputs))

if __name__ == "__main__":
    main()
