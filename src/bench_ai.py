#!/usr/bin/python3
import torch
import argparse
import os

from torch.utils.data import DataLoader
from ai.dataset import NodeDataset
from ai.model import CrossRoadModel
import torch.nn.functional as F

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
    pos1 = 0
    pos2 = 0

    with torch.no_grad():
        model = model.eval()
        for batch_idx, (inputs, expected) in enumerate(dataloader):
            results = F.softmax(model(inputs), dim=1)
            for output, target in zip(results, expected):
                hitormiss = "\033[92mHIT \033[0m" if output.argmax().item() == target.argmax().item() else "\033[91mMISS\033[0m"
                # print(f"{hitormiss} Got: {output.argmax().item():2d} Expected: {target.argmax().item():2d}")
                if output.argmax().item() == target.argmax().item():
                    hit += 1
                else:
                    miss += 1

                # Sort output indexes
                sorted_output = sorted(enumerate(output), key=lambda x: x[1], reverse=True)
                sorted_output = [x[0] for x in sorted_output]

                sorted_target = sorted(enumerate(target), key=lambda x: x[1], reverse=True)
                sorted_target = [x[0] for x in sorted_target]

                best_index = sorted_output.index(sorted_target[0])
                pos1 += best_index

                output_index = sorted_target.index(sorted_output[0])
                pos2 += output_index

                # print("output:", output)
                # print("target:", target)
                # print()
                # print(sorted_output)
                # print(sorted_target)
                # print()
                # print(best_index)
                # print(sorted_output[best_index])
                # print()
                # print(output_index)
                # print(sorted_target[output_index])
                # return

        print()
        print(f"HIT rate: {hit / (hit + miss) * 100:.1f}% ({hit}/{hit+miss})")
        print(f"Average position of best case of expected in prediction: {pos1 / (hit + miss):.1f}")
        print(f"Average position of best case of prediction in expected: {pos2 / (hit + miss):.1f} (this one if more important)")

if __name__ == "__main__":
    main()
