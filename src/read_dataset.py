#!/usr/bin/python3
from ai.dataset import NodeDataset
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("target", nargs=1)
    parser.add_argument("-v", action="store_true", help="Verbose mode")
    parser.add_argument("--describe-index", type=int, default=-1, help="Describe a specific index of the dataset")
    args = parser.parse_args()

    ds = NodeDataset.load(args.target[0])
    print(f"Loaded {len(ds)} entries (shape {ds.inputs.shape[1:]} => {ds.outputs.shape[1:]})")

    if args.v:
        for entry in ds:
            print(entry)

    if args.describe_index >= 0:
        print(f"====== Describing index {args.describe_index}:")
        ds.describe_index(args.describe_index)

if __name__ == "__main__":
    main()