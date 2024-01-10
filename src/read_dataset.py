#!/usr/bin/python3
from ai.dataset import NodeDataset
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("target", nargs=1)
    parser.add_argument("-v", action="store_true")
    args = parser.parse_args()

    ds = NodeDataset.load(args.target[0])
    print(f"Loaded {len(ds)} entries (shape {ds.inputs.shape[1:]} => {ds.outputs.shape[1:]})")

    if args.v:
        for entry in ds:
            print(entry)

if __name__ == "__main__":
    main()