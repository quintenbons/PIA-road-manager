#!/usr/bin/python3
from typing import List
from ai.dataset import NodeDataset
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=str, default="dataset.pt")
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()

    if len(args.files) < 2:
        print("You need to specify at least 2 files to merge")
        exit(1)

    if os.path.exists(args.dest):
        answer = input(f"Destination file {args.dest} already exists, are you sure you want to overwrite it? (y/n)")
        if answer != "y":
            exit(1)

    datasets: List[NodeDataset] = []

    for f in args.files:
        if not os.path.exists(f):
            print(f"File {f} does not exist")
            exit(1)
        else:
            datasets.append(NodeDataset.load(f))

    print("====== Merging datasets:")
    ds = datasets[0].merge_all(datasets[1:])
    print(f"Generated {len(ds)} entries")

    ds.save(args.dest)

if __name__ == "__main__":
    main()