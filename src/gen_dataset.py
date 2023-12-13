#!/usr/bin/python3
from ai.dataset import NodeDataset, score_tester
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-scores", action="store_true")
    parser.add_argument("--size", type=int, default=1)
    parser.add_argument("--dest", type=str, default="dataset.pt")
    args = parser.parse_args()

    if args.test_scores:
        print("====== Testing scores:")
        score_tester()

    print("====== Generating dataset:")
    ds = NodeDataset.from_generation(args.size, tqdm_disable=False)
    print(f"Generated {len(ds)} entries")

    for entry in ds:
        print(entry)

    ds.save(args.dest)

if __name__ == "__main__":
    main()