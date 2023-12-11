#!/usr/bin/python3
from ai.dataset import NodeDataset

def main():
    ds = NodeDataset.from_generation(100, tqdm_disable=False)
    print(f"Generated {len(ds)} entries")
    print(f"First entry: {ds[0]}")

if __name__ == "__main__":
    main()