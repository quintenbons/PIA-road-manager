#!/usr/bin/python3
from ai.dataset import NodeDataset

def main():
    ds = NodeDataset.from_generation(1, tqdm_disable=False)
    print(f"Generated {len(ds)} entries")

    for entry in ds:
        print(entry)

if __name__ == "__main__":
    main()