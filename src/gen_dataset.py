#!/usr/bin/python3
import signal
import sys
from ai.dataset import NodeDataset, score_tester
import argparse

global_dataset = None

def main():
    global global_dataset
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-scores", action="store_true")
    parser.add_argument("--size", type=int, default=1)
    parser.add_argument("--dest", type=str, default="dataset.pt")
    args = parser.parse_args()

    def signal_handler(sig, frame):
        print("\nInterrupt received, saving dataset...")
        if global_dataset is not None:
            global_dataset.save(args.dest)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    if args.test_scores:
        print("====== Testing scores:")
        score_tester()

    print("====== Generating dataset:")
    global_dataset = NodeDataset.from_generation(args.size, tqdm_disable=False)
    
    save_interval = 10  # Save every 10 entries
    for i, entry in enumerate(global_dataset):
        if i % save_interval == 0:
            global_dataset.save(args.dest)
        print(entry)

    print(f"Generated {len(global_dataset)} entries")
    global_dataset.save(args.dest)

if __name__ == "__main__":
    main()
