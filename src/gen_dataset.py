#!/usr/bin/python3
from ai.dataset import NodeDataset, score_tester
import argparse
import os

def generate_dataset(size: int, dest: os.PathLike, map_folder: str, tqdm_disable=False, quiet=False):
    if os.path.exists(dest) and not quiet:
        answer = input("Dataset already exists, are you sure you want to overwrite it? (y/n)")
        if answer.lower() != "y":
            print("Aborting")
            exit(1)

    ds = None
    try:
        ds = NodeDataset.from_generation(size, map_folder, tqdm_disable=tqdm_disable)
    except KeyboardInterrupt:
        print("Keyboard interrupt, saving dataset...")
    except Exception as e:
        print("Exception occured, saving dataset...")
        raise e
    finally:
        if ds is None or len(ds) == 0:
            print("Could not generate dataset, aborting")
            exit(1)
        ds.save(dest)
        print(f"Generated {len(ds)} entries")

        if not quiet:
            for entry in ds:
                print(entry)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-scores", action="store_true")
    parser.add_argument("--size", type=int, default=1)
    parser.add_argument("--dest", type=str, default="dataset.pt")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--tqdm-disable", action="store_true")
    parser.add_argument("map_folder", type=str, default="src/maps/build/GUI/Training-4/Uniform", help="Folder containing map.csv and paths.csv")
    args = parser.parse_args()

    if args.test_scores:
        print("====== Testing scores:")
        score_tester(map_folder=args.map_folder)

    print(f"====== Generating dataset for map {args.map_folder} ======")
    generate_dataset(args.size, args.dest, args.map_folder, tqdm_disable=args.tqdm_disable, quiet=args.quiet)

if __name__ == "__main__":
    main()