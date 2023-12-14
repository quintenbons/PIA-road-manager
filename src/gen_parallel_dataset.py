#!/usr/bin/env python3
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import random
import time

from gen_dataset import generate_dataset

def generate_dataset_seeded(seed: int, size: int, dest: os.PathLike, tqdm_disable=False, quiet=True):
    random.seed(seed)
    generate_dataset(size, dest, tqdm_disable=tqdm_disable, quiet=quiet)

def main():
    parser = argparse.ArgumentParser(description='Generate datasets in parallel.')
    parser.add_argument('duration', type=int, help='Duration in minutes for dataset generation.')
    parser.add_argument('--dest', type=str, default='datasets', help='Destination directory to save datasets.')
    args = parser.parse_args()

    if os.path.exists(args.dest):
        answer = input(f"Folder {args.dest} already exists, are you sure you want to overwrite it? (y/n)")
        if answer.lower() != "y":
            print("Aborting")
            exit(1)

    # ENV:
    # MAX_CORES: Number of cores to use for dataset generation
    max_cores = os.environ.get("MAX_CORES") or 4
    max_cores = int(max_cores)
    duration_in_minutes = args.duration
    dest_directory = args.dest
    seconds_per_entry = 12  # Time to generate one entry
    size = duration_in_minutes * 60 // seconds_per_entry

    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)

    print(f"Generating datasets of size {size} on {max_cores} cores for {duration_in_minutes} minutes...", file=sys.stderr)

    futures = set()

    base_seed = int(time.time())

    with ProcessPoolExecutor(max_workers=max_cores) as executor:
        for core_num in range(max_cores):
            dest = os.path.join(dest_directory, f"dataset-{core_num}.pt")
            future = executor.submit(generate_dataset_seeded, base_seed + core_num, size, dest)
            futures.add(future)

        for future in as_completed(futures):
            result = future.result()
            print("================= future done res:", result, file=sys.stderr)

if __name__ == "__main__":
    main()
