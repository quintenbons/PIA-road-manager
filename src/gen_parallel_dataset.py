#!/usr/bin/env python3
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

from gen_dataset import generate_dataset

def main():
    parser = argparse.ArgumentParser(description='Generate datasets in parallel.')
    parser.add_argument('duration', type=int, help='Duration in minutes for dataset generation.')
    parser.add_argument('--dest', type=str, default='datasets', help='Destination directory to save datasets.')
    args = parser.parse_args()

    # ENV:
    # MAX_CORES: Number of cores to use for dataset generation
    max_cores = os.environ.get("MAX_CORES") or 4
    max_cores = int(max_cores)
    duration_in_minutes = args.duration
    dest_directory = args.dest
    seconds_per_entry = 12  # Time to generate one entry
    size = 1 # duration_in_minutes * 60 // seconds_per_entry

    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)

    print(f"Generating datasets of size {size} on {max_cores} cores for {duration_in_minutes} minutes...", file=sys.stderr)

    futures = set()

    with ProcessPoolExecutor(max_workers=max_cores) as executor:
        for core_num in range(max_cores):
            dest = os.path.join(dest_directory, f"dataset-{core_num}.pt")
            future = executor.submit(generate_dataset, size, dest)
            futures.add(future)

        for future in as_completed(futures):
            result = future.result()
            print("================= future done res:", result, file=sys.stderr)

if __name__ == "__main__":
    main()
