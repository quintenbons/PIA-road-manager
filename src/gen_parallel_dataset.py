#!/usr/bin/env python3
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

parser = argparse.ArgumentParser(description='Generate datasets in parallel.')
parser.add_argument('duration', type=int, help='Duration in minutes for dataset generation.')
parser.add_argument('--dest', type=str, default='datasets', help='Destination directory to save datasets.')
args = parser.parse_args()

# ENV:
# MAX_CORES: Number of cores to use for dataset generation
max_cores = os.environ.get("MAX_CORES") or 4
max_cores = int(max_cores)

def generate_dataset(duration, size, dest, core_num):
    dataset_filename = f"{dest}_core_{core_num}.pt"
    command = f"python src/gen_dataset.py --size {size} --dest {dataset_filename}"
    os.system(command)
    return f"Dataset of size {size} generated in {duration} minutes and saved as {dataset_filename}."

if __name__ == "__main__":
    duration_in_minutes = args.duration
    dest_directory = args.dest
    seconds_per_entry = 12  # Time to generate one entry
    size = duration_in_minutes * 60 // seconds_per_entry

    print(f"Generating datasets of size {size} on {max_cores} cores for {duration_in_minutes} minutes...", file=sys.stderr)

    futures = set()

    with ProcessPoolExecutor(max_workers=max_cores) as executor:
        for core_num in range(max_cores):
            future = executor.submit(generate_dataset, duration_in_minutes, size, dest_directory, core_num)
            futures.add(future)

        for future in as_completed(futures):
            result = future.result()
            print(result, file=sys.stderr)
