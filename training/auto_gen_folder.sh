#!/bin/bash
if [ -z $2 ]; then
    echo "Usage: $0 <maps_folder> <output_folder>"
    echo "    maps_folder: the folder containing the maps"
    echo "    output_folder: the folder to put the datasets in"
    echo "    Example: $0 src/maps/Training-4/Uniform data/training/cpp"
    exit 1
fi

INPUT=$1
OUTPUT=$2

echo "Generating datasets for maps in $INPUT to $OUTPUT"

read -sp "Enter your ensimag password: " password
echo ""

for r in $INPUT/*; do
    if [ -d "$r" ]; then
        map_folder="$r"
        echo "Generating dataset for $map_folder"
        rm -f ensimag.pt
        { echo $password; yes "y"; } | ./training/gen_datasets.sh bonsq main 250 "$map_folder"

        map_name=$(basename "$map_folder")
        mv ensimag.pt "$OUTPUT/$map_name.pt"
    fi
done
