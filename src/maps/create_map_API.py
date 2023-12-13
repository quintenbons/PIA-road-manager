#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(__file__))
from api_map_creation.processing import process_road_data
from api_map_creation.plotting import plot_roads_and_intersections, analyze_times
from api_map_creation.utils import BUILD_DIR
import time


if not os.path.exists(BUILD_DIR):
    print("Creating build directory...")
    os.makedirs(BUILD_DIR)


if len(sys.argv) > 1:
    city_name = sys.argv[1]
else:
    city_name = input("Enter the name of the city you want to process : ")


file_path = os.path.join(BUILD_DIR, city_name)

# Create the folder for this city if it doesn't exist, and if it does ask the user if he wants to overwrite it
if os.path.exists(file_path):
    print(f"The folder '{file_path}' already exists.")
    while True:
        overwrite_input = input("Do you want to overwrite it ? (y/n)\n")
        if overwrite_input == "y":
            break
        elif overwrite_input == "n":
            exit()
        else:
            print("Please enter 'y' or 'n'")
            continue
else:
    os.makedirs(file_path)

start = time.time()
roads = process_road_data(city_name, GENERATE_CSV=True) # Use GENERATE_CSV=True, False needs to be fixed
print("Process time : ", time.time() - start)

print("You can find time analysis in the 'build' folder")
analyze_times()

while True:
    user_input = input("Do you want to plot the roads and intersections ? (y/n)\n")
    
    if user_input == "y":
        start = time.time()
        plot_roads_and_intersections(roads)
        print("Plot time : ", time.time() - start)
        break
    elif user_input == "n":
        break
    else:
        print("Please enter 'y' or 'n' \n")
        continue



while True:
    create_paths = input("Also create corresponding paths ? (y/n)\n")

    if create_paths == "y":
        start = time.time()
        path_filename = os.path.join(BUILD_DIR+city_name, "paths.csv")
        os.system(f"./src/maps/cpp/dijkstra {file_path}/map.csv > {path_filename}")
        # print la commande
        print(f"Paths file generated: '{path_filename}'")
        print("Path creation time : ", time.time() - start)
        break
    elif create_paths == "n":
        break
    else:
        print("Please enter 'y' or 'n'")
        continue