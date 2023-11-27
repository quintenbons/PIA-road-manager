import os
import csv
from utils import timing, BUILD_DIR

ROADS_FILE = '0_roads.csv'
SIMPLIFIED_ROADS_FILE = '1_simplified_roads.csv'
INTERSECTIONS_FILE = '2_intersections.csv'
LARGEST_CONNECTED_COMPONENT = '3_largest_connected_component.csv'
FINAL_ROADS_FILE = '4_final_roads.csv'
MAP_FILE = 'map.csv'

@timing
def save_to_csv(filename, data, directory=BUILD_DIR):
    """
    Sauvegarde les données dans un fichier CSV.
    """
    # Créer le répertoire s'il n'existe pas
    if not os.path.exists(directory):
        os.makedirs(directory)

    filepath = os.path.join(directory, filename)
    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        for row in data:
            writer.writerow(row)
