import os
import csv


BUILD_DIR = os.path.join(os.path.dirname(__file__), 'build')
ROADS_FILE = 'roads.csv'
SIMPLIFIED_ROADS_FILE = 'simplified_roads.csv'
INTERSECTIONS_FILE = 'intersections.csv'


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
