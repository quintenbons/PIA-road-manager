import ast
import csv
import os
import time
from shapely.geometry import LineString
from rtree import index
from fetching import fetch_road_data
from file_manager import save_to_csv, BUILD_DIR
from utils import timing

@timing
def find_intersections(roads):
    intersections = set()
    idx = index.Index()

    # Indexation des routes
    for i, road in enumerate(roads):
        road_line = LineString(road)
        idx.insert(i, road_line.bounds)

    # Recherche d'intersections
    for i, road1 in enumerate(roads):
        road1_line = LineString(road1)
        # Nous obtenons les identifiants des routes potentiellement intersectantes
        potential_matches = list(idx.intersection(road1_line.bounds))

        for j in potential_matches:
            # Assurez-vous de ne pas comparer une route avec elle-même
            # et évitez les comparaisons redondantes
            if j <= i:
                continue

            road2_line = LineString(roads[j])
            if road1_line.intersects(road2_line):
                intersection = road1_line.intersection(road2_line)

                if intersection.geom_type == 'Point':
                    intersections.add((intersection.x, intersection.y))
                elif intersection.geom_type == 'MultiPoint':
                    for point in intersection.geoms:
                        intersections.add((point.x, point.y))
                elif intersection.geom_type == 'LineString':
                    for point in intersection.coords:
                        intersections.add(point)

    return list(intersections)

@timing
def simplify_roads_csv(input_filepath, output_filepath):
    simplified_roads = []

    # Lire le fichier CSV d'entrée
    file_path = os.path.join(BUILD_DIR, input_filepath)
    output_filepath = os.path.join(BUILD_DIR, output_filepath)
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in reader:
            # Convertir les strings de tuples en tuples
            first_point = ast.literal_eval(row[0])
            last_point = ast.literal_eval(row[-1])

            simplified_road = [first_point, last_point]
            simplified_roads.append(simplified_road)

    save_to_csv(output_filepath, simplified_roads)

    # Retourner la liste des routes simplifiées
    return simplified_roads

@timing
def process_road_data(city_name, simplify=True):
    road_data = fetch_road_data(city_name)

    print("Parsing data...")
    nodes = {element['id']: (element['lat'], element['lon'])
             for element in road_data['elements']
             if element['type'] == 'node'}

    roads = []
    for element in road_data['elements']:
        if element['type'] == 'way':
            road_nodes = [nodes[node_id] for node_id in element['nodes']]
            roads.append(road_nodes)

    print("Finding intersections...")
    intersections = find_intersections(roads)

    # Sauvegarder les intersections et les routes dans des fichiers CSV
    print("Saving data...")
    save_to_csv('intersections.csv', intersections)
    save_to_csv('roads.csv', [[point for point in road] for road in roads])

    input_filepath = 'roads.csv'
    output_filepath = 'simplified_roads.csv'
    roads = simplify_roads_csv(input_filepath, output_filepath)

    return intersections, roads
