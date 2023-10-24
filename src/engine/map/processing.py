import ast
import csv
import os
from shapely.geometry import LineString
import networkx as nx
from rtree import index
from fetching import fetch_road_data
from file_manager import save_to_csv
from utils import timing, BUILD_DIR

@timing
def find_intersections(roads):
    intersections = set()
    idx = index.Index()

    # Indexation des routes pour les rtrees
    for i, road in enumerate(roads):
        road_line = LineString(road)
        idx.insert(i, road_line.bounds)

    # Recherche d'intersections
    for i, road1 in enumerate(roads):
        road1_line = LineString(road1)
        # routes potentiellement intersectantes
        potential_matches = list(idx.intersection(road1_line.bounds))

        for j in potential_matches:
            # évite les comparaisons redondantes
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

    file_path = os.path.join(BUILD_DIR, input_filepath)
    output_filepath = os.path.join(BUILD_DIR, output_filepath)
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in reader:
            # Convert strings de tuples en tuples, todo: ne pas stocker les routes en string dans le csv
            first_point = ast.literal_eval(row[0])
            last_point = ast.literal_eval(row[-1])

            simplified_road = [first_point, last_point]
            simplified_roads.append(simplified_road)

    save_to_csv(output_filepath, simplified_roads)

    return simplified_roads

@timing
def extract_connex_roads(roads):
    G = nx.Graph()

    for road in roads:
        G.add_edge(road[0], road[1])
    
    largest_cc = max(nx.connected_components(G), key=len)

    largest_subgraph = G.subgraph(largest_cc)

    road_complex = []
    filename = os.path.join(BUILD_DIR, 'largest_connected_component.csv')
    with open(filename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Parcourir chaque bord du plus grand sous-graphe connecté
        for edge in largest_subgraph.edges():
            node1, node2 = edge
            
            # Écrire les nœuds de chaque bord dans le fichier CSV
            writer.writerow([str(node1), str(node2)])
            road = [node1, node2]
            road_complex.append(road)
    return road_complex
            

@timing
def process_road_data(city_name):
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

    print("Saving data...")
    save_to_csv('intersections.csv', intersections)
    save_to_csv('roads.csv', [[point for point in road] for road in roads])

    input_filepath = 'roads.csv'
    output_filepath = 'simplified_roads.csv'
    roads = simplify_roads_csv(input_filepath, output_filepath)

    print("Extract largest connected component...")
    roads = extract_connex_roads(roads)

    return roads
