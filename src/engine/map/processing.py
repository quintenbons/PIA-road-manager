import ast
import csv
import os
from shapely.geometry import LineString
import networkx as nx
from rtree import index
from fetching import fetch_road_data
from file_manager import save_to_csv
from utils import timing, BUILD_DIR

from file_manager import ROADS_FILE, SIMPLIFIED_ROADS_FILE, INTERSECTIONS_FILE, LARGEST_CONNECTED_COMPONENT, FINAL_ROADS_FILE, MAP_FILE

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
def simplify_roads(roads):
    simplified_roads = []
    for road in roads:
        if road:
            simplified_road = [road[0], road[-1]]
            simplified_roads.append(simplified_road)
    return simplified_roads

@timing
def create_raw_data(largest_connected_component):
    raw_data = []
    for road in largest_connected_component:
        if len(road) == 2:  # Assurez-vous que chaque route a deux points (début et fin)
            start, end = road
            raw_data.append((f'"{start[0]}, {start[1]}"', f'"{end[0]}, {end[1]}"'))
    return raw_data

@timing
def extract_connex_roads_from_csv(roads):
    G = nx.Graph()

    for road in roads:
        G.add_edge(road[0], road[1])
    
    largest_cc = max(nx.connected_components(G), key=len)

    largest_subgraph = G.subgraph(largest_cc)

    road_complex = []
    filename = os.path.join(BUILD_DIR, LARGEST_CONNECTED_COMPONENT)
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

def extract_connex_roads(roads):
    G = nx.Graph()

    # Ajout des routes au graphe
    for road in roads:
        G.add_edge(road[0], road[1])

    # Trouver le plus grand composant connecté
    largest_cc = max(nx.connected_components(G), key=len)
    largest_subgraph = G.subgraph(largest_cc).edges()

    # Extraire les routes correspondant au plus grand composant connecté
    road_complex = [(edge[0], edge[1]) for edge in largest_subgraph]

    return road_complex

def read_raw_data():
    filename = os.path.join(BUILD_DIR, FINAL_ROADS_FILE)
    with open(filename, 'r') as file:
        return [line.strip().split('","') for line in file]

@timing
def create_nodes_and_routes(raw_data):
    nodes = {}
    routes = []
    node_index = 0

    for pair in raw_data:
        start, end = pair
        start = start.replace('"', '')
        end = end.replace('"', '')

        if start not in nodes:
            nodes[start] = node_index
            node_index += 1
        if end not in nodes:
            nodes[end] = node_index
            node_index += 1

        routes.append((nodes[start], nodes[end]))

    return nodes, routes

def write_to_csv(nodes, routes, output_file):
    output_filepath = os.path.join(BUILD_DIR, output_file)
    with open(output_filepath, 'w') as file:
        for node, index in nodes.items():
            x, y = node[1:-1].split(', ')
            file.write(f"{x} {y} : {index}\n")

        file.write("===\n")

        for start, end in routes:
            file.write(f"{start} {end}\n")


@timing
def process_road_data(city_name, GENERATE_CSV=False):
    road_data = fetch_road_data(city_name)

    print("Parsing data...")
    nodes = {element['id']: (element['lat'], element['lon'])
             for element in road_data['elements']
             if element['type'] == 'node'}

    roads = [ [nodes[node_id] for node_id in element['nodes']] 
              for element in road_data['elements']
              if element['type'] == 'way']

    print("Finding intersections...")
    intersections = find_intersections(roads)

    if GENERATE_CSV:
        print("Saving data...")
        save_to_csv(INTERSECTIONS_FILE, intersections)
        save_to_csv(ROADS_FILE, [[point for point in road] for road in roads])

        roads = simplify_roads_csv(ROADS_FILE, SIMPLIFIED_ROADS_FILE)

        print("Extract largest connected component...")
        roads = extract_connex_roads_from_csv(roads)
        # write roads to file
        save_to_csv(FINAL_ROADS_FILE, [[point for point in road] for road in roads])
        
        raw_data = read_raw_data()
        nodes, routes = create_nodes_and_routes(raw_data)
        write_to_csv(nodes, routes, MAP_FILE)
    else:
        roads = simplify_roads(roads)

        largest_connected_component = extract_connex_roads(roads)

        raw_data = create_raw_data(largest_connected_component)

        nodes, routes = create_nodes_and_routes(raw_data)

        write_to_csv(nodes, routes, MAP_FILE)

    return roads
