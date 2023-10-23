import requests
import json
from shapely.geometry import LineString, MultiPoint, Point
import matplotlib.pyplot as plt
import csv
import os

OVERPASS_URL = "http://overpass-api.de/api/interpreter"
BUILD_DIR = os.path.join(os.path.dirname(__file__), 'build')


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


def fetch_road_data(city_name):
    """
    Fetches the road data for the specified city from the OpenStreetMap database.
    """
    # Requête Overpass pour fetch les routes (highways), données non triées
    overpass_query = f"""
    [out:json];
    area[name="{city_name}"];
    (way[highway](area);
     way[railway](area);
    );
    out body;
    >;
    out skel qt;
    """

    print("Fetching data from Overpass API...")
    response = requests.get(OVERPASS_URL, params={'data': overpass_query})
    data = response.json()

    return data


def find_intersections(roads):
    intersections = set()

    for i, road1 in enumerate(roads):
        for j, road2 in enumerate(roads):
            if i != j:
                line1 = LineString(road1)
                line2 = LineString(road2)

                if line1.intersects(line2):
                    intersection = line1.intersection(line2)
                    if intersection.is_empty:
                        continue

                    if intersection.geom_type == 'Point':
                        intersections.add((intersection.x, intersection.y))
                    elif intersection.geom_type == 'MultiPoint':
                        for point in intersection.geoms:
                            intersections.add((point.x, point.y))
                    elif intersection.geom_type == 'LineString':
                        for point in intersection.coords:
                            intersections.add(point)

    return list(intersections)


def simplify_roads(roads, intersections):
    simplified_roads = []

    intersection_points = [Point(i) for i in intersections]

    for road in roads:
        road_line = LineString(road)

        # Trouver les points d'intersection entre la route et les intersections
        intersecting_points = [
            p for p in intersection_points if p.intersects(road_line)]

        # Si il y a des intersections, diviser la route
        if intersecting_points:
            # Assurez-vous que les extrémités de la route sont incluses
            intersecting_points.extend([Point(road[0]), Point(road[-1])])

            # Supprimer les doublons et trier les points le long de la route
            intersecting_points = sorted(
                set(intersecting_points), key=lambda p: road_line.project(p))

            # Créer des segments de route entre chaque paire de points d'intersection
            for i in range(len(intersecting_points) - 1):
                start = intersecting_points[i].coords[0]
                end = intersecting_points[i+1].coords[0]
                simplified_roads.append([start, end])

        else:
            # Si il n'y a pas d'intersections, garder la route telle quelle
            simplified_roads.append(road)

    return simplified_roads


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

    if simplify:
        print("Simplifying roads...")
        roads = simplify_roads(roads, intersections)
        intersections = find_intersections(roads)

    # Sauvegarder les intersections et les routes dans des fichiers CSV
    print("Saving data...")
    save_to_csv('intersections.csv', intersections)
    save_to_csv('roads.csv', [[point for point in road] for road in roads])

    return intersections, roads


# Process the road data
intersections, roads = process_road_data("Sault")


def plot_roads_and_intersections(roads, intersections):
    """
    Plot the roads and intersections on a graph.
    """
    print("Plotting data...")
    plt.figure(figsize=(10, 10))

    for road in roads:
        x_coords = [node[1] for node in road]
        y_coords = [node[0] for node in road]
        plt.plot(x_coords, y_coords, color='blue', linewidth=1)

    for intersection in intersections:
        # Les longitudes et latitudes sont inversées
        plt.scatter(intersection[1], intersection[0], color='red', s=40)

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Roads and Intersections')

    plt.show()


plot_roads_and_intersections(roads, intersections)
