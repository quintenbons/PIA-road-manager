import requests
import json
from shapely.geometry import LineString, MultiPoint
import matplotlib.pyplot as plt

def fetch_road_data(city_name):
    """
    Fetches the road data for the specified city from the OpenStreetMap database.
    """
    # Define the Overpass query. This query fetches highways in the specified city.
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
    
    # Define the Overpass API URL
    overpass_url = "http://overpass-api.de/api/interpreter"
    
    # Send the request to the Overpass API and get the JSON response
    print("Sending the request to the Overpass API...")
    response = requests.get(overpass_url, params={'data': overpass_query})
    data = response.json()
    
    # Save the JSON data to a file for easier inspection and debugging
    print("Saving the JSON data to a file...")
    with open('road_data.json', 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    
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


def process_road_data(filename='road_data.json'):
    with open(filename, 'r') as file:
        data = json.load(file)
        
    nodes = {element['id']: (element['lat'], element['lon']) 
             for element in data['elements'] 
             if element['type'] == 'node'}
    
    roads = []
    for element in data['elements']:
        if element['type'] == 'way':
            road_nodes = [nodes[node_id] for node_id in element['nodes']]
            roads.append(road_nodes)
            
    intersections = find_intersections(roads)
    
    return intersections, roads

# Call the function with the name of your city
road_data = fetch_road_data("Saint-Trinit")

# Process the road data
intersections, roads = process_road_data()

# write the intersections to a file
with open('intersections.txt', 'w') as file:
    for intersection in intersections:
        file.write(f'{intersection[0]},{intersection[1]}\n')

# write the roads to a file
with open('roads.txt', 'w') as file:
    for road in roads:
        for node in road:
            file.write(f'{node[0]},{node[1]}\n')
        file.write('\n')

def plot_roads_and_intersections(roads, intersections):
    """
    Plot the roads and intersections on a graph.
    """
    # Création d'une nouvelle figure
    plt.figure(figsize=(10, 10))
    
    # Tracer les routes
    for road in roads:
        x_coords = [node[1] for node in road]  # Les longitudes
        y_coords = [node[0] for node in road]  # Les latitudes
        plt.plot(x_coords, y_coords, color='blue', linewidth=1)
        
    # Tracer les intersections
    for intersection in intersections:
        plt.scatter(intersection[1], intersection[0], color='red', s=40)  # Les longitudes et latitudes sont inversées
    
    # Configuration des labels et du titre
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Roads and Intersections')
    
    # Affichage du graphique
    plt.show()

# Appelez cette fonction avec vos données de routes et d'intersections
plot_roads_and_intersections(roads, intersections)