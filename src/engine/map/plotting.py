import matplotlib.pyplot as plt
from utils import timing

@timing
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
        plt.scatter(intersection[1], intersection[0], color='red', s=10)

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Roads and Intersections')

    plt.show()


def analyze_times():
    with open("execution_times.txt", "r") as file:
        lines = file.readlines()

    # Créer un dictionnaire pour stocker le total du temps d'exécution par fonction
    execution_times = {}
    total_time = 0

    for line in lines:
        function_name, execution_time = line.strip().split(',')
        execution_time = float(execution_time)

        if function_name not in execution_times:
            execution_times[function_name] = 0
        execution_times[function_name] += execution_time

        total_time += execution_time

    # Créer un bar chart
    function_names = list(execution_times.keys())
    times = [execution_times[fn] for fn in function_names]
    percentages = [ (t / total_time) * 100 for t in times]

    plt.barh(function_names, percentages)
    plt.xlabel('Percentage of Total Execution Time')
    plt.title('Function Execution Time Analysis')
    plt.gca().invert_yaxis()  # Inverser l'ordre pour avoir la fonction la plus lente en haut
    plt.show()