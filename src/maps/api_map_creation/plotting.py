import matplotlib.pyplot as plt
from .utils import timing
from .utils import BUILD_DIR
import os


# @timing
def plot_roads_and_intersections(roads):
    """
    Plot the roads on a graph. Intersections are identified by the endpoints of the roads.
    """
    print("Plotting data...")
    plt.figure(figsize=(10, 10))

    # Set to keep track of intersections already plotted
    plotted_intersections = set()

    for road in roads:
        x_coords = [point[1] for point in road]
        y_coords = [point[0] for point in road]
        
        # Plot the road
        plt.plot(x_coords, y_coords, color='blue', linewidth=1)
        
        # Plot the intersections (endpoints of the roads)
        for point in road:
            point_tuple = tuple(point)
            if point_tuple not in plotted_intersections:
                plt.scatter(point[1], point[0], color='red', s=10)
                plotted_intersections.add(point_tuple)

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Roads and Intersections')
    plt.show()


def analyze_times():
    file_path = os.path.join(BUILD_DIR, "execution_times.txt")
    with open(file_path, "r") as file:
        lines = file.readlines()

    # dictionnaire pour stocker le total du temps d'exécution par fonction
    execution_times = {}
    total_time = 0

    for line in lines:
        function_name, execution_time = line.strip().split(',')
        execution_time = float(execution_time)

        if function_name not in execution_times:
            execution_times[function_name] = 0
        execution_times[function_name] += execution_time

        total_time += execution_time

    function_names = list(execution_times.keys())
    times = [execution_times[fn] for fn in function_names]
    percentages = [(t / total_time) * 100 for t in times]

    plt.barh(function_names, percentages)
    plt.xlabel('Percentage of Total Execution Time')
    plt.title('Function Execution Time Analysis')
    plt.gca().invert_yaxis()  # fonction la plus lente en haut
    plt.show()

    os.remove(file_path)
