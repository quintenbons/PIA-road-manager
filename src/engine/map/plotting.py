import matplotlib.pyplot as plt

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