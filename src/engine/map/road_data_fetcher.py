import matplotlib.pyplot as plt

from processing import process_road_data
from plotting import plot_roads_and_intersections


# Process the road data
# intersections, roads = process_road_data("Saint-Trinit")
# intersections, roads = process_road_data("Revest-du-Bion")
# intersections, roads = process_road_data("Eybens")
intersections, roads = process_road_data("Grenoble")




plot_roads_and_intersections(roads, intersections)
