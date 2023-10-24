from processing import process_road_data
from plotting import plot_roads_and_intersections,analyze_times

# Process the road data
# intersections, roads = process_road_data("Saint-Trinit")
# intersections, roads = process_road_data("Revest-du-Bion")
# intersections, roads = process_road_data("Grenoble")
intersections, roads = process_road_data("Eybens")

plot_roads_and_intersections(roads, intersections)

analyze_times()