from processing import process_road_data
from plotting import plot_roads_and_intersections,analyze_times
import time

# Process the road data
start = time.time()

# roads = process_road_data("Saint-Trinit")
# roads = process_road_data("Revest-du-Bion")
roads = process_road_data("Eybens", GENERATE_CSV=True)  #  laissez à true pour le moment, false pas encore totalement fonctionnel
# roads = process_road_data("Grenoble", True)

print("Time of processing road creation: ", time.time() - start)

start = time.time()
plot_roads_and_intersections(roads)
print("Time of plotting: ", time.time() - start)

analyze_times()