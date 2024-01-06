import time
import random
from engine.simulation import Simulation
import cProfile
sim_seed = int(time.time())
random.seed(0)
sim_duration = 60 * 60 * 60
map_file = "src/benchmark/benchmark_dataset/Dataset1/map.csv"
paths_file = "src/benchmark/benchmark_dataset/Dataset1/paths.csv"
simulation = Simulation(map_file=map_file, paths_file=paths_file, nb_movables=800)

def simuma():
    simulation.run(sim_duration=sim_duration)


cProfile.run("simuma()")
# simuma()

# if __name__ == "__main__":
#     main()