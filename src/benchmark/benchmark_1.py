import time
import random

from engine.simulation import Simulation
from ai.model_constants import *

class Benchmark1():
    def run():
        sim_seed = int(time.time())
        map_file = "src/benchmark/benchmark_dataset/Dataset1/map.csv"
        paths_file = "src/benchmark/benchmark_dataset/Dataset1/paths.csv"
        sim_duration = 60 * 60 * 6

        random.seed(sim_seed)
        simulation = Simulation(map_file=map_file, paths_file=paths_file, nb_movables=0)

        t = time.time()
        print("Launching simulation #1")
        simulation.run(sim_duration=sim_duration)
        print("Time to run simulation #1: ", time.time() - t)