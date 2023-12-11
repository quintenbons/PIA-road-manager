#!/usr/bin/env python3
import sys
import os

from engine.strategies.strategies_manager import StrategyManager
sys.path.append(os.path.dirname(__file__))
from random import randint, random, seed
from maps.maps_functions import read_map, read_paths, set_strategies, set_traffic_lights
from engine.movable.movable import Movable
from engine.road import Road
from engine.node import Node
from typing import List

class Simulation:
    strategy_manager: StrategyManager
    current_tick: int = 0

    def __init__(self, map_file: str, paths_file: str,debug_mode: bool = False):
        self.debug_mode = debug_mode
        print("\n\n ---------------------------------- \n")

        self.strategy_manager = StrategyManager()

        self.roads: List[Road]
        self.nodes: List[Node]
        self.roads, self.nodes = read_map(map_file)
        read_paths(self.nodes, paths_file)
        set_traffic_lights(self.nodes)
        set_strategies(self.nodes, self.strategy_manager)
        self.movables: List[Movable] = []

    def add_movables(self, count: int = 1):
        for _ in range(count):
            r = self.roads[randint(0, len(self.roads) - 1)]
            m = Movable(5, 2, random(), random() * (r.road_len), 2)
            if r.add_movable(m, 0):
                m.get_path(self.nodes[randint(0, len(self.nodes) - 1)])
                self.movables.append(m)

    def run(self):
        seed(0)

    def run_tick(self):
        for r in self.roads:
            r.update()
        for n in self.nodes:
            n.update(self.current_tick)
        for m in self.movables:
            # TODO: wtf is this mess?
            if not m.update():
                # self.movables.remove(m)
                # m.pos = m.road.road_len - 5
                # m.pos = 0
                if m.road.add_movable(m, 0):
                    u = randint(0, len(self.nodes) - 1)
                    # print(u)
                    m.get_path(self.nodes[u])

if __name__ == "__main__":
    simulation = Simulation(debug_mode=True)
    simulation.run()