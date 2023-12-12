#!/usr/bin/env python3
import sys
import os
from engine.constants import GENERATION_SEGMENT_DUARTION, TIME

from engine.strategies.strategies_manager import StrategyManager
sys.path.append(os.path.dirname(__file__))
import random
from maps.maps_functions import read_map, read_paths, set_strategies, set_traffic_lights
from engine.movable.movable import Movable
from engine.road import Road
from engine.node import Node
from typing import List

class Simulation:
    strategy_manager: StrategyManager
    current_tick: int = 0

    def __init__(self, map_file: str, paths_file: str):
        self.strategy_manager = StrategyManager()

        self.roads: List[Road]
        self.nodes: List[Node]
        self.roads, self.nodes = read_map(map_file)
        read_paths(self.nodes, paths_file)
        set_traffic_lights(self.nodes)
        set_strategies(self.nodes, self.strategy_manager)
        self.movables: List[Movable] = []

        random.seed(0)

    def add_movables(self, count: int = 1):
        for _ in range(count):
            r = self.roads[random.randint(0, len(self.roads) - 1)]
            m = Movable(5, 2, random.random(), random.random() * (r.road_len), 2)
            if r.add_movable(m, 0):
                random_road = self.roads[random.randint(0, len(self.roads) - 1)]
                random_pos = random_road.road_len * (random.random()*0.8 + 0.1)
                m.set_road_goal(random_road, random_pos)
                self.movables.append(m)

    def run(self, sim_duration: int = GENERATION_SEGMENT_DUARTION):
        start_tick = self.current_tick

        while (self.current_tick - start_tick) * TIME < sim_duration:
            self.run_tick()

    def run_tick(self):
        for r in self.roads:
            r.update()
        for n in self.nodes:
            n.update(self.current_tick)
        remove_list = []
        for m in self.movables:
            if not m.update():
                remove_list.append(m)
        for m in remove_list:
            self.movables.remove(m)

        self.current_tick += 1

if __name__ == "__main__":
    simulation = Simulation()
    simulation.run()