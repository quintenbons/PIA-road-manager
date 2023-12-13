#!/usr/bin/env python3
import sys
import os
from engine.constants import GENERATION_SEGMENT_DUARTION, TIME
from engine.spawners.spawner import Spawner
from engine.spawners.spawner_utils import every_ten_seconds

from engine.strategies.strategies_manager import StrategyManager
from engine.strategies.strategy_mutator import StrategyTypes
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

    def __init__(self, map_file: str, paths_file: str, nb_movables: int = 0):
        random.seed(0)

        self.strategy_manager = StrategyManager()

        self.roads: List[Road]
        self.nodes: List[Node]
        self.roads, self.nodes = read_map(map_file)
        read_paths(self.nodes, paths_file)
        set_traffic_lights(self.nodes)
        set_strategies(self.nodes, self.strategy_manager)

        self.spawners: List[Spawner] = []
        spawner = Spawner(self.roads, self.roads, every_ten_seconds, nb_movables)
        self.spawners.append(spawner)


    def run(self, sim_duration: int = GENERATION_SEGMENT_DUARTION):
        start_tick = self.current_tick

        while (self.current_tick - start_tick) * TIME < sim_duration:
            self.run_tick()

    def run_tick(self):
        for r in self.roads:
            r.update()
        for n in self.nodes:
            n.update(self.current_tick)
        self.current_tick += 1
        for s in self.spawners:
            s.update(self.current_tick * TIME)

    def get_total_score(self) -> int:
        """Get total simulation score (not node specific)

        Kind of see through for spawner.get_total_score"""
        total_active_score = 0
        for s in self.spawners:
            total_active_score += s.get_total_score(self.current_tick)
        return total_active_score

    def reset_score(self):
        """Reset score of previously despawned movables (still maintains ongoing movables)"""
        for s in self.spawners:
            s.reset_score()

    def set_node_strategy(self, node_id: int, type: StrategyTypes, mutation: int):
        target_node = self.nodes[node_id]
        strat = self.strategy_manager.get_strategy(target_node, type, mutation)
        target_node.set_strategy(strat)

if __name__ == "__main__":
    simulation = Simulation()
    simulation.run()