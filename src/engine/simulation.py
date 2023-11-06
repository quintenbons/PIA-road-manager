"""
Simulation of the road network
"""
from .movable.car import Car
from .road import Road
from .node import Node
import time
from typing import List
from dataclasses import dataclass

@dataclass
class Simulation:
    cars: List[Car]
    roads: List[Road]
    nodes: List[Node]
    timestamp: int = 0
    speed: int = 5

    def __init__(self, speed: int = 5):
        self.speed = speed
        self.cars = []
        self.roads = []
        self.nodes = []
        self.time = 0

    def add_node(self, node: Node):
        self.nodes.append(node)

    def add_car(self, car: Car):
        self.cars.append(car)

    def add_road(self, road: Road):
        self.roads.append(road)

    def next_tick(self):
        for node in self.nodes:
            node.update(self.time)
        self.time += self.speed

    def end(self) -> bool:
        return len(self.cars) == 0

    def run(self):
        while not self.end():
            self.next_tick()
            for node in self.nodes:
                node.update(self.time)
            for car in self.cars:
                car.update()
                if car.next_road is None and car.pos == 1:
                    self.cars.remove(car)

            print(f'{{ "state": {self.time}, "node": [{", ".join([node.__str__() for node in self.nodes])}], "cars": [{", ".join([car.__str__() for car in self.cars])}] }}')
            print("--------------------------------------------------")