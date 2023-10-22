"""
Simulation of the road network
"""
from .movable.car import Car
from .road import Road
from typing import List

class Simulation:
    cars: List[Car] = []
    roads: List[Road] = []

    def add_car(self, car: Car):
        self.cars.append(car)

    def add_road(self, road: Road):
        self.roads.append(road)

    def next_tick(self):
        for car in self.cars:
            car.move()
