from .road import Road
from .traffic.flow_controller import FlowController
from typing import List

nid = 0

class Node:
    position: tuple[float, float]
    road_in: List[Road]
    road_out: List[Road]
    controllers: List[FlowController]

    _id: int

    def __init__(self, x: float, y: float):
        #TODO remove it later
        global nid
        self._id = nid
        nid += 1
        self.road_in = []
        self.road_out = []
        self.controllers = []
        self.position = (x, y)

    def update(self, time) -> None:
        for controller in self.controllers:
            controller.update(time)

    def try_to_travel_to(self, road_from, road_to):
        if road_from not in self.road_in:
            return False
        if road_to not in self.road_out:
            return False
        for controller in self.controllers:
            if controller.road_in == road_from  and road_to in controller.road_out:
                return controller.is_open(road_to)

    def is_allowed_to_travel(self, road_from, road_to):
        return road_from 

    def add_road_in(self, road: Road):
        self.road_in.append(road)

    def add_road_out(self, road: Road):
        self.road_out.append(road)

    def __str__(self) -> str:
        return f'{{"position":{{"x":{self.position[0]}, "y":{self.position[1]}}},"controllers":{"[]" if len(self.controllers) == 0 else "["+", ".join([controller.__str__() for controller in self.controllers])+"]"}}}'
