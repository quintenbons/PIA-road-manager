from dataclasses import dataclass
from typing import List
from ..movable.hiker import Hiker
from ..movable.car import Car
from ..road import Road
class FlowController:
    vehicle_queue: List[Car] = []
    hiker_queue: List[Hiker] = []
    pos = tuple[float, float]
    road_in: Road = None
    road_out: List[Road] = []

    def update(self, time) -> None:
        pass

    def __init__(self):
        self.vehicle_queue = []
        self.hiker_queue = []

    def is_open(self, road) -> bool:
        pass

    def __str__(self) -> str:
        return f'{{"vehicle_queue":{[vehicle.__str__() for vehicle in self.vehicle_queue]},"hiker_queue":{[hiker.__str__() for hiker in self.hiker_queue]}, "roadIn":{self.road_in.__str__()},"roadOut":[{", ".join([road.__str__() for road in self.road_out])}]}}'