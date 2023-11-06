from __future__ import annotations
from enum import Enum

import sys
from typing import TYPE_CHECKING
from engine.tree import Nodable
from .utils import carSpeed

if TYPE_CHECKING:
    sys.path.append('../engine')
    from engine.road import Road

class Movable(Nodable):

    road: Road
    lane: int = 0
    speed: float = 0.0
    latency: float = 0.0
    # Pos = 0 <=> start of the road, Pos = 1 <=> end of the road
    pos: float = 0.0
    size: float = 1.0
    category: Enum
    currentRoad: Road

    # Represent the next road the vehicle will take
    next_road: Road = None

    def __init__(self, speed, latency, pos, size):
        self.speed = speed
        self.latency = latency
        self.pos = pos
        self.size = size

    def update(self) -> None:
        self.speed = carSpeed(self.road.speedLimit, self.speed)
        self.pos = min(1, self.pos + self.speed)

        if self.pos == 1:
            if self.next_road is None:
                return # TODO: Finish

            next_node = self.get_next_node()
            can_travel = next_node.try_to_travel_to(self.road, self.next_road)
            if not can_travel:
                self.stop()
            else:
                self.pos = 0
                self.road = self.next_road
                self.next_road = None
            
    def get_next_node(self):
        if self.road.end == self.next_road.start or self.road.end == self.next_road.end:
            return self.road.end
        if self.road.start == self.next_road.start or self.road.start == self.next_road.end:
            return self.road.start

    def setRoad(self, road: Road):
        self.road = road

    def stop(self):
        self.speed = 0

    def __str__(self):
        return f'{{"pos": {self.pos}, "speed": {self.speed}, "latency": {self.latency}, "size": {self.size}, "road": {self.road}, "next_road": {"None" if self.next_road is None else self.next_road}}}'

    

class Category(Enum):
    CAR = 0
    BIKE = 1
    HUMAN = 2
