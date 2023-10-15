import numpy as np
from dataclasses import dataclass
from .types import Coordinate
from .road import Road

@dataclass
class Car:
    road: Road
    pos: int = 0

    def move(self):
        # TODO: end of road, car in front etc.
        self.pos += 1

    def get_coord(self) -> Coordinate:
        return self.road.start + self.road.norm() * self.pos
