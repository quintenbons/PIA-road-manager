from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import TYPE_CHECKING

# if TYPE_CHECKING:
sys.path.append('../engine')
from engine.road import Road

@dataclass
class Car:
    road: Road
    pos: int = 0

    def move(self):
        # TODO: end of road, car in front etc.
        self.pos += 1

    # def get_coord(self) -> Coordinate:
    #     return self.road.start + self.road.norm() * self.pos
