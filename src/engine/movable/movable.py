from __future__ import annotations
from enum import Enum



import sys
from typing import TYPE_CHECKING
from engine.tree import Nodable


if TYPE_CHECKING:
    sys.path.append('../engine')
    from engine.road import Road

class Movable(Nodable):

    speed: float = 0.0
    latency: float = 0.0
    pos: float = 0.0
    size: float
    category: Enum
    currentRoad: Road

    def update(self) -> None:
        #TODO what happen
        self.pos += self.speed
    
    def changeRoad(self, Road):
        pass

    def handleTrafficLight(self):
        pass
    def handleStop(self):
        pass
    def handleCrosswalk(self):
        pass

    def maxValue(self):
        return self.pos + self.size
    def minValue(self):
        return self.pos - self.size
    
class Category(Enum):
    CAR = 0
    BIKE = 1
    HUMAN = 2
