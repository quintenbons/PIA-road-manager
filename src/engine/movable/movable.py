from __future__ import annotations
from enum import Enum



import sys
from typing import TYPE_CHECKING
from engine.tree import Nodable, Node


if TYPE_CHECKING:
    sys.path.append('../engine')
    from engine.road import Road

class Movable(Nodable):

    road: Road
    lane: int = 0
    speed: float = 0.0
    latency: float = 0.0
    pos: float = 0.0
    size: float
    category: Enum
    currentRoad: Road

    _roadNode: Node
    
    def __init__(self, speed, latency, pos, size):
        self.speed = speed
        self.latency = latency
        self.pos = pos
        self.size = size
    def update(self) -> None:
        #TODO what happen
        self.pos = self.nextPosition()

    def setRoad(self, road: Road):
        self.road = road

    def nextPosition(self) -> float:
        return self.pos + self.speed
    
    def changeRoad(self, Road):
        pass

    def handleTrafficLight(self):
        pass
    def handleStop(self):
        pass
    def handleCrosswalk(self):
        pass

    def handlePossibleCollision(self, other: Movable):
        pass

    def maxValue(self):
        return self.pos + self.size
    def minValue(self):
        return self.pos - self.size
    
    def bindTree(self, node: Node):
        self._roadNode = node

    def getNode(self) -> Node:
        return self._roadNode
    
    def __str__(self):
        return str(self.pos)

    

class Category(Enum):
    CAR = 0
    BIKE = 1
    HUMAN = 2
