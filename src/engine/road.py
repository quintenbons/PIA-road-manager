from __future__ import annotations
from dataclasses import dataclass
from typing import List, TYPE_CHECKING
from .tree import BinarySearchTree

if TYPE_CHECKING:
    from .movable.movable import Movable
    from .node import Node

rid = 0
# @dataclass
class Road:
    start: Node
    end: Node
    _id: int
    _length: float
    _speedLimit: float
    _numberOfLane: int = 1
    _isOneWay: bool = True
    _trafficFlow: float = None
    _avgSpeed: float = None
    lanes: List[BinarySearchTree[Movable]] = None
    
    def __init__(self, start: Node, end: Node, length: float, speedLimit: float):
        self.lanes = [BinarySearchTree() for _ in range(self._numberOfLane)]
        self.start = start
        self.end = end
        self._length = length
        self._speedLimit = speedLimit
        start.addRoadOut(self)
        end.addRoadIn(self)
        #TODO remove it later
        global rid
        self._id = rid
        rid += 1

    def update(self) -> None:
        for lane in self.lanes:
            previous: Movable = None
            for mov in lane:
                mov: Movable
                
                self.collisionDetection(previous, mov)
                mov.update()

    def collisionDetection(self, previous: Movable, nxt: Movable) -> float:
        if previous is None:
            return
        if previous.nextPosition() + previous.size > nxt.nextPosition() - nxt.size:
            previous.handlePossibleCollision(nxt)

    def addMovable(self, movable: Movable, lane: int):
        self.lanes[lane].insert(movable)
        movable.setRoad(self)
        
    def removeMovable(self, mov: Movable):
        mov.getNode().remove()

    def __str__(self) -> str:
        return f"{self.start} -> {self._id} -> {self.end}"