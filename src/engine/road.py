from __future__ import annotations
from dataclasses import dataclass
from typing import List, TYPE_CHECKING
from .tree import BinarySearchTree
from .utils import getLength

if TYPE_CHECKING:
    from .movable.movable import Movable
    from .node import Node

rid = 0
# @dataclass
class Road:
    start: Node
    end: Node
    bidirectional: bool = True
    length: float
    speedLimit: float = 50
    
    # _id: int
    # _length: float
    # _speedLimit: float = 50
    # _numberOfLane: int = 1
    # _isOneWay: bool = True
    # _trafficFlow: float = None
    # _avgSpeed: float = None
    # lanes: List[BinarySearchTree[Movable]] = None
    
    def __init__(self, start, end, speedLimit = 50, isBidirectional = True):
        self.start = start
        self.end = end
        self.bidirectional = isBidirectional
        self.length = getLength(start.position, end.position)
        self.speedLimit = speedLimit

        self.start.add_road_out(self)
        self.end.add_road_in(self)

        if self.bidirectional:
            self.end.add_road_out(self)
            self.start.add_road_in(self)

    # def __init__(self, start: Node, end: Node, length: float, speedLimit: float):
    #     self.lanes = [BinarySearchTree() for _ in range(self._numberOfLane)]
    #     self.start = start
    #     self.end = end
    #     self._length = length
    #     self._speedLimit = speedLimit
    #     start.addRoadOut(self)
    #     end.addRoadIn(self)
    #     #TODO remove it later
    #     global rid
    #     self._id = rid
    #     rid += 1

    # def update(self) -> None:
    #     for lane in self.lanes:
    #         previous: Movable = None
    #         for mov in lane:
    #             mov: Movable
    #             self.collisionDetection(previous, mov)
    #             mov.update()

    # No need for collision detection??
    # def collisionDetection(self, previous: Movable, nxt: Movable) -> float:
    #     if previous is None:
    #         return
    #     if previous.nextPosition() + previous.size > nxt.nextPosition() - nxt.size:
    #         previous.handlePossibleCollision(nxt)

    # def addMovable(self, movable: Movable, lane: int):
    #     self.lanes[lane].insert(movable)
    #     movable.setRoad(self)
        
    # def remove_movable(self, mov: Movable):
    #     mov.getNode().remove()

    def __str__(self) -> str:
        return f'{{"start": {self.start._id}, "end": {self.end._id}, "length": {self.length}, "bidirectional": "{self.bidirectional}", "speedLimit": {self.speedLimit}}}'