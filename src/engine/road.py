from __future__ import annotations
from dataclasses import dataclass
from typing import List, TYPE_CHECKING
from .tree import BinarySearchTree

if TYPE_CHECKING:
    from .movable.movable import Movable
    from .node import Node

@dataclass
class Road:
    start: Node
    end: Node
    _length: float
    _speedLimit: float
    _numberOfLane: int = 1
    _isOneWay: bool = True
    _trafficFlow: float = None
    _avgSpeed: float = None
    content: List[BinarySearchTree[Movable]] = None

    def update(self) -> None:
        for lane in self.content:
            previous: Movable = None
            for mov in lane:
                mov: Movable
                if previous is not None:
                    self.collisionDetection(previous, mov)
                mov.update()
                


    def collisionDetection(self, previous: Movable, nxt: Movable) -> float:
        if previous.nextPosition() + previous.size > nxt.nextPosition() - nxt.size:
            previous.handlePossibleCollision(nxt)

    def removeMovable(self, mov: Movable):
        #TODO optimize this
        for lane in self.content:
            try:
                lane.remove(mov.getNode())
            except ValueError:
                continue