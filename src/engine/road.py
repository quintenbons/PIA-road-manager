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
    content: List[BinarySearchTree] = None

    def update(self) -> None:
        for lane in self.content:
            for mov in lane:
                mov.update()
    
    # TODO call before update
    def collisionDetection(self) -> None:
        #TODO clear intentation hell
        for lane in self.content:
            previous: Movable = None
            for movable in lane.iterate():
                movable: Movable
                if previous is not None:
                    if previous.nextPosition() + previous.size > movable.nextPosition() - movable.size:
                        previous.handlePossibleCollision(movable)

    def removeMovable(self, mov):
        #TODO optimize this
        for lane in self.content:
            try:
                lane.remove(mov)
            except ValueError:
                continue