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
    
    def collisionDetection(self) -> List[Movable, Movable]:

        collision = []
        for lane in self.content:
            #TODO use a binary search tree to optimize this
            
            for movable, otherMovable in filter(lambda obj: (obj[0] != obj[1]), zip(lane, lane)):
                m1, m2 = (movable, otherMovable) if movable.pos < otherMovable.pos else (otherMovable, movable)
                
                if(m1.pos + m1.size > m2.pos - m2.size):
                    collision.append([m1, m2])
                    #TODO m1.triggerBehaviour() is probably better than this
                    
        return collision

    def removeMovable(self, mov):
        #TODO optimize this
        for lane in self.content:
            try:
                lane.remove(mov)
            except ValueError:
                continue