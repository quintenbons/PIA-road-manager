from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, TYPE_CHECKING

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
    content: List[List[Movable]] = None

    def update(self) -> None:
        for m in self.content:
            m.update()
    
    def collisionDetection(self) -> List[Movable, Movable]:

        collision = []
        for lane in self.content:
            #TODO use a binary search tree to optimise this
            for movable, otherMovable in filter(lambda obj: (obj[0] != obj[1]), zip(lane, lane)):
                m1, m2 = (movable, otherMovable) if movable.pos < otherMovable.pos else (otherMovable, movable)
                
                if(m1.pos + m1.size > m2.pos - m2.size):
                    collision.append([m1, m2])
                    #TODO m1.triggerBehaviour() is probably better than this
                    
        return collision

