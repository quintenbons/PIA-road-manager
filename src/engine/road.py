from __future__ import annotations
from dataclasses import dataclass
from typing import List, TYPE_CHECKING
from .tree import BinarySearchTree
from .utils import getLength
import build.engine_ia as engine_ia

if TYPE_CHECKING:
    from .movable.movable import Movable
    from .node import Node

rid = 0
# @dataclass

class Road:
    
    croad = None

    def __init__(self, start: Node, end: Node, speedLimit: float):
        self.croad = engine_ia.Road(start.cnode, end.cnode, speedLimit)

    def __str__(self) -> str:
        # return f'{{"start": {self.start._id}, "end": {self.end._id}, "length": {self.length}, "bidirectional": "{self.bidirectional}", "speedLimit": {self.speedLimit}}}'
        return f"todo"