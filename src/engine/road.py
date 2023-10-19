import numpy as np
from dataclasses import dataclass
from .types import Coordinate
from typing import Optional, List
from .movable import Movable

@dataclass
class Road:
    start: Coordinate
    end: Coordinate
    speedLimit: int
    numberOfLane: int
    _norm: Optional[Coordinate] = None
    _trafficFlow: float = None
    _avgSpeed: float = None

    content: List[Movable] = []

    def update(self) -> None:
        for m in self.content:
            m.update()
        

    def norm(self) -> Coordinate:
        if self._norm is None:
            self._norm = (self.end - self.start) / np.linalg.norm(self.end - self.start)
        return self._norm
