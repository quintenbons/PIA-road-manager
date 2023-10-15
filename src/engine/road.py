import numpy as np
from dataclasses import dataclass
from .types import Coordinate
from typing import Optional


@dataclass
class Road:
    start: Coordinate
    end: Coordinate
    _norm: Optional[Coordinate] = None

    def norm(self) -> Coordinate:
        if self._norm is None:
            self._norm = (self.end - self.start) / np.linalg.norm(self.end - self.start)
        return self._norm
