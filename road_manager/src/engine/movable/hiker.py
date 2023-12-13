from .movable import Movable
from dataclasses import dataclass

import sys
from typing import TYPE_CHECKING
from ..road import Road

if TYPE_CHECKING:
    sys.path.append('../engine')
    from engine.road import Road

@dataclass
class Hiker(Movable):
    road: Road = None