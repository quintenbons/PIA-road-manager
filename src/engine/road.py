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


class CRoad:

    def get_id(self) -> int:
        pass

    def update(self):
        pass

    def spawn_movable(self, movable: Movable) -> bool:
        pass

    def get_road_len(self) -> float:
        pass

    def get_speed_limit(self) -> float:
        pass

    def get_block_traffic(self) -> bool:
        pass

    def get_pos_start(self) -> (float, float):
        pass

    def get_pos_end(self) -> (float, float):
        pass

    def get_ai_flow_count_0(self) -> int:
        pass

    def get_ai_flow_count_1(self) -> int:
        pass

    def get_ai_wait_duration(self) -> float:
        pass

    def get_ai_wait_duration_cumulative(self) -> float:
        pass

    def reset_ai_metrics(self):
        pass


class Road:

    croad: CRoad = None

    def __init__(self, start: Node, end: Node, speedLimit: float):
        self.croad = engine_ia.Road(start.cnode, end.cnode, speedLimit)

    def spawn_movable(self, movable: Movable) -> bool:
        return self.croad.spawn_movable(movable.cmovable)

    def get_road_len(self) -> float:
        return self.croad.get_road_len()

    def get_speed_limit(self) -> float:
        return self.croad.get_speed_limit()

    def get_block_traffic(self) -> bool:
        return self.croad.get_block_traffic()
    
    def reset_ai_metrics(self):
        return self.croad.reset_ai_metrics()

    def get_pos_start(self) -> (float, float):
        return self.croad.get_pos_start()

    def get_pos_end(self) -> (float, float):
        return self.croad.get_pos_end()

    def get_ai_wait_duration(self) -> float:
        return self.croad.get_ai_wait_duration()

    def get_ai_wait_duration_cumulative(self) -> float:
        return self.croad.get_ai_wait_duration_cumulative()

    def __str__(self) -> str:
        # return f'{{"start": {self.start._id}, "end": {self.end._id}, "length": {self.length}, "bidirectional": "{self.bidirectional}", "speedLimit": {self.speedLimit}}}'
        return f"todo"
