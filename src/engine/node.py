from __future__ import annotations
from engine.constants import MAX_MOVABLES_IN_NODE
from engine.strategies.model.cross_duplex_strategy import CrossDuplexStrategy
from engine.strategies.model.strategy import Strategy

from engine.strategies.strategy_mutator import StrategyTypes
from .road import CRoad, Road
from .traffic.flow_controller import FlowController
from .movable.movable import Movable
from .utils import circle_collision, vecteur, scalaire, norm
from typing import List
import build.engine_ia as engine_ia

nid = 0


class CNode:

    def update(self, strategy: Strategy, tick: int):
        pass

    def get_id(self) -> int:
        pass

    def get_x(self) -> float:
        pass

    def get_y(self) -> float:
        pass

    def set_position(self, x, y):
        pass

    def set_path(self, dest: CNode, prev: CNode):
        pass

    def get_road_in(self) -> list[CRoad]:
        pass

    def get_road_out(self) -> list[CRoad]:
        pass


class Node:

    strategy: Strategy = None
    cnode: CNode = None
    # paths = {}
    controllers: List[FlowController]

    def __init__(self, x: float, y: float):
        self.cnode = engine_ia.Node(x, y)
        self.controllers = []

    def get_x(self) -> float:
        return self.cnode.get_x()

    def get_y(self) -> float:
        return self.cnode.get_y()

    def get_position(self) -> (float, float):
        return (self.get_x(), self.get_y())
    def set_position(self, x, y):
        self.cnode.set_position(x, y)

    def set_path(self, dest: Node, prev: Node):
        self.cnode.set_path(dest.cnode, prev.cnode)

    def get_road_in(self) -> list[CRoad]:
        return self.cnode.get_road_in()

    def get_road_out(self) -> list[CRoad]:
        return self.cnode.get_road_out()

    def __str__(self) -> str:
        # return f'{{"position":{{"x":{self.position[0]}, "y":{self.position[1]}}},"controllers":{"[]" if len(self.controllers) == 0 else "["+", ".join([controller.__str__() for controller in self.controllers])+"]"}}}'
        return f"todo"
    # def printPath(self):
    #     print(self._id)
    #     for p in self.paths:
    #         print(f"{p._id,self.paths[p]._id}", end='|')
    #     print("")

    def set_strategy(self, strategy: Strategy):
        self.strategy = strategy
