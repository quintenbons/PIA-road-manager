from __future__ import annotations
from engine.constants import MAX_MOVABLES_IN_NODE
from engine.strategies.model.cross_duplex_strategy import CrossDuplexStrategy
from engine.strategies.model.strategy import Strategy

from engine.strategies.strategy_mutator import StrategyTypes
from .road import Road
from .traffic.flow_controller import FlowController
from .movable.movable import Movable
from .utils import circle_collision, vecteur, scalaire, norm
from typing import List
import build.engine_ia as engine_ia

nid = 0

class Node:

    strategy: Strategy = None
    cnode = None
    # paths = {}
    controllers: List[FlowController]
    def __init__(self, x: float, y: float):
        self.cnode = engine_ia.Node(x, y)
        self.controllers = []

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
