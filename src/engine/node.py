from __future__ import annotations
from engine.constants import MAX_MOVABLES_IN_NODE
from engine.strategies.model.strategy import Strategy

from engine.strategies.strategy_mutator import StrategyTypes
from .road import Road
from .traffic.flow_controller import FlowController
from .movable.movable import Movable
from .utils import circle_collision, vecteur, scalaire, norm
from typing import List

nid = 0

class Node:
    position: tuple[float, float]
    road_in: List[Road]
    road_out: List[Road]
    controllers: List[FlowController]
    strategy: Strategy = None

    paths: dict[Node: Node]
    movables: List[Movable]
    _id: int

    def __init__(self, x: float, y: float):
        #TODO remove it later
        global nid
        self._id = nid
        nid += 1
        self.road_in = []
        self.road_out = []
        self.controllers = []
        self.position = (x, y)
        self.paths = {}
        self.movables = []

    def update(self, time) -> None:
        self.strategy.update(time)
        n = len(self.movables)
        # print(n)
        for i in range(n):
            for j in range(i + 1, n):
                movable1 = self.movables[i]
                movable2 = self.movables[j]
                pos1, speed1, node_pos1 = movable1.next_node_position()
                pos2, speed2, node_pos2 = movable2.next_node_position()

                A = movable1.node_pos
                B = movable2.node_pos
                O = vecteur(A, B)
                U = vecteur(A, node_pos1)
                V = vecteur(B, node_pos2)
                
                N = (U[1], -U[0])
                #BA et V
                sca = scalaire(N, O)
                priority_mov = None
                stop_mov = None
                if sca > 0:
                    priority_mov = movable2
                    stop_mov = movable1
                else:
                    priority_mov = movable1
                    stop_mov = movable2
                
                if circle_collision(node_pos1, node_pos2, movable1.size, movable2.size):
                    # notifier la voiture à droite qu'elle doit stoper
                    stop_mov.notify_node_collision()
                    priority_mov.notify_node_priority()

                det = V[0]*U[1]-V[1]*U[0]
                t = (V[0]*O[1]-O[0]*V[1])

                if det != 0:
                    t /= det
                
                    P = (A[0] + t*U[0], A[1] + t*U[1])

                    AP = vecteur(A, P)
                    BP = vecteur(B, P)

                    sca1 = scalaire(AP, U)
                    sca2 = scalaire(BP, V)

                    if sca1 >= 0 and sca2 >= 0:
                        if norm(AP) < norm(U) and norm(BP) < norm(V):
                            stop_mov.notify_node_collision()
                            priority_mov.notify_node_priority()


    def position_available(self, pos, size):

        if len(self.movables) > MAX_MOVABLES_IN_NODE:
            return False
        for m in self.movables:
            #TODO use next node pos
            # print(m.node_pos, pos, m.size, size)
            if circle_collision(m.node_pos, pos, m.size, size):
                # print("collision ?")
                return False
        return True

    def add_road_in(self, road: Road):
        self.road_in.append(road)

    def add_road_out(self, road: Road):
        self.road_out.append(road)

    def pathTo(self, node: Node) -> List[Node]:
        return self.paths[node]

    def road_to(self, node: Node) -> Road:
        # precondition : there is a road in between this node and the next
        # TODO : decide if we change data structure to add roads inside paths
        dist = float('inf')
        road = None
        for r in self.road_out:
            if(r.end == node and r.length < dist):
                dist = r.length
                road = r
        assert(road)
        return road
    
    def neighbors(self) -> Node:
        for r in self.road_out:
            yield r.end, r.length()

    def __str__(self) -> str:
        return f'{{"position":{{"x":{self.position[0]}, "y":{self.position[1]}}},"controllers":{"[]" if len(self.controllers) == 0 else "["+", ".join([controller.__str__() for controller in self.controllers])+"]"}}}'

    def printPath(self):
        print(self._id)
        for p in self.paths:
            print(f"{p._id,self.paths[p]._id}", end='|')
        print("")
        
    def set_strategy(self, strategy: Strategy):
        self.strategy = strategy
