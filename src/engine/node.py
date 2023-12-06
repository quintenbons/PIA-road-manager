from __future__ import annotations
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
        for controller in self.controllers:
            controller.update(time)
        n = len(self.movables)
        # print(n)
        for i in range(n):
            for j in range(i + 1, n):
                movable1 = self.movables[i]
                movable2 = self.movables[j]
                # if circle_collision(movable1.node_pos, movable2.node_pos, movable1.size, movable2.size):
                    # print("Col in node")
                    # exit(0)
                pos1, speed1, node_pos1 = movable1.next_node_position()
                pos2, speed2, node_pos2 = movable2.next_node_position()

                if circle_collision(node_pos1, node_pos2, movable1.size, movable2.size):
                    # print("collision")
                    pass
                    #TODO gérer cela, par exemple, si une voiture attend à un feu on peut ignorer
                    # exit(0)
                A = movable1.node_pos
                B = movable2.node_pos
                O = vecteur(A, B)
                U = vecteur(node_pos1, A)
                V = vecteur(node_pos2, B)

                t = (V[0]*O[1]-O[0]*V[1])

                det = V[1]*U[0]-V[0]*U[1]
                if det != 0:
                    t /= det
                
                    P = (A[0] + t*U[0], A[1] + t*U[1])

                    AP = vecteur(P, A)
                    BP = vecteur(P, B)

                    sca1 = scalaire(AP, U)
                    sca2 = scalaire(BP, V)

                    if sca1 >= 0 and sca2 >= 0:
                        if norm(AP) < norm(U) and norm(BP) < norm(V):
                            # print("Collision ici")
                            # exit(0)
                            pass


    def position_available(self, pos, size):
        for m in self.movables:
            #TODO use next node pos
            if circle_collision(m.node_pos, pos, m.size, size):
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
