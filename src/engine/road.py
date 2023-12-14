from __future__ import annotations
from dataclasses import dataclass
from typing import List, TYPE_CHECKING
from .tree import BinarySearchTree
from .utils import getLength

if TYPE_CHECKING:
    from .movable.movable import Movable
    from .node import Node

rid = 0
# @dataclass

class Road:
    start: Node
    end: Node

    pos_start: tuple(float, float)
    pos_end: tuple(float, float)
    road_len: float

    bidirectional: bool = True
    length: float
    speedLimit: float = 50

    block_traffic: bool = False

    ai_flow_count: List[int] # road in, road out
    _id: int
    _numberOfLane: int = 1
    lanes: List[BinarySearchTree[Movable]] = None

    def __init__(self, start: Node, end: Node, speedLimit: float):
        self.length = getLength(start.position, end.position)

        # assert(self.length > 30)

        ux = end.position[0] - start.position[0]
        uy = end.position[1] - start.position[1]
        u_norm = (ux*ux+uy*uy)**0.5
        ux /= u_norm
        uy /= u_norm

        vx = uy
        vy = -ux

        self.pos_start = list(start.position)
        self.pos_end = list(end.position)
        self.road_len = self.length - 10
        self.pos_start[0] += 5*ux + 2*vx
        self.pos_start[1] += 5*uy + 2*vy
        self.pos_end[0] += -5*ux + 2*vx
        self.pos_end[1] += -5*uy + 2*vy

        
        self.start = start
        self.end = end
        self.speedLimit = speedLimit

        self.start.add_road_out(self)
        self.end.add_road_in(self)
        self.lanes = [BinarySearchTree() for _ in range(self._numberOfLane)]
        self.ai_flow_count = [0, 0]
        global rid
        self._id = rid
        rid += 1

    def update(self) -> None:
        for lane in self.lanes:
            previous: Movable = None
            #previous is ahead
            for mov in lane.iter(True):
                mov: Movable
                assert(previous != mov)
                if previous is None:
                    mov.handle_first_movable()
                else:
                    self.collision_detection(previous, mov)
                mov.handle_road_target()
                previous = mov

    # No need for collision detection??
    def collision_detection(self, previous: Movable, nxt: Movable) -> float:
        # previous is ahead and nxt is behind on the road
        if previous is None:
            return
        if nxt.next_position()[0] + 2*nxt.size > previous.next_position()[0] - 2*previous.size:
            nxt.handle_possible_collision(previous)
        else:
            nxt.no_possible_collision(previous)

    def add_movable(self, movable: Movable, lane: int):
        assert(movable.tree_node is None)
        if self.lanes[lane].insert(movable):
            self.ai_flow_count[0] += 1
            movable.set_road(self)
            return True
        # print("Can't add movable")
        self.ai_flow_count[0] += 1
        return False

    def spawn_movable(self, movable: Movable, lane: int):
        assert(movable.tree_node is None)
        if self.lanes[lane].insert(movable):

            movable.set_road(self)
            return True
        return False

    def remove_movable(self, mov: Movable):
        mov.getTreeNode().remove()
        self.ai_flow_count[1] += 1

    def despawn_movable(self, mov: Movable):
        mov.getTreeNode().remove()

    def get_length(self):
        return self.length

    def __str__(self) -> str:
        return f'{{"start": {self.start._id}, "end": {self.end._id}, "length": {self.length}, "bidirectional": "{self.bidirectional}", "speedLimit": {self.speedLimit}}}'
