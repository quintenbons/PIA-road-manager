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

    _id: int
    # _length: float
    # _speedLimit: float = 50
    _numberOfLane: int = 1
    # _isOneWay: bool = True
    # _trafficFlow: float = None
    # _avgSpeed: float = None
    lanes: List[BinarySearchTree[Movable]] = None

    def __init__(self, start: Node, end: Node, speedLimit: float):
        self.length = getLength(start.position, end.position)

        assert(self.length > 30)

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
        self.pos_end[0] -= 5*ux + 2*vx
        self.pos_end[1] -= 5*uy + 2*vy

        
        self.start = start
        self.end = end
        self.speedLimit = speedLimit

        self.start.add_road_out(self)
        self.end.add_road_in(self)
        self.lanes = [BinarySearchTree() for _ in range(self._numberOfLane)]
        global rid
        self._id = rid
        rid += 1

    def update(self) -> None:
        for lane in self.lanes:
            previous: Movable = None
            for mov in lane:
                mov: Movable
                self.collisionDetection(previous, mov)
                mov.update()
                previous = mov
            if(previous):
                previous.no_possible_collision(None)

    # No need for collision detection??
    def collisionDetection(self, previous: Movable, nxt: Movable) -> float:
        if previous is None:
            return
        if previous.next_position() + 2*previous.size > nxt.next_position() - 2*nxt.size:
            previous.handle_possible_collision(nxt)
        else:
            previous.no_possible_collision(nxt)

    def add_movable(self, movable: Movable, lane: int):
        self.lanes[lane].insert(movable)
        movable.set_road(self)

    def remove_movable(self, mov: Movable):
        mov.getTreeNode().remove()

    def get_length(self):
        return self.length

    def __str__(self) -> str:
        return f'{{"start": {self.start._id}, "end": {self.end._id}, "length": {self.length}, "bidirectional": "{self.bidirectional}", "speedLimit": {self.speedLimit}}}'
