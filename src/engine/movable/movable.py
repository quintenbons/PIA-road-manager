from __future__ import annotations

import sys
from typing import TYPE_CHECKING, List
from engine.tree import Nodable, TreeNode
from .utils import car_speed, car_position, mov_node_position

from ..utils import getLength

from ..constants import TIME

sys.path.append('../maps')


if TYPE_CHECKING:
    sys.path.append('../engine')
    
    from engine.road import Road
    from engine.node import Node
    

mid = 0

class Movable(Nodable):

    road: Road
    node: Node = None
    lane: int = 0
    speed: float = 0.0
    acceleration: float = 0.0

    current_acceleration: float = 0.0
    latency: float = 0.0
    # Pos = 0 <=> start of the road, Pos = 1 <=> end of the road
    pos: float = 0.0
    node_pos: list(float, float)
    node_dir: tuple(float, float)
    node_len: float 
    size: float = 1.0

    path: List[Node] = None
    tree_node = None
    _id : int

    def __init__(self, speed, acceleration, latency, pos, size):
        self.speed = speed
        self.acceleration = acceleration
        
        self.latency = latency
        self.pos = pos
        self.size = size

        global mid
        self._id = mid
        mid += 1


    def update_position(self):
        self.pos, self.speed = self.next_position()

    def next_position(self) -> list(float, float):
        sp = self.speed + TIME*self.current_acceleration
        if sp > self.road.speedLimit and self.current_acceleration > 0:
            t = (self.road.speedLimit - self.speed)/self.current_acceleration
            assert(t >= 0)
            pos = TIME*t*self.current_acceleration/2 + self.speed*t + self.road.speedLimit*(TIME-t) + self.pos
            speed = self.road.speedLimit
        elif sp < 0:
            t = -self.speed/self.current_acceleration
            pos = TIME*t*self.current_acceleration/2 + self.speed*t + self.pos
            speed = 0
        else:
            pos = TIME*TIME*self.current_acceleration/2 + self.speed*TIME + self.pos
            speed = sp
        assert(speed <= self.road.speedLimit)
        return pos, speed

    def handle_possible_collision(self, other: Movable):
        dx = (other.next_position()[0] - 2*other.size) - (self.next_position()[0] + 2*self.size)
        if dx <= 0:
            self.speed = 0
            self.current_acceleration = 0
        else:
            self.current_acceleration = -1/dx

    def no_possible_collision(self, other: Movable):
        if not other:
            self.current_acceleration = self.acceleration
        else:
            future_other = other.next_position()[0] - other.size
            future_self = self.next_position()[0] + self.size
            dx = min(future_other - future_self, future_other - future_self - self.size)

            da = 1.75*dx/TIME/TIME
            self.current_acceleration = min(self.acceleration, self.current_acceleration + da)


    def update_road(self) -> None:
        self.pos, self.speed = self.next_position()
        if self.pos >= self.road.road_len:
            if len(self.path) == 0:
                self.pos = 0
                self.road.remove_movable(self)
                self.tree_node = None
                # self.road = None
                self.node = None
                return False
            
            next_node = self.path.pop(-1)
            self.node = next_node
            self.node.movables.append(self)

            next_road = self.find_next_road(next_node)
            self.road.remove_movable(self)
            self.tree_node = None
            self.pos = 0
            self.node_pos = self.road.pos_end

            x1, y1 = self.road.pos_end
            x2, y2 = next_road.pos_start
            norm = getLength((x1, y1), (x2, y2))

            ux, uy = (x2-x1)/norm, (y2 - y1)/norm
            self.node_dir = (ux, uy)
            self.speed = 0
            self.current_acceleration = 1
            self.node_len = getLength(self.road.pos_end, next_road.pos_start)

            self.road = next_road

        return True
    
    def next_node_position(self):
        pos, speed = self.next_position()
        pos = min(pos, self.node_len)

        nx, ny = self.node_pos
        dx, dy = self.node_dir
        return pos, speed, (nx + pos*dx, ny + pos * dy)
    
    def update_node(self):
        self.pos, self.speed, self.node_pos = self.next_node_position()

        if self.pos >= self.node_len:
            self.pos = 0
            self.node.movables.remove(self)
            self.node = None

    def update(self):
        if self.tree_node is None and self.node is None:
            return False
        if self.node is None and self.road:
            self.update_road()
            return True
        elif self.node:
            self.update_node()
            return True
        return False

    def set_road(self, road: Road):
        self.road = road

    def stop(self):
        self.speed = 0

    def get_path(self, arrival: Node):
        from maps.maps import find_path
        self.path = find_path(self.road.end, arrival)
        assert(self.road.end == self.path.pop(-1))

    def find_next_road(self, next_node: Node):
        current_node = self.road.end
        next_road = current_node.road_to(next_node)
        #TODO things
        return next_road

    def getTreeNode(self):
        return self.tree_node

    def maxValue(self):
        return self.pos + self.size
    
    def minValue(self):
        return self.pos - self.size
    
    def bindTree(self, tree_node: TreeNode):
        self.tree_node = tree_node

    def __str__(self):
        return f'{{"pos": {self.pos}, "speed": {self.speed}, "latency": {self.latency}, "size": {self.size}, "road": {self.road}}}'

