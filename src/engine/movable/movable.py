from __future__ import annotations

import sys
from typing import TYPE_CHECKING, List
from engine.tree import Nodable, TreeNode
from .utils import car_speed, car_position

sys.path.append('../maps')


if TYPE_CHECKING:
    sys.path.append('../engine')
    
    from engine.road import Road
    from engine.node import Node
    

mid = 0

class Movable(Nodable):

    road: Road
    node: Node
    lane: int = 0
    speed: float = 0.0
    acceleration: float = 0.0

    current_acceleration: float = 0.0
    latency: float = 0.0
    # Pos = 0 <=> start of the road, Pos = 1 <=> end of the road
    pos: float = 0.0
    node_pos: tuple(float, float) = (0.0, 0.0)
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

    def next_position(self):
        return car_position(self.road.road_len, self.pos, self.speed)
    
    def handle_possible_collision(self, other: Movable):
        dx = (other.pos - 2*other.size) - (self.pos + 2*self.size)
        print(f"mid = {self._id}, hpc = {-1.5/dx}, dx = {(other.pos - 2*other.size) - (self.pos + 2*self.size)}")
        if dx < 0:
            dxx = (other.pos - 1.1*other.size) - (self.pos + 1.1*self.size)
            if dxx < 0:
                print("emergency stop")
                self.speed = 0
            else:
                self.current_acceleration = -3/dxx
        else:
            self.current_acceleration = -1/dx

    def no_possible_collision(self, other: Movable):
        if not other:
            self.current_acceleration = self.acceleration
        else:
            dx = abs((other.pos - other.size) - (self.pos + self.size))
            print(f"mid = {self._id}, npc = {self.acceleration* (1 - 1.5/dx)}, dx = {dx}, op = {other.pos}, sp = {self.pos}")
            self.current_acceleration = self.acceleration* (1 - 1.5/dx)

    def update(self) -> None:
        self.speed = car_speed(self.road.speedLimit, self.speed, self.current_acceleration)
        self.pos = car_position(self.road.road_len, self.pos, self.speed)

        if self.pos >= self.road.road_len:
            if len(self.path) == 0:
                self.pos = 0
                self.road.remove_movable(self)
                self.road = None
                return False
            
            #TODO move between nodes
            next_node = self.path.pop(-1)
            next_road = self.find_next_road(next_node)
            self.pos = 0
            self.road.remove_movable(self)
            self.road = next_road
            self.road.add_movable(self, 0)
        return True

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

