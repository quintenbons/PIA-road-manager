from __future__ import annotations

import sys
from typing import TYPE_CHECKING, List
from engine.tree import Nodable, TreeNode
from random import randint
from ..utils import getLength, vecteur_norm, scalaire

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

    color: int
    path: List[Node] = None
    tree_node = None
    _id : int

    road_goal: List(Road, float) = None

    def __init__(self, speed, acceleration, latency, pos, size):
        self.speed = speed
        self.acceleration = acceleration
        
        self.latency = latency
        self.pos = pos
        self.size = size

        global mid
        self._id = mid
        mid += 1
        self.color = randint(0, 255)


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

    def handle_first_movable(self):

        future_pos, _ = self.next_position()
        if self.pos == future_pos:
            self.current_acceleration = self.acceleration
        future_pos, _ = self.next_position()
        
        dx = self.road.road_len - future_pos
        self.current_acceleration = max(0, self.current_acceleration)

        if self.road.block_traffic:
            da = 1.75*dx/TIME/TIME if dx > 0 else  2.5*dx/TIME/TIME
            self.current_acceleration = min(self.acceleration, self.current_acceleration + da)
            return
        if future_pos > self.road.road_len and len(self.path) > 0:
            # Leaving the road
            # Check if it can leave or not
            # Check for slowing down
            next_road = self.find_next_road(self.path[-1])
            direction = vecteur_norm(self.road.pos_end, next_road.pos_start)
            
            if not self.road.end.position_available(self.road.pos_end, self.size):
                da = 2.5*dx/TIME/TIME # < 0
                self.current_acceleration = min(self.acceleration, self.current_acceleration + da)

            #TODO add a way to check for other roads
    
        
    def handle_node_collision(self, other: Movable):
        ortho = (self.node_dir[1], -self.node_dir[0])
        scal = scalaire(ortho, other.node_dir)
        if scal > 0:
            # Priotité à droite
            if other.speed == 0 and other.current_acceleration <= 0:
                #TODO test if it works
                self.current_acceleration = self.acceleration
            else:
                self.speed = 0
                self.current_acceleration = 0
        else:
            # On a la priorité
            if self.speed == 0 and self.current_acceleration <= 0:
                #TODO test if it works
                other.current_acceleration = other.acceleration
            else:
                other.speed = 0
                other.current_acceleration = 0

    def handle_possible_collision(self, other: Movable):
        dx = (other.next_position()[0] - 1*other.size) - (self.next_position()[0] + 1*self.size)
        if dx <= 0:
            da = 2.5*dx/TIME/TIME
            self.current_acceleration += da
        else:
            da = 2.5*dx/TIME/TIME
            self.current_acceleration -= da

    def no_possible_collision(self, other: Movable):
        if not other:
            self.current_acceleration = self.acceleration
        else:
            self.current_acceleration = max(0, self.current_acceleration)
            future_other = other.next_position()[0] - other.size
            future_self = self.next_position()[0] + self.size
            dx = min(future_other - future_self, future_other - future_self - self.size)

            da = 1.75*dx/TIME/TIME
            self.current_acceleration = min(self.acceleration, self.current_acceleration + da)


    def update_road(self) -> None:
        self.pos, self.speed = self.next_position()
        #TODO handle going further road_len
        if self.road == self.road_goal[0] and self.pos > self.road_goal[1]:
            self.pos = self.road.road_len
            self.road.despawn_movable(self)
            self.tree_node = None
            self.node = None
            return False
        if self.pos >= self.road.road_len and not self.road.block_traffic:
            
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
            if not self.road.add_movable(self, 0):
                self.pos = self.node_len
            else:
                self.node.movables.remove(self)
                self.node = None


    def update(self):
        if self.tree_node is None and self.node is None:
            return False
        if self.node is None and self.road:
            return self.update_road()
        elif self.node:
            self.update_node()
            return True
        return False

    def set_road(self, road: Road):
        self.road = road

    def stop(self):
        self.speed = 0

    def set_road_path(self, arrival: Road):
        from maps.maps_functions import find_path
        self.path = [arrival.end]
        self.path += find_path(self.road.end, arrival.start)
        assert(self.road.end == self.path.pop(-1))

    def set_road_goal(self, arrival: Road, pos):
        self.road_goal = [arrival, pos]
        self.set_road_path(arrival)

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

    def to_coord_xy(self):        
        road = self.road
        pos_start = road.pos_start
        pos_end = road.pos_end
        ux, uy = pos_end[0] - pos_start[0], pos_end[1] - pos_start[1]

        length = (ux*ux + uy*uy)**0.5
        ux, uy = ux/length, uy/length
        pos = self.pos

        x = pos_start[0] + pos*ux
        y = pos_start[1] + pos*uy

        return x, y

    def __str__(self):
        return f'{{(x,y): {self.to_coord_xy()}, "pos on the road": {self.pos}, "speed": {self.speed}, "latency": {self.latency}, "size": {self.size}, "road": {self.road}}}'

