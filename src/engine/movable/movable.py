from __future__ import annotations

import sys
from typing import TYPE_CHECKING, List
from engine.tree import Nodable, TreeNode
from ..utils import getLength, vecteur_norm, scalaire

from ..constants import LEAVING_DIST, LEAVING_TIME, TIME, TIME2, TIME_DIV_175, TIME_DIV_25

import engine_ia as engine_ia

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
    pos: float = 0.0

    nxt_pos: float = 0.0
    nxt_speed: float = 0.0

    acceleration: float = 0.0

    current_acceleration: float = 0.0
    latency: float = 0.0
    # Pos = 0 <=> start of the road, Pos = 1 <=> end of the road
    node_pos: list(float, float)
    node_dir: tuple(float, float)
    node_mov: bool = True
    node_len: float 
    size: float = 1.0
    spawn_tick: int = 0

    path: List[Node] = None
    tree_node = None
    _id : int

    road_goal: List(Road, float) = None
    inner_timer: float = 0

    def __init__(self, speed, acceleration, latency, pos, size, spawn_tick: int = 0):
        self.speed = speed
        self.acceleration = acceleration
        
        self.latency = latency
        self.pos = pos
        self.size = size
        self.spawn_tick = spawn_tick

        global mid
        self._id = mid
        mid += 1


    def update_position(self):
        self.pos, self.speed = self.nxt_pos, self.next_speed()

    def next_pos(self) -> float:
        tca = TIME*self.current_acceleration
        sp = self.speed + tca
        # if self.inner_timer > 0:
        #     return self.pos
        if sp > self.road.speedLimit and self.current_acceleration > 0:
            t = (self.road.speedLimit - self.speed)/self.current_acceleration
            pos = t*tca/2 + self.speed*t + self.road.speedLimit*(TIME-t) + self.pos
        elif sp < 0:
            t = -self.speed/self.current_acceleration
            pos = t*tca/2 + self.speed*t + self.pos
        else:
            pos = TIME*tca/2 + self.speed*TIME + self.pos
        return pos
    
    def next_speed(self) -> float:
        sp = self.speed + TIME*self.current_acceleration
        if self.inner_timer > 0:
            return 0
        if sp > self.road.speedLimit and self.current_acceleration > 0:
            speed = self.road.speedLimit
        elif sp < 0:
            speed = 0
        else:
            speed = sp
        # assert(speed <= self.road.speedLimit)
        return speed     
    def next_position(self) -> list(float, float):
        sp = self.speed + TIME*self.current_acceleration
        if self.inner_timer > 0:
            speed = 0
            pos = self.pos
            return pos, speed
        if sp > self.road.speedLimit and self.current_acceleration > 0:
            t = (self.road.speedLimit - self.speed)/self.current_acceleration
            # assert(t >= 0)
            pos = TIME*t*self.current_acceleration/2 + self.speed*t + self.road.speedLimit*(TIME-t) + self.pos
            speed = self.road.speedLimit
        elif sp < 0:
            t = -self.speed/self.current_acceleration
            pos = TIME*t*self.current_acceleration/2 + self.speed*t + self.pos
            speed = 0
        else:
            pos = TIME*TIME*self.current_acceleration/2 + self.speed*TIME + self.pos
            speed = sp
        # assert(speed <= self.road.speedLimit)
        return pos, speed
    
    def handle_road_target(self):
        dx = self.road_goal[1] - self.pos
        # assert(dx > 0)
        # #TODO correction
        if self.road == self.road_goal[0] and 0 < dx < LEAVING_DIST:
            da = 0
            if self.speed < 0.5:
                return
            else:
                da = (0.5 - self.speed - TIME*self.current_acceleration)/TIME
                if da > 0:
                    da = 0
            self.current_acceleration += da
            self.nxt_pos = self.next_pos()

    def handle_first_movable(self):

        # future_pos, _ = self.next_position()
        if self.pos >= self.nxt_pos:
            self.current_acceleration = self.acceleration
            self.nxt_pos = self.next_pos()
        # future_pos, _ = self.next_position()
        
        dx = self.road.road_len - self.nxt_pos
        # self.current_acceleration = max(0, self.current_acceleration)
        if self.current_acceleration < 0:
            self.current_acceleration = 0
            self.nxt_pos = self.next_pos()
            # TODO investigate : dx = self.road.road_len - self.nxt_pos

        if self.road.block_traffic:
            da = TIME_DIV_175*dx if dx > 0 else  TIME_DIV_25*dx
            self.current_acceleration = min(self.acceleration, self.current_acceleration + da)
            self.nxt_pos = self.next_pos()
            return
        if self.nxt_pos > self.road.road_len and len(self.path) > 0:
            # Leaving the road
            # Check if it can leave or not
            # Check for slowing down
            # next_road = self.find_next_road(self.path[-1])
            #direction = vecteur_norm(self.road.pos_end, next_road.pos_start)
            # print(self.road.end.position, self.road.pos_end)
            if not self.road.end.position_available(self.road.pos_end, self.size):
                # print("je me bloque ?")
                da = TIME_DIV_25*dx # < 0
                self.current_acceleration = min(self.acceleration, self.current_acceleration + da)
                self.nxt_pos = self.next_pos()

            #TODO add a way to check for other roads
    def handle_possible_collision(self, other: Movable):
        da = TIME_DIV_25*((other.nxt_pos - other.size) - (self.nxt_pos + self.size))
        if da <= 0:
            # da = *dx
            self.current_acceleration += da

        else:
            # da = 2.5*dx/TIME/TIME
            self.current_acceleration -= da
        self.nxt_pos = self.next_pos()

    def no_possible_collision(self, other: Movable):

        # assert(other is not None)
        # self.current_acceleration = max(0, self.current_acceleration)
        if self.current_acceleration < 0:
            self.current_acceleration = 0
            self.nxt_pos = self.next_pos()
        future_other = other.nxt_pos - other.size
        future_self = self.nxt_pos + self.size
        dx = min(future_other - future_self, future_other - future_self - self.size)

        da = TIME_DIV_175*dx
        # acceleration to speedLimit
        damax = (self.road.speedLimit - self.speed)/TIME - self.current_acceleration
        da = min(da, damax)
        self.current_acceleration = min(self.acceleration, self.current_acceleration + da)
        self.nxt_pos = self.next_pos()


    def update_road(self) -> None:
        self.pos, self.speed = self.nxt_pos, self.next_speed()
        #TODO handle going further road_len
        if self.road == self.road_goal[0] and self.pos > self.road_goal[1]:
            timer = self.inner_timer * TIME
            if timer <= LEAVING_TIME:
                self.speed = 0
                self.acceleration = 0
                self.current_acceleration = 0
                self.inner_timer += 1
                return True
            self.road.despawn_movable(self)
            self.tree_node = None
            self.node = None
            return False
        if self.pos >= self.road.road_len and not self.road.block_traffic:
            
            next_node = self.path.pop(-1)
            self.node = next_node
            self.node.movables.append(self)

            next_road = self.find_next_road(self.path[-1])
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
    
    def notify_node_collision(self):
        # print(f"my id is : {self._id}")
        # pass
        self.speed = 0
        self.current_acceleration = 0
        self.node_mov = False
    
    def notify_node_priority(self):
        self.current_acceleration = self.acceleration/2

    def next_node_position(self):
        pos, speed = self.next_position()
        pos = min(pos, self.node_len)

        nx, ny = self.node_pos
        dx, dy = self.node_dir
        return pos, speed, (nx + (pos - self.pos)*dx, ny + (pos - self.pos) * dy)
    
    def update_node(self):
        tmp_pos, tmp_node_pos = self.pos, self.node_pos

        self.pos, self.speed, self.node_pos = self.next_node_position()
        if self.node_mov and self.pos <= tmp_pos:
            self.current_acceleration = self.acceleration
        if self.pos >= self.node_len:
            self.pos = 0
            if not self.road.add_movable(self, 0):
                self.pos = self.node_len
                self.node_pos = tmp_node_pos
            else:
                self.node.movables.remove(self)
                self.node = None
        self.node_mov = True

    def update(self):
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
        # assert(self.road.end == self.path.pop(-1))

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
        if self.node is not None:
            return self.node_pos
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

    def get_score(self, current_tick) -> int:
        """Game score. Play with this, and the AI will try to minimize it."""
        return (current_tick - self.spawn_tick) ** 2

    def __str__(self):
        return f'{{(x,y): {self.to_coord_xy()}, "pos on the road": {self.pos}, "speed": {self.speed}, "latency": {self.latency}, "size": {self.size}, "node": {self.node} "road": {self.road}, "id": {self._id}}}'

