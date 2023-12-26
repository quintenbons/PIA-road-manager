from __future__ import annotations

import sys
from typing import TYPE_CHECKING, List
from engine.tree import Nodable, TreeNode
from ..utils import getLength, vecteur_norm, scalaire

from ..constants import LEAVING_DIST, LEAVING_TIME, TIME, TIME2, TIME_DIV_175, TIME_DIV_25

import build.engine_ia as engine_ia

sys.path.append('../maps')


if TYPE_CHECKING:
    sys.path.append('../engine')
    
    from engine.road import Road
    from engine.node import Node
    

mid = 0

class Movable(Nodable):

  

    cmovable = None

    def __init__(self, speed, acceleration, pos, size, spawn_tick: int = 0):

        self.cmovable = engine_ia.Movable(speed, acceleration, pos, size, spawn_tick, TIME)
    
    def get_score(self, current_tick):
        return self.cmovable.get_score(current_tick)
    
    def get_pos(self):
        return self.cmovable.get_pos()
    
    def set_road_goal(self, destination: Road, pos: float):
        self.cmovable.set_road_goal(destination.croad, pos)

    def get_id(self):
        self.cmovable.get_id()
    
    def to_coord_xy(self):
        self.cmovable.to_coord_xy()
    # def __str__(self):
    #     return f'{{(x,y): {self.to_coord_xy()}, "pos on the road": {self.pos}, "speed": {self.speed}, "latency": {self.latency}, "size": {self.size}, "node": {self.node} "road": {self.road}, "id": {self._id}}}'
    def __str__(self):
        return f"id = {self.cmovable.get_id()}"
