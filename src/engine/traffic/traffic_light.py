from .flow_controller import FlowController
from .crosswalk import Crosswalk
from typing import List

class TrafficLight(FlowController):
    flag = False

    def __init__(self, road_in, road_out):
        self.road_in = road_in
        self.road_out = road_out
        self.pos = road_in.get_pos_end()
        self.flag = True

    def set_flag(self, v) -> None:
        self.flag = v
        self.road_in.set_block_traffic(not v)