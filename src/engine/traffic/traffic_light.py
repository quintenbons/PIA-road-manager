from .flow_controller import FlowController
from .crosswalk import Crosswalk
from typing import List

class TrafficLight(FlowController):
    crosswalk: Crosswalk | None
    flags: List[bool] = []

    def __init__(self, road_in, road_out, crosswalk: Crosswalk = None):
        self.crosswalk = crosswalk
        self.road_in = road_in
        self.road_out = road_out

        for road in self.road_out:
            self.flags.append(True)

    def set_flag(self, v, i) -> None:
        if i < len(self.flags):
            self.flags[i] = v

    def is_open(self, road) -> bool:
        for i in range(len(self.road_out)):
            if self.road_out[i] == road:
                return self.flags[i]
        return False
    
    def is_one_open(self) -> bool:
        for flag in self.flags:
            if flag:
                return True
        return False

    def update(self, time) -> None:
        if self.crosswalk is not None:
            self.crosswalk.update(time, self.is_one_open())