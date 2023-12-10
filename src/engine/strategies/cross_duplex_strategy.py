from ..traffic.traffic_light import TrafficLight
from typing import List
from .strategy import Strategy
import math

class TrafficGroup:
    traffic_lights: set[TrafficLight]
    pos: tuple[float, float]

    def __init__(self):
        self.traffic_lights = set()
        self.pos = (None, None)

    def add_traffic_light(self, traffic_light: TrafficLight):
        self.traffic_lights.add(traffic_light)
        if self.pos is None:
            self.pos = traffic_light.pos
        else:
            # This is wrong code, it should set the position to the center of all traffic lights, we need to discuss together to find the correct way to do it
            self.pos = (0, 0)
            x = 0
            y = 0
            for traffic_light in self.traffic_lights:
                x += traffic_light.pos[0]
                y += traffic_light.pos[1]
            x /= len(self.traffic_lights)
            y /= len(self.traffic_lights)
            self.pos = (x, y)


class CrossDuplexStrategy(Strategy):
    traffic_lights_group: List[TrafficGroup]
    node_pos: tuple[float, float]
    opposit_degree_treshold :int= 15

    # 
    def __init__(self, controllers:List[TrafficLight], position: tuple[float, float], cycles:List[int]):
        super().__init__(controllers, cycles)

        self.traffic_lights_group = []
        self.node_pos = position

        # print("--------- node pos -----------", self.node_pos)
        # print("traffic lights")
        # i = 0
        # for traffic_light in self.trafficLights:
        #     print(i, traffic_light.pos)
        #     i += 1

        for traffic_light in self.trafficLights:
            self.associate_to_group(traffic_light)

        # print("traffic lights group")
        # i = 0
        # for group in self.traffic_lights_group:
        #     print(i, group.pos)
        #     for traffic_light in group.traffic_lights:
        #         print("    ", traffic_light.pos)
        #     i += 1
        # print("-------------------------------")

        self.stateCount = len(self.traffic_lights_group)

    def associate_to_group(self, traffic_light: TrafficLight):
        for group in self.traffic_lights_group:
            if self.is_opposit(self.node_pos, traffic_light.road_in.pos_end, group.pos):
                group.add_traffic_light(traffic_light)
                return
        group = TrafficGroup()
        group.add_traffic_light(traffic_light)
        self.traffic_lights_group.append(group)

    def is_opposit(self, center:tuple[float, float], a:tuple[float, float], b:tuple[float, float]) -> bool:
        return abs(self.get_angle(center, a) % 180 - self.get_angle(center, b) % 180) < self.opposit_degree_treshold
    
    def get_angle(self, center:tuple[float, float], point:tuple[float, float]) -> float:
        x, y = point[0] - center[0], point[1] - center[1]
        angle_rad = math.atan2(y, x)
        angle_deg = math.degrees(angle_rad)
        return angle_deg
    
    def next(self):
        super().next()
        activeGroup = self.traffic_lights_group[self.currentState]
        for traffic_light in activeGroup.traffic_lights:
            traffic_light.set_flag(True)

        for i in range(len(self.traffic_lights_group)):
            if i != self.currentState:
                for traffic_light in self.traffic_lights_group[i].traffic_lights:
                    traffic_light.set_flag(False)