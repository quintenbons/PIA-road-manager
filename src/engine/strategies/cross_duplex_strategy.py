from ..traffic.traffic_light import TrafficLight
from ..node import Node
from typing import List
from strategy import Strategy
import math

class TrafficGroup:
    traffic_lights: set[TrafficLight]
    pos: tuple(float, float)

    def __init__(self):
        self.traffic_lights = set()
        self.pos = None

    def add_traffic_light(self, traffic_light: TrafficLight):
        self.traffic_lights.add(traffic_light)
        if self.pos is None:
            self.pos = traffic_light.pos
        else:
            self.pos = (0, 0)
            for traffic_light in self.traffic_lights:
                self.pos[0] += traffic_light.pos[0]
                self.pos[1] += traffic_light.pos[1]
            self.pos[0] /= len(self.traffic_lights)
            self.pos[1] /= len(self.traffic_lights)


class CrossDuplexStrategy(Strategy):
    traffic_lights_group: List[TrafficGroup]
    node_pos: tuple(float, float)
    opposit_degree_treshold :int= 15

    # 
    def __init__(self, node: Node):
        super().__init__(node)

        self.traffic_lights_group = []
        self.node_pos = node.position

        for traffic_light in self.trafficLights:
            self.associate_to_group(traffic_light)

    def associate_to_group(self, traffic_light: TrafficLight):
        for group in self.traffic_lights_group:
            if self.is_opposit(self.node_pos, traffic_light.road_in.pos_end, group.pos):
                group.add_traffic_light(traffic_light)
                return
        group = TrafficGroup()
        group.add_traffic_light(traffic_light)

    def is_opposit(self, center:tuple(float, float), a:tuple(float, float), b:tuple(float, float)) -> bool:
        print(a, b)
        print(get_angle(center, a), get_angle(center, b))
        return abs(get_angle(center, a) - get_angle(center, b)) > self.opposit_degree_treshold
    
    def get_angle(self, center:tuple(float, float), point:tuple(float, float)) -> float:
        x, y = point[0] - center[0], point[1] - center[1]
        angle_rad = math.atan2(y, x)
        angle_deg = math.degrees(angle_rad)
        return angle_deg
    
    def getAngle(center:tuple(float, float), point:tuple(float, float)) -> float:
        x = point[0] - center[0]
        y = point[1] - center[1]
        if x == 0:
            if y > 0:
                return 90
            else:
                return 270
        elif y == 0:
            if x > 0:
                return 0
            else:
                return 180
        else:
            angle = math.atan(y/x)
            angle = math.degrees(angle)
            if x > 0 and y > 0:
                return angle
            elif x < 0 and y > 0:
                return 180 + angle
            elif x < 0 and y < 0:
                return 180 + angle
            else:
                return 360 + angle

        
    def set_corridor(self, index: int):
        if index < len(self.trafficLights):
            self.corridor = self.trafficLights[index]
            self.corridor.set_flag(True, 0)
            for i in range(len(self.trafficLights)):
                if i != index:
                    self.otherTrafficLights.append(self.trafficLights[i])
            if len(self.otherTrafficLights) > 0:
                self.otherTrafficLights[0].set_flag(True, 0)
            self.stateCount = min(len(self.otherTrafficLights), 1)
        else:
            raise IndexError("Index out of range")


    def next(self):
        if self.otherTrafficLights is None:
            raise Exception("Corridor not set")
        super().next()
        for trafficLight in self.otherTrafficLights:
            trafficLight.set_flag(False, 0)
        if len(self.otherTrafficLights) > 0:
            self.otherTrafficLights[self.currentState].set_flag(True, 0)     
            