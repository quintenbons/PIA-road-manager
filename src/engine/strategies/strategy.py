from ..traffic.traffic_light import TrafficLight
from typing import List
from ..node import Node

class Strategy:
    trafficLights: List[TrafficLight]
    stateCount: int = 1
    currentState: int = 0

    def __init__(self, node:Node):
        self.trafficLights = []
        for trafficLight in node.controllers:
            if isinstance(trafficLight, TrafficLight):
                self.trafficLights.append(trafficLight)

    def next(self):
        if self.stateCount == 0:
            return
        self.currentState = (self.currentState + 1) % self.stateCount
