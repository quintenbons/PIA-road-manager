from engine.constants import TIME
from ..traffic.traffic_light import TrafficLight
from typing import List
from ..node import Node

class Strategy:
    trafficLights: List[TrafficLight]
    stateCount: int = 1
    currentState: int = 0
    time_per_state: int = 50 / TIME
    last_time: int = 0
    is_initialized: bool = False

    def __init__(self, node:Node):
        self.trafficLights = []
        for trafficLight in node.controllers:
            if isinstance(trafficLight, TrafficLight):
                self.trafficLights.append(trafficLight)

    # This function is private, it is used to set the state of the traffic lights
    def next(self):
        if self.stateCount == 0:
            return
        self.currentState = (self.currentState + 1) % self.stateCount

    # This function is called every tick, it is used to update the state of the traffic lights
    def update(self, time):
        if not self.is_initialized:
            self.is_initialized = True
            self.next()
            return
        if self.last_time == 0:
            self.last_time = time
        if time - self.last_time >= self.time_per_state:
            self.next()
            self.last_time = time
        