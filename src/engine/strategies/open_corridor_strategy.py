from ..traffic.traffic_light import TrafficLight
from ..node import Node
from typing import List
from .strategy import Strategy

class OpenCorridorStrategy(Strategy):
    corridor: TrafficLight = None
    otherTrafficLights: List[TrafficLight] = []

    def __init__(self, node: Node):
        super().__init__(node)

    def set_corridor(self, index: int):
        if index < len(self.trafficLights):
            self.corridor = self.trafficLights[index]
            self.corridor.set_flag(True)
            for i in range(len(self.trafficLights)):
                if i != index:
                    self.otherTrafficLights.append(self.trafficLights[i])
            if len(self.otherTrafficLights) > 0:
                self.otherTrafficLights[0].set_flag(True)
            self.stateCount = max(len(self.otherTrafficLights), 1)
        else:
            raise IndexError("Index out of range")


    def next(self):
        if self.otherTrafficLights is None:
            raise Exception("Corridor not set")
        super().next()
        for trafficLight in self.otherTrafficLights:
            trafficLight.set_flag(False)
        if len(self.otherTrafficLights) > 0:
            self.otherTrafficLights[self.currentState].set_flag(True)     
            