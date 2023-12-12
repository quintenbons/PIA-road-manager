from typing import List
from engine.traffic.traffic_light import TrafficLight
from .strategy import Strategy

class OpenStrategy(Strategy):
    def __init__(self, controllers:List[TrafficLight], cycles:List[int]):
        super().__init__(controllers, cycles)
        self.stateCount = 1
        for trafficLight in self.trafficLights:
            trafficLight.set_flag(True)

    def next(self):
        # No need to call super().next() because there is only one state
        pass
            