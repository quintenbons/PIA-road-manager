from engine.constants import DEFAULT_DURATION
from engine.traffic.flow_controller import FlowController
from ...traffic.traffic_light import TrafficLight
from typing import List

class Strategy:
    trafficLights: List[TrafficLight]
    stateCount: int = 1
    currentState: int = 0
    time_per_state: int = DEFAULT_DURATION
    last_time: int = 0
    is_initialized: bool = False
    state_cycles: List[int] = []

    def __init__(self, controllers:List[FlowController], cycles:List[int]):
        self.trafficLights = []
        for trafficLight in controllers:
            if isinstance(trafficLight, TrafficLight):
                self.trafficLights.append(trafficLight)

        # Cycle represents the time to wait before switching to the next state
        self.state_cycles = [cycle for cycle in cycles if cycle is not None]

    # This function is private, it is used to set the state of the traffic lights
    def next(self):
        self.currentState = (self.currentState + 1) % self.stateCount

    # This function is called every tick, it is used to update the state of the traffic lights
    def update(self, time):
        if self.stateCount == 1 or self.stateCount == 0: 
            return
        if not self.is_initialized:
            self.is_initialized = True
            self.next()
            return
        if self.last_time == 0:
            self.last_time = time
        if time - self.last_time >= self.state_cycles[self.currentState]:
            self.next()
            self.last_time = time
        