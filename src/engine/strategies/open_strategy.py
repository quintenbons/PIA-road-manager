from ..node import Node
from strategy import Strategy

class OpenStrategy(Strategy):
    def __init__(self, node: Node):
        super().__init__(node)
        self.stateCount = 1
        for trafficLight in self.trafficLights:
            trafficLight.set_flag(True, 0)

    def next(self):
        # No need to call super().next() because there is only one state
        pass
            