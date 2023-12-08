from ..node import Node
from .strategy import Strategy

class PieceOfCakeStrategy(Strategy):
    def __init__(self, node: Node):
        super().__init__(node)
        self.stateCount = max(len(self.trafficLights), 1)
        if len(self.trafficLights) > 0:
            self.trafficLights[0].set_flag(True)
        

    def next(self):
        super().next()

        for trafficLight in self.trafficLights:
            trafficLight.set_flag(False)
        if len(self.trafficLights) > 0:
            self.trafficLights[self.currentState].set_flag(True)