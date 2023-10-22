from enum import Enum

class Movable:

    speed: float = 0.0
    latency: float = 0.0
    pos: float = 0.0
    size: float
    category: Enum

    def update(self) -> None:
        #TODO what happen
        self.pos += self.speed

    def handleTrafficLight(self):
        pass
    def handleStop(self):
        pass
    def handleCrosswalk(self):
        pass

class Category(Enum):
    CAR = 0
    BIKE = 1
    HUMAN = 2
