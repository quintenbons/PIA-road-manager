from dataclasses import dataclass
from .flow_controller import FlowController
from ..constants import CROSSWALK_SPEED

@dataclass
class Crosswalk(FlowController):
    hicker_buffer: int = 0
    is_open_to_vehicles: bool = True

    # Logic when handled by traffic light
    def update(self, time, greenForVehicles):
        if greenForVehicles:
            self.is_open_to_vehicles = True
        else:
            self.update(time)

    def update(self, time) -> None:
        if len(self.hikerQueue) > 0:
            self.hickerQueue = []
            self.hicker_buffer = time

        if self.hicker_buffer + CROSSWALK_SPEED <= time:
            self.is_open_to_vehicles = True

    def is_open(self) -> bool:
        return self.is_open_to_vehicles
        




