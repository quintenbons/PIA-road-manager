
from os import PathLike
from typing import Dict, List
import pygame
from graphics.constants import *
from dataclasses import dataclass, field

from engine.movable.movable import Movable

def scale_to_road_width(car_asset):
    width = int(2 * ROAD_WIDTH)
    height = car_asset.get_height()
    car_width, car_height = car_asset.get_size()
    height = int((width / car_width) * car_height)
    return pygame.transform.scale(car_asset, (width, height))

def load_resource(path: PathLike) -> pygame.Surface:
    img = pygame.image.load(path)
    return scale_to_road_width(img)

@dataclass
class AssetManager:
    cars: Dict[int, pygame.Surface] = field(default_factory=dict)

    car_assets: List[pygame.Surface] = field(default_factory=list)

    def __post_init__(self):
        self.car_assets = list(map(load_resource, [BLUE_CAR, RED_CAR, GREEN_CAR]))

    def get_car_asset(self, car: Movable) -> pygame.Surface:
        if not car._id in self.cars:
            car_asset = self.car_assets[car._id % len(self.car_assets)]
            self.cars[car._id] = car_asset
            # self.width, self.height = car_asset.get_size()

        return self.cars[car._id]
