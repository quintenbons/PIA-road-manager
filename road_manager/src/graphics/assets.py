
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
    def get_car_asset(self, car: Movable) -> int:
        quick_hash = ((car._id >> 16) ^ car._id) * 0x45d9f3b
        return quick_hash % 255
