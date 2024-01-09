
from os import PathLike
from typing import Dict, List
import pygame
from graphics.constants import *
from dataclasses import dataclass, field

from engine.movable.movable import Movable


def scale_car_asset(car_asset, node_radius):
    car_width, car_height = car_asset.get_size()

    new_width = node_radius * 0.75
    new_height = int((new_width / car_width) * car_height)

    return pygame.transform.scale(car_asset, (new_width, new_height))


def load_resource(path: PathLike) -> pygame.Surface:
    img = pygame.image.load(path)
    # return scale_to_road_width(img)
    return img


@dataclass
class AssetManager:
    def get_car_asset(self, car: Movable) -> int:
        quick_hash = ((id(car) >> 16) ^ id(car)) * 0x45d9f3b
        return quick_hash % 255
