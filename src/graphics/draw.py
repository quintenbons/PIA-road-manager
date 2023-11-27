import pygame
from .constants import *
from engine.road import Road
from engine.movable.car import Car
from engine.node import Node
from engine.movable.movable import Movable

def draw_car(screen, car: Car):
    car_pos = car.get_coord()
    car_rect = pygame.Rect(car_pos[0], car_pos[1], CAR_WIDTH, CAR_LENGTH)
    pygame.draw.rect(screen, CAR_COLOR, car_rect)

def draw_road(screen, road: Road):
    pygame.draw.line(screen, ROAD_COLOR, road.pos_start, road.pos_end, ROAD_WIDTH)

def draw_node(screen, node:Node):
    pygame.draw.circle(screen, NODE_COLOR, node.position, NODE_RADIUS)

def draw_movable(movable: Movable, screen):
    x, y = movable.to_coord_xy()
    pygame.draw.circle(screen, (255, 0, 0), (int(x), int(y)), 5)

def draw_grid(screen, grid_size, window_size):
    for x in range(0, window_size[0], grid_size):
        pygame.draw.line(screen, (200, 200, 200), (x, 0), (x, window_size[1]))
    for y in range(0, window_size[1], grid_size):
        pygame.draw.line(screen, (200, 200, 200), (0, y), (window_size[0], y))
    for x in range(0, window_size[0], grid_size):
        for y in range(0, window_size[1], grid_size):
            coord_text = f"{x},{y}"
            font = pygame.font.Font(None, 14)
            text = font.render(coord_text, True, (200, 200, 200))
            screen.blit(text, (x + 5, y + 5))
