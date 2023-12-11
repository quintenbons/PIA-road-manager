import pygame
from .constants import *
from engine.road import Road
from engine.movable.car import Car
from engine.node import Node
from engine.movable.movable import Movable

def draw_car(movable: Movable, screen):
    x, y = movable.to_coord_xy()
    centered_x = x
    centered_y = y
    screen.blit(movable.car_asset, (centered_x, centered_y))

def draw_road(screen, road: Road):
    if road.block_traffic:
        pygame.draw.line(screen, CLOSED_ROAD_COLOR, road.pos_start, road.pos_end, ROAD_WIDTH)
    else:
        pygame.draw.line(screen, ROAD_COLOR, road.pos_start, road.pos_end, ROAD_WIDTH)

def draw_node(screen, node:Node):
    pygame.draw.circle(screen, NODE_COLOR, node.position, NODE_RADIUS)

def draw_movable(movable: Movable, screen):
    draw_car(movable, screen)

def get_rect(obj):
    if isinstance(obj, Node):
        return pygame.Rect(obj.position[0] - NODE_RADIUS, obj.position[1] - NODE_RADIUS, NODE_RADIUS * 2, NODE_RADIUS * 2)
    elif isinstance(obj, Movable):
        x, y = obj.to_coord_xy()
        centered_x = x
        centered_y = y
        return pygame.Rect(centered_x, centered_y, obj.width, obj.height)

def create_grid_surface(screen):
    grid_size = 50
    window_size = screen.get_size()
    grid_surface = pygame.Surface(window_size, pygame.SRCALPHA)  # Créez une surface transparente
    grid_surface.fill((0, 0, 0, 0))  # Remplissez la surface avec une couleur transparente

    # Dessinez les lignes de la grille sur la surface de la grille
    for x in range(0, window_size[0], grid_size):
        pygame.draw.line(grid_surface, (200, 200, 200), (x, 0), (x, window_size[1]))
    for y in range(0, window_size[1], grid_size):
        pygame.draw.line(grid_surface, (200, 200, 200), (0, y), (window_size[0], y))

    # Dessinez les coordonnées à toutes les intersections
    for x in range(0, window_size[0], grid_size):
        for y in range(0, window_size[1], grid_size):
            font = pygame.font.Font(None, 14)
            coord_text = f"{x},{y}"
            text = font.render(coord_text, True, (200, 200, 200))
            grid_surface.blit(text, (x + 5, y + 5))

    return grid_surface
