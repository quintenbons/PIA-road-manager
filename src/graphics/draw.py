import pygame

from .constants import *
from engine.road import Road
from engine.node import Node
from engine.movable.movable import Movable
from graphics.assets import NODE_RADIUS
from engine.constants import TIME

def draw_car(movable: Movable, screen: pygame.Surface, color: int):
    x, y = movable.to_coord_xy()
    centered_x = x
    centered_y = y
    pygame.draw.circle(screen, ((color * 26)%255, (color * 12)%255, (color*3)%255), (centered_x, centered_y), 4)
    # draw also the rect
    pygame.draw.rect(screen, (255, 0, 0), get_rect(movable), 1)

def draw_road(screen, road: Road):
    if road.block_traffic:
        pygame.draw.line(screen, CLOSED_ROAD_COLOR, road.pos_start, road.pos_end, ROAD_WIDTH)
    else:
        pygame.draw.line(screen, ROAD_COLOR, road.pos_start, road.pos_end, ROAD_WIDTH)

def draw_node(screen, node:Node):
    pygame.draw.circle(screen, NODE_COLOR, node.position, NODE_RADIUS)

def draw_movable(movable: Movable, screen, color: int):
    draw_car(movable, screen, color)

def get_rect(obj):
    if isinstance(obj, Node):
        return pygame.Rect(obj.position[0] - NODE_RADIUS, obj.position[1] - NODE_RADIUS, NODE_RADIUS * 2, NODE_RADIUS * 2)
    elif isinstance(obj, Movable):
        x, y = obj.to_coord_xy()
        centered_x = x  - NODE_RADIUS
        centered_y = y  - NODE_RADIUS
        return pygame.Rect(centered_x, centered_y, NODE_RADIUS * 2, NODE_RADIUS * 2)

def create_grid_surface(screen):
    grid_size = 50
    window_size = screen.get_size()
    grid_surface = pygame.Surface(window_size, pygame.SRCALPHA)
    grid_surface.fill((0, 0, 0, 0))

    for x in range(0, window_size[0], grid_size):
        pygame.draw.line(grid_surface, (200, 200, 200), (x, 0), (x, window_size[1]))
    for y in range(0, window_size[1], grid_size):
        pygame.draw.line(grid_surface, (200, 200, 200), (0, y), (window_size[0], y))

    for x in range(0, window_size[0], grid_size):
        for y in range(0, window_size[1], grid_size):
            font = pygame.font.Font(None, 14)
            coord_text = f"{x},{y}"
            text = font.render(coord_text, True, (200, 200, 200))
            grid_surface.blit(text, (x + 5, y + 5))

    return grid_surface


def draw_hud(display):
    elapsed_time_in_seconds = display.simulation.current_tick * TIME
    time_surface = pygame.font.SysFont('Arial', 20).render(f'Time: {elapsed_time_in_seconds:.2f} s', True, (0, 0, 0))
    display.screen.blit(time_surface, (10, 10))

    speed_surface = pygame.font.SysFont('Arial', 20).render(f'Speed: x{display.speed_factor}', True, (0, 0, 0))
    display.screen.blit(speed_surface, (10, 35))

    hug_help_surface = pygame.font.SysFont('Arial', 20).render(f'P to pause, + to speed up, - to slow down', True, (0, 0, 0))
    display.screen.blit(hug_help_surface, (10, 60))

def draw_paused_text(display):
        if display.paused and pygame.time.get_ticks() // 1000 % 2:
            paused_text = pygame.font.SysFont('Arial', 50).render('Appuyez sur P pour reprendre', True, (255, 0, 0))
            text_rect = paused_text.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2))
            display.screen.blit(paused_text, text_rect)
