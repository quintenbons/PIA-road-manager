import pygame

from .constants import *
from engine.road import Road
from engine.node import Node
from engine.movable.movable import Movable
# from graphics.assets import NODE_RADIUS
from engine.constants import TIME, ROAD_OFFSET, NODE_RADIUS

def draw_car(movable: Movable, screen: pygame.Surface, color: int, engine_x_min, engine_x_max, engine_y_min, engine_y_max, screen_width, screen_height):
    padding_x = screen_width * 0.05
    padding_y = screen_height * 0.05
    scaled_width = screen_width - 2 * padding_x
    scaled_height = screen_height - 2 * padding_y

    x, y = movable.cmovable.to_coord_xy()

    normalized_x = (x - engine_x_min) / (engine_x_max - engine_x_min)
    normalized_y = (y - engine_y_min) / (engine_y_max - engine_y_min)
    centered_x = normalized_x * scaled_width + padding_x
    centered_y = normalized_y * scaled_height + padding_y

    pygame.draw.circle(screen, ((color * 26)%255, (color * 12)%255, (color*3)%255), (int(centered_x), int(centered_y)), 4)


def draw_road(screen, road: Road, engine_x_min, engine_x_max, engine_y_min, engine_y_max, screen_width, screen_height):
    padding_x = screen_width * 0.05
    padding_y = screen_height * 0.05
    scaled_width = screen_width - 2 * padding_x
    scaled_height = screen_height - 2 * padding_y

    start_x, start_y = road.croad.get_pos_start()
    end_x, end_y = road.croad.get_pos_end()

    normalized_start_x = (start_x - engine_x_min) / (engine_x_max - engine_x_min)
    normalized_start_y = (start_y - engine_y_min) / (engine_y_max - engine_y_min)
    normalized_end_x = (end_x - engine_x_min) / (engine_x_max - engine_x_min)
    normalized_end_y = (end_y - engine_y_min) / (engine_y_max - engine_y_min)

    display_start_x = normalized_start_x * scaled_width + padding_x
    display_start_y = normalized_start_y * scaled_height + padding_y
    display_end_x = normalized_end_x * scaled_width + padding_x
    display_end_y = normalized_end_y * scaled_height + padding_y

    if road.croad.get_block_traffic():
        pygame.draw.line(screen, CLOSED_ROAD_COLOR, (int(display_start_x), int(display_start_y)), (int(display_end_x), int(display_end_y)), ROAD_WIDTH)
    else:
        pygame.draw.line(screen, ROAD_COLOR, (int(display_start_x), int(display_start_y)), (int(display_end_x), int(display_end_y)), ROAD_WIDTH)


def draw_node(screen, node: Node, engine_x_min, engine_x_max, engine_y_min, engine_y_max, screen_width, screen_height):
    padding_x = screen_width * 0.05
    padding_y = screen_height * 0.05
    scaled_width = screen_width - 2 * padding_x
    scaled_height = screen_height - 2 * padding_y

    normalized_x = (node.cnode.get_x() - engine_x_min) / (engine_x_max - engine_x_min)
    normalized_y = (node.cnode.get_y() - engine_y_min) / (engine_y_max - engine_y_min)
    display_x = normalized_x * scaled_width + padding_x
    display_y = normalized_y * scaled_height + padding_y

    pygame.draw.circle(screen, NODE_COLOR, (int(display_x), int(display_y)), NODE_RADIUS)

def draw_movable(movable: Movable, screen, color: int, engine_x_min, engine_x_max, engine_y_min, engine_y_max, screen_width, screen_height):
    draw_car(movable, screen, color, engine_x_min, engine_x_max, engine_y_min, engine_y_max, screen_width, screen_height)

def get_rect(obj):
    if isinstance(obj, Node):
        return pygame.Rect(obj.cnode.get_x() - NODE_RADIUS, obj.cnode.get_y() - NODE_RADIUS, NODE_RADIUS * 2, NODE_RADIUS * 2)
    elif isinstance(obj, Movable):
        x, y = obj.cmovable.to_coord_xy()
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

def draw_scale(screen, pixel_length, real_length_m):
    scale_pos = (50, SCREEN_HEIGHT - 50)  # Position de l'échelle sur l'écran
    pygame.draw.line(screen, (0,0,0), scale_pos, (scale_pos[0] + pixel_length, scale_pos[1]), 5)
    font = pygame.font.SysFont(None, 24)
    text = font.render(f"{pixel_length}px = {real_length_m}m", True, (0,0,0))
    screen.blit(text, (scale_pos[0], scale_pos[1] - 30))
    
def draw_paused_text(display):
        if display.paused and pygame.time.get_ticks() // 1000 % 2:
            paused_text = pygame.font.SysFont('Arial', 50).render('Appuyez sur P pour reprendre', True, (255, 0, 0))
            text_rect = paused_text.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2))
            display.screen.blit(paused_text, text_rect)
