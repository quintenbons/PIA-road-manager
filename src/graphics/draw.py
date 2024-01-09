import pygame

from graphics.assets import scale_car_asset

from .constants import *
from engine.road import Road
from engine.node import Node
from engine.movable.movable import Movable
from graphics.constants import *
from engine.constants import TIME


def draw_car(movable: Movable, screen: pygame.Surface, car_asset: pygame.Surface, engine_x_min, engine_x_max, engine_y_min, engine_y_max, screen_width, screen_height, scale_factor):
    padding_x = screen_width *0.1
    padding_y = screen_height *0.1
    scaled_width = screen_width - 2 * padding_x
    scaled_height = screen_height - 2 * padding_y

    node_radius = NODE_RADIUS * scale_factor
    scaled_car_asset = scale_car_asset(car_asset, node_radius)

    x, y = movable.cmovable.to_coord_xy()
    normalized_x = (x - engine_x_min) / (engine_x_max - engine_x_min)
    normalized_y = (y - engine_y_min) / (engine_y_max - engine_y_min)
    centered_x = normalized_x * scaled_width + padding_x
    centered_y = normalized_y * scaled_height + padding_y

    car_width, car_height = scaled_car_asset.get_size()

    car_x = int(centered_x - car_width / 2)
    car_y = int(centered_y - car_height / 2)

    screen.blit(scaled_car_asset, (car_x, car_y))


def draw_road(screen, road: Road, engine_x_min, engine_x_max, engine_y_min, engine_y_max, screen_width, screen_height, scale_factor):
    padding_x = screen_width *0.1
    padding_y = screen_height *0.1
    scaled_width = screen_width - 2 * padding_x
    scaled_height = screen_height - 2 * padding_y

    start_x, start_y = road.croad.get_pos_start()
    end_x, end_y = road.croad.get_pos_end()

    normalized_start_x = (start_x - engine_x_min) / \
        (engine_x_max - engine_x_min)
    normalized_start_y = (start_y - engine_y_min) / \
        (engine_y_max - engine_y_min)
    normalized_end_x = (end_x - engine_x_min) / (engine_x_max - engine_x_min)
    normalized_end_y = (end_y - engine_y_min) / (engine_y_max - engine_y_min)

    display_start_x = normalized_start_x * scaled_width + padding_x
    display_start_y = normalized_start_y * scaled_height + padding_y
    display_end_x = normalized_end_x * scaled_width + padding_x
    display_end_y = normalized_end_y * scaled_height + padding_y

    road_width = int(ROAD_WIDTH * scale_factor)

    pygame.draw.line(screen, ROAD_COLOR, (int(display_start_x), int(
        display_start_y)), (int(display_end_x), int(display_end_y)), road_width)


def draw_node(screen, node: Node, engine_x_min, engine_x_max, engine_y_min, engine_y_max, screen_width, screen_height, scale_factor):
    padding_x = screen_width *0.1
    padding_y = screen_height *0.1
    scaled_width = screen_width - 2 * padding_x
    scaled_height = screen_height - 2 * padding_y

    normalized_x = (node.cnode.get_x() - engine_x_min) / \
        (engine_x_max - engine_x_min)
    normalized_y = (node.cnode.get_y() - engine_y_min) / \
        (engine_y_max - engine_y_min)
    display_x = normalized_x * scaled_width + padding_x
    display_y = normalized_y * scaled_height + padding_y

    node_radius = NODE_RADIUS * scale_factor

    pygame.draw.circle(screen, NODE_COLOR, (int(
        display_x), int(display_y)), node_radius)


def draw_movable(movable: Movable, screen, color: int, engine_x_min, engine_x_max, engine_y_min, engine_y_max, screen_width, screen_height, scale_factor):
    draw_car(movable, screen, color, engine_x_min, engine_x_max,
             engine_y_min, engine_y_max, screen_width, screen_height, scale_factor)


def draw_traffic_light(screen, road: Road, engine_x_min, engine_x_max, engine_y_min, engine_y_max, screen_width, screen_height, scale_factor):
    padding_x = screen_width *0.1
    padding_y = screen_height *0.1
    scaled_width = screen_width - 2 * padding_x
    scaled_height = screen_height - 2 * padding_y

    start_x, start_y = road.get_pos_start()

    normalized_x = (start_x - engine_x_min) / (engine_x_max - engine_x_min)
    normalized_y = (start_y - engine_y_min) / (engine_y_max - engine_y_min)
    display_x = normalized_x * scaled_width + padding_x
    display_y = normalized_y * scaled_height + padding_y

    traffic_light_width = 3.33 * scale_factor
    traffic_light_height = 6.66 * scale_factor
    traffic_light_x = int(display_x - traffic_light_width / 2)
    traffic_light_y = int(display_y - traffic_light_height / 2)

    pygame.draw.rect(screen, (80, 80, 80), (traffic_light_x,
                     traffic_light_y, traffic_light_width, traffic_light_height))

    is_green = not road.get_block_traffic()

    red_light_color = (255, 0, 0) if not is_green else (50, 0, 0)
    green_light_color = (0, 255, 0) if is_green else (0, 50, 0)
    circle_radius = 1.5 * scale_factor
    red_light_y = traffic_light_y + traffic_light_height / 4
    green_light_y = traffic_light_y + 3 * traffic_light_height / 4

    pygame.draw.circle(screen, red_light_color, (traffic_light_x +
                       traffic_light_width // 2, red_light_y), circle_radius)
    pygame.draw.circle(screen, green_light_color, (traffic_light_x +
                       traffic_light_width // 2, green_light_y), circle_radius)


def draw_speed_sign(screen, road, engine_x_min, engine_x_max, engine_y_min, engine_y_max, screen_width, screen_height, scale_factor):
    speed_limit = int(road.get_speed_limit() * 10)

    start_pos = road.get_pos_start()
    end_pos = road.get_pos_end()
    road_id_hash = ((id(road) >> 16) ^ id(road)) * 0x45d9f3b
    ratio = 0.25 + (road_id_hash % 500) / 1000.0

    pos = (start_pos[0] + ratio * (end_pos[0] - start_pos[0]),
           start_pos[1] + ratio * (end_pos[1] - start_pos[1]))

    normalized_x = (pos[0] - engine_x_min) / \
        (engine_x_max - engine_x_min)
    normalized_y = (pos[1] - engine_y_min) / \
        (engine_y_max - engine_y_min)
    display_x = normalized_x * \
        (screen_width - screen_width * 0.1) + screen_width *0.1
    display_y = normalized_y * \
        (screen_height - screen_height * 0.1) + screen_height *0.1 - 20

    sign_radius = 5 * scale_factor
    post_width = 1 * scale_factor
    post_height = 10 * scale_factor

    pygame.draw.rect(screen, NODE_COLOR, (display_x -
                     post_width / 2, display_y, post_width, post_height))

    pygame.draw.circle(screen, (255, 0, 0), (int(
        display_x), int(display_y)), sign_radius)
    pygame.draw.circle(screen, (255, 255, 255), (int(
        display_x), int(display_y)), sign_radius - 1 * scale_factor)

    font = pygame.font.SysFont(None, int(7 * scale_factor))
    speed_text = font.render(f"{speed_limit}", True, (0, 0, 0))
    screen.blit(speed_text, (display_x - speed_text.get_width() /
                2, display_y - speed_text.get_height() / 2))


def get_rect(obj):
    if isinstance(obj, Node):
        return pygame.Rect(obj.cnode.get_x() - NODE_RADIUS, obj.cnode.get_y() - NODE_RADIUS, NODE_RADIUS * 2, NODE_RADIUS * 2)
    elif isinstance(obj, Movable):
        x, y = obj.cmovable.to_coord_xy()
        centered_x = x - NODE_RADIUS
        centered_y = y - NODE_RADIUS
        return pygame.Rect(centered_x, centered_y, NODE_RADIUS * 2, NODE_RADIUS * 2)


def create_grid_surface(screen):
    grid_size = 50
    window_size = screen.get_size()
    grid_surface = pygame.Surface(window_size, pygame.SRCALPHA)
    grid_surface.fill((0, 0, 0, 0))

    for x in range(0, window_size[0], grid_size):
        pygame.draw.line(grid_surface, (200, 200, 200),
                         (x, 0), (x, window_size[1]))
    for y in range(0, window_size[1], grid_size):
        pygame.draw.line(grid_surface, (200, 200, 200),
                         (0, y), (window_size[0], y))

    for x in range(0, window_size[0], grid_size):
        for y in range(0, window_size[1], grid_size):
            font = pygame.font.Font(None, 14)
            coord_text = f"{x},{y}"
            text = font.render(coord_text, True, (200, 200, 200))
            grid_surface.blit(text, (x + 5, y + 5))

    return grid_surface


def draw_hud(display):
    elapsed_time_in_seconds = display.simulation.current_tick * TIME
    time_surface = pygame.font.SysFont('Arial', 20).render(
        f'Time: {elapsed_time_in_seconds:.2f} s', True, (0, 0, 0))
    display.screen.blit(time_surface, (10, 10))

    speed_surface = pygame.font.SysFont('Arial', 20).render(
        f'Speed: x{display.speed_factor}', True, (0, 0, 0))
    display.screen.blit(speed_surface, (10, 35))

    hug_help_surface = pygame.font.SysFont('Arial', 20).render(
        f'P to pause, + to speed up, - to slow down', True, (0, 0, 0))
    display.screen.blit(hug_help_surface, (10, 60))


def draw_scale(screen, pixel_length, real_length_m):
    scale_pos = (50, SCREEN_HEIGHT - 50)  # Position de l'échelle sur l'écran
    pygame.draw.line(screen, (0, 0, 0), scale_pos,
                     (scale_pos[0] + pixel_length, scale_pos[1]), 5)
    font = pygame.font.SysFont(None, 24)
    text = font.render(f"{pixel_length}px = {real_length_m}m", True, (0, 0, 0))
    screen.blit(text, (scale_pos[0], scale_pos[1] - 30))


def draw_paused_text(display):
    if display.paused and pygame.time.get_ticks() // 1000 % 2:
        paused_text = pygame.font.SysFont('Arial', 50).render(
            'Appuyez sur P pour reprendre', True, (255, 0, 0))
        text_rect = paused_text.get_rect(
            center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2))
        display.screen.blit(paused_text, text_rect)
