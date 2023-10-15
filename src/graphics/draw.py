import pygame
from .constants import *

def draw_road(screen, p1, p2):
    pygame.draw.line(screen, ROAD_COLOR, p1, p2, ROAD_WIDTH)

def draw_car(screen, p1):
    car_rect = pygame.Rect(p1[0], p1[1], CAR_WIDTH, CAR_LENGTH)
    pygame.draw.rect(screen, CAR_COLOR, car_rect)
