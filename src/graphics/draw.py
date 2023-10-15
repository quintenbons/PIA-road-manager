import pygame
from .constants import *
from engine.road import Road
from engine.car import Car

def draw_road(screen, road: Road):
    pygame.draw.line(screen, ROAD_COLOR, road.start, road.end, ROAD_WIDTH)

def draw_car(screen, car: Car):
    car_pos = car.get_coord()
    car_rect = pygame.Rect(car_pos[0], car_pos[1], CAR_WIDTH, CAR_LENGTH)
    pygame.draw.rect(screen, CAR_COLOR, car_rect)
