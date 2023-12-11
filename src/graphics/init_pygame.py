import pygame
from .constants import SCREEN_WIDTH, SCREEN_HEIGHT

def pygame_init():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    return screen