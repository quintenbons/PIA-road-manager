import pygame
from .constants import load_ressources, SCREEN_WIDTH, SCREEN_HEIGHT
import time

def pygame_init():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    load_ressources()
    time.sleep(0.5)
    return screen