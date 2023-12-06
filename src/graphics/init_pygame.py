import pygame
from .constants import load_ressources
import time

def pygame_init():
    pygame.init()
    screen = pygame.display.set_mode((1200, 800))
    load_ressources()
    time.sleep(0.5)
    return screen