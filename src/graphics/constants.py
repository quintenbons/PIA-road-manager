import pygame

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

ROAD_WIDTH = 5
NODE_RADIUS = 6

# Colors
BACKGROUND_COLOR = (0, 0, 0)    # Black
ROAD_COLOR = (150, 150, 150)    # Gray
CLOSED_ROAD_COLOR = (50, 50, 50)    # Dark gray
CAR_COLOR = (0, 0, 200) # Blue
NODE_COLOR = (100, 150, 100)    # Green

BLUE_CAR = None
RED_CAR = None
GREEN_CAR = None

def load_ressources():
    global BLUE_CAR, RED_CAR, GREEN_CAR
    BLUE_CAR = pygame.image.load("src/assets/blue_car.png")
    BLUE_CAR = pygame.transform.scale(BLUE_CAR, (24, 15))
    RED_CAR = pygame.image.load("src/assets/red_car.png")
    RED_CAR = pygame.transform.scale(RED_CAR, (82, 52))
    GREEN_CAR = pygame.image.load("src/assets/green_car.png")
    GREEN_CAR = pygame.transform.scale(GREEN_CAR, (82, 52))