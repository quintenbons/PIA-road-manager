import pygame

SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800

ROAD_WIDTH = 15
NODE_RADIUS = 11

# Colors
BACKGROUND_COLOR = (0, 0, 0)    # Black
ROAD_COLOR = (150, 150, 150)    # Gray
CLOSED_ROAD_COLOR = (50, 50, 50)    # Dark gray
CAR_COLOR = (0, 0, 200) # Blue
NODE_COLOR = (100, 150, 100)    # Green

BLUE_CAR = None
RED_CAR = None
GREEN_CAR = None


def scale_to_road_width(car_asset):
    width = int(2 * ROAD_WIDTH)
    height = car_asset.get_height()
    car_width, car_height = car_asset.get_size()
    height = int((width / car_width) * car_height)
    return pygame.transform.scale(car_asset, (width, height))

def load_ressources():
    global BLUE_CAR, RED_CAR, GREEN_CAR
    BLUE_CAR = pygame.image.load("src/assets/blue_car.png")
    BLUE_CAR = scale_to_road_width(BLUE_CAR)
    RED_CAR = pygame.image.load("src/assets/red_car.png")
    RED_CAR =scale_to_road_width(RED_CAR)
    GREEN_CAR = pygame.image.load("src/assets/green_car.png")
    GREEN_CAR = scale_to_road_width(GREEN_CAR)