"""
Main interface for the display.
"""
import pygame
from .draw import draw_car, draw_road

class PygameDisplay:
    def __init__(self, simulation):
        self.simulation = simulation

    def run(self):
        pygame.init()
        screen = pygame.display.set_mode((800, 600))

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            screen.fill((0, 0, 0))

            for road in self.simulation.roads:
                draw_road(screen, road.start, road.end)

            for car in self.simulation.cars:
                draw_car(screen, (car.x, car.y))

            pygame.display.flip()

