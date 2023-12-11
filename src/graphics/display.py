"""
Main interface for the display.
"""
import pygame
from .draw import draw_car, draw_road
from graphics.constants import SCREEN_WIDTH, SCREEN_HEIGHT

class PygameDisplay:
    def __init__(self, simulation):
        self.simulation = simulation

    def run(self):
        pygame.init()
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.simulation.next_tick()

            screen.fill((0, 0, 0))

            for road in self.simulation.roads:
                draw_road(screen, road)

            for car in self.simulation.cars:
                draw_car(screen, car)

            font = pygame.font.SysFont('Arial', 20)
            text = font.render('Press space to advance', True, (255, 255, 255))
            screen.blit(text, (300, 550))

            pygame.display.flip()

