"""
Main interface for the display.
"""
from collections import defaultdict
import pygame
import random
from engine.simulation import Simulation
from graphics.assets import AssetManager
from graphics.utils import get_clicked_movable, get_clicked_node

from graphics.init_pygame import pygame_init
from graphics.draw import create_grid_surface, draw_movable, draw_node, draw_road
from graphics.constants import SCREEN_WIDTH, SCREEN_HEIGHT

class PygameDisplay:
    simulation: Simulation
    debug_mode: bool = False
    grid_surface: pygame.Surface
    clock: pygame.time.Clock
    screen: pygame.Surface

    # Map car id to surface
    asset_manager: AssetManager = AssetManager()

    def __init__(self, simulation: Simulation, debug_mode: bool = False):
        self.simulation = simulation
        self.debug_mode = debug_mode

        pygame.display.set_caption("Simulation de réseau routier")
        if self.debug_mode:
            print("Debug mode enabled")
            print("press Space to advance 10 steps")

        self.clock = pygame.time.Clock()

    def run(self):
        print("Start simulation \n ----------------------------------")
        self.screen = pygame_init()
        self.grid_surface = create_grid_surface(self.screen)
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                elif event.type == pygame.KEYDOWN and self.debug_mode:
                    if event.key == pygame.K_SPACE:
                        for _ in range(10):
                            self.simulation.run_tick()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        pos = pygame.mouse.get_pos()
                        clicked_movable = get_clicked_movable(movables=self.simulation.movables, pos=pos)
                        if clicked_movable and self.debug_mode:
                            print("Clicked on car: ", clicked_movable)
                            break
                        clicked_node = get_clicked_node(nodes=self.simulation.nodes, pos=pos)
                        if clicked_node:
                            print("Clicked on node: ", clicked_node)

            if not self.debug_mode:
                self.simulation.run_tick()

            self.screen.fill((255, 255, 255))
            if self.debug_mode:
                self.screen.blit(self.grid_surface, (0, 0)) 
            for road in self.simulation.roads:
                draw_road(self.screen, road)
            for node in self.simulation.nodes:
                draw_node(self.screen, node)
            for spawner in self.simulation.spawners:
                for movable in spawner.movables:
                    asset = self.asset_manager.get_car_asset(movable)
                    draw_movable(movable, self.screen, asset)

            pygame.display.flip()
            self.clock.tick(30)  # FPS

        pygame.quit()