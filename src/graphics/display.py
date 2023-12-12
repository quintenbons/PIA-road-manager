"""
Main interface for the display.
"""
from collections import defaultdict
import pygame
import random
from engine.constants import TIME
from engine.simulation import Simulation
from graphics.assets import AssetManager
from graphics.utils import get_clicked_movable, get_clicked_node

from graphics.init_pygame import pygame_init
from graphics.draw import create_grid_surface, draw_movable, draw_node, draw_road
from graphics.constants import SCREEN_WIDTH, SCREEN_HEIGHT

class PygameDisplay:
    simulation: Simulation
    paused: bool = True
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
    
    def show_paused_text(self):
        if self.paused and pygame.time.get_ticks() // 1000 % 2:
            paused_text = pygame.font.SysFont('Arial', 50).render('Appuyez sur P pour reprendre', True, (255, 0, 0))
            text_rect = paused_text.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2))
            self.screen.blit(paused_text, text_rect)

    def run(self):
        print("Start simulation \n ----------------------------------")
        self.screen = pygame_init()
        self.grid_surface = create_grid_surface(self.screen)
        running = True

        fps = 1 / TIME # FPS = tickers per second 
        self.speed_factor = 1

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                elif event.type == pygame.KEYDOWN:
                    if self.debug_mode and event.key == pygame.K_SPACE:
                        for _ in range(10):
                            self.simulation.run_tick()
                    if event.key == pygame.K_p:
                        self.paused = not self.paused
                    elif event.key == pygame.K_PLUS or event.key == pygame.K_KP_PLUS:
                        self.speed_factor *= 2
                    elif event.key == pygame.K_MINUS or event.key == pygame.K_KP_MINUS:
                        self.speed_factor = max(1, self.speed_factor // 2)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        pos = pygame.mouse.get_pos()
                        for spawner in self.simulation.spawners:
                            clicked_movable = get_clicked_movable(movables=spawner.movables, pos=pos)
                        if clicked_movable:
                            print("Clicked on car: ", clicked_movable)
                            break
                        clicked_node = get_clicked_node(nodes=self.simulation.nodes, pos=pos)
                        if clicked_node:
                            print("Clicked on node: ", clicked_node)

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

            if not self.debug_mode:
                if not self.paused:
                    for _ in range(self.speed_factor):
                        self.simulation.run_tick()
                else:
                    self.show_paused_text()
            
            
            elapsed_time_in_seconds = self.simulation.current_tick * TIME
            time_surface = pygame.font.SysFont('Arial', 30).render(f'Temps écoulé: {elapsed_time_in_seconds:.2f} s', True, (0, 0, 0))
            self.screen.blit(time_surface, (10, 10))

            speed_surface = pygame.font.SysFont('Arial', 30).render(f'Vitesse: x{self.speed_factor}', True, (0, 0, 0))
            self.screen.blit(speed_surface, (10, 50))


            pygame.display.flip()
            self.clock.tick(fps)

        pygame.quit()