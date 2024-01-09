"""
Main interface for the display.
"""
import pygame
from engine.constants import TIME
from engine.simulation import Simulation
from graphics.assets import AssetManager
from graphics.utils import get_clicked_movable, get_clicked_node

from graphics.init_pygame import pygame_init
from graphics.draw import create_grid_surface, draw_movable, draw_node, draw_road, draw_hud, draw_paused_text
from graphics.constants import MAX_SCALE_VALUE, SCREEN_WIDTH, SCREEN_HEIGHT


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
        self.engine_x_min = float('inf')
        self.engine_x_max = float('-inf')
        self.engine_y_min = float('inf')
        self.engine_y_max = float('-inf')
        self._calculate_engine_bounds()

        pygame.display.set_caption("Simulation de réseau routier")
        if self.debug_mode:
            print("Debug mode enabled")
            print("Press Space to advance 10 steps")

        self.clock = pygame.time.Clock()

    def _calculate_engine_bounds(self):
        for node in self.simulation.nodes:
            self.engine_x_min = min(self.engine_x_min, node.cnode.get_x())
            self.engine_x_max = max(self.engine_x_max, node.cnode.get_x())
            self.engine_y_min = min(self.engine_y_min, node.cnode.get_y())
            self.engine_y_max = max(self.engine_y_max, node.cnode.get_y())

        self.scale_factor = min(SCREEN_WIDTH / (self.engine_x_max - self.engine_x_min),
                    SCREEN_HEIGHT / (self.engine_y_max - self.engine_y_min))
        self.scale_factor = min(self.scale_factor, 4)

    def draw(self):
        self.screen.fill((255, 255, 255))
        if self.debug_mode:
            self.screen.blit(self.grid_surface, (0, 0))
        for road in self.simulation.roads:
            draw_road(self.screen, road, self.engine_x_min, self.engine_x_max,
                      self.engine_y_min, self.engine_y_max, SCREEN_WIDTH, SCREEN_HEIGHT, self.scale_factor)
        for node in self.simulation.nodes:
            draw_node(self.screen, node, self.engine_x_min, self.engine_x_max,
                      self.engine_y_min, self.engine_y_max, SCREEN_WIDTH, SCREEN_HEIGHT, self.scale_factor)
        for spawner in self.simulation.spawners:
            for movable in spawner.movables:
                color = self.asset_manager.get_car_asset(movable)
                draw_movable(movable, self.screen, color, self.engine_x_min, self.engine_x_max,
                             self.engine_y_min, self.engine_y_max, SCREEN_WIDTH, SCREEN_HEIGHT, self.scale_factor)

        draw_hud(self)

    def run(self, max_time: int = None):
        print("Start simulation \n ----------------------------------")
        self.screen = pygame_init()
        self.grid_surface = create_grid_surface(self.screen)
        running = True

        self.speed_factor = 1
        base_fps = 45
        base_tick_interval = TIME * 1000
        last_tick_time = pygame.time.get_ticks()

        while running:
            current_time = pygame.time.get_ticks()
            time_since_last_tick = current_time - last_tick_time

            if max_time is not None and max_time <= self.simulation.current_tick * TIME:
                self.paused = True

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
                    elif event.unicode == '+':  # Unicode is better than key, since it will work with modifiers
                        if self.speed_factor == 128:
                            print("Speed factor already at max")
                            break
                        self.speed_factor *= 2
                    elif event.unicode == '-':
                        self.speed_factor = max(1, self.speed_factor // 2)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        pos = pygame.mouse.get_pos()
                        for spawner in self.simulation.spawners:
                            clicked_movable = get_clicked_movable(
                                movables=spawner.movables, pos=pos)
                        if clicked_movable:
                            print("Clicked on car: ", clicked_movable)
                            break
                        clicked_node = get_clicked_node(
                            nodes=self.simulation.nodes, pos=pos)
                        if clicked_node:
                            print("Clicked on node: ", clicked_node)

            self.draw()

            if not self.debug_mode:
                if not self.paused:
                    if time_since_last_tick >= base_tick_interval / self.speed_factor:
                        self.simulation.run_tick()
                        last_tick_time = current_time
                else:
                    draw_paused_text(self)

            pygame.display.flip()

            if self.speed_factor > 8:
                self.clock.tick(base_fps*2)
            else:
                self.clock.tick(base_fps)

        pygame.quit()
