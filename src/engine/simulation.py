#!/usr/bin/env python3
import sys
import os

from engine.strategies.strategies_manager import StrategyManager
sys.path.append(os.path.dirname(__file__))
import pygame
from engine.constants import TIME
from graphics.draw import draw_movable, draw_node, draw_road, create_grid_surface
from graphics.init_pygame import pygame_init
from random import randint, random, seed
from maps.maps_functions import read_map, read_paths, set_strategies, set_traffic_lights
from engine.movable.movable import Movable
from engine.road import Road
from engine.node import Node
from typing import List
import time

class Simulation:
    strategy_manager: StrategyManager

    def __init__(self, map_file: str, paths_file: str,debug_mode: bool = False, nb_movables: int = 1):
        self.debug_mode = debug_mode
        self.nb_movables = nb_movables
        print("\n\n ---------------------------------- \n")

        self.strategy_manager = StrategyManager()

        self.roads, self.nodes = read_map(map_file)
        read_paths(self.nodes, paths_file)
        set_traffic_lights(self.nodes)
        set_strategies(self.nodes, self.strategy_manager)

        
        self.movables: List[Movable] = []
        self.screen = pygame_init()

        pygame.display.set_caption("Simulation de réseau routier")
        if self.debug_mode:
            print("Debug mode enabled")
            print("press Space to advance 10 steps")
            self.grid_surface = create_grid_surface(self.screen)


        self.clock = pygame.time.Clock()
        self.running = True

    def add_movables(self, count: int = 1):
        for _ in range(count):
            r = self.roads[randint(0, len(self.roads) - 1)]
            m = Movable(5, 2, random(), random() * (r.road_len), 2)
            if r.add_movable(m, 0):
                m.get_path(self.nodes[randint(0, len(self.nodes) - 1)])
                self.movables.append(m)

    def get_clicked_movable(self, pos: tuple[int, int]) -> Movable:
        for m in self.movables:
            if m.get_rect().collidepoint(pos):
                return m
        return None

    def run(self):
        step = 0
        seed(0)
        self.add_movables(self.nb_movables)
        print("Start simulation \n ----------------------------------")
        loop_timer = 0
        while self.running:
            loop_timer += 1
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN and self.debug_mode:
                    if event.key == pygame.K_SPACE:
                        step = 10
                elif event.type == pygame.MOUSEBUTTONDOWN and self.debug_mode:
                    if event.button == 1:
                        pos = pygame.mouse.get_pos()
                        clicked_movable = self.get_clicked_movable(pos)
                        if clicked_movable:
                            print("Clicked on car: ", clicked_movable)
                            self.selected_movable = clicked_movable

            if not self.debug_mode or step > 0:
                for _ in range(step or 1):
                    for r in self.roads:
                        r.update()
                    for n in self.nodes:
                        n.update(loop_timer)
                    for m in self.movables.copy():

                        if not m.update():
                            # self.movables.remove(m)
                            # m.pos = m.road.road_len - 5
                            # m.pos = 0
                            if m.road.add_movable(m, 0):
                                u = randint(0, len(self.nodes) - 1)
                                # print(u)
                                m.get_path(self.nodes[u])
                step -= 1 if step > 0 else 0

            self.screen.fill((255, 255, 255))
            if self.debug_mode:
                self.screen.blit(self.grid_surface, (0, 0)) 
            for road in self.roads:
                draw_road(self.screen, road)
            for node in self.nodes:
                draw_node(self.screen, node)
            for movable in self.movables:
                draw_movable(movable, self.screen)

            pygame.display.flip()
            self.clock.tick(30)  # FPS

        pygame.quit()

if __name__ == "__main__":
    simulation = Simulation(debug_mode=True)
    simulation.run()