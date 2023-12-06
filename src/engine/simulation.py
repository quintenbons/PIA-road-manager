#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(__file__))
import pygame
from engine.constants import TIME
from graphics.draw import draw_movable, draw_node, draw_road, draw_grid
from graphics.init_pygame import pygame_init
from random import randint, random, seed
from maps.maps_functions import read_map, read_paths
from engine.movable.movable import Movable
from engine.road import Road
from engine.node import Node
from typing import List
import time

class Simulation:
    def __init__(self, map_file: str=None, paths_file: str=None,debug_mode: bool = False, grid_size: int = 50, nb_movables: int = 1):
        self.debug_mode = debug_mode
        self.grid_size = grid_size
        self.nb_movables = nb_movables
        print("\n\n ---------------------------------- \n")
        
        if not map_file:
            map_file = "src/maps/cpp/eybens_map.csv"
            print("No map file specified, using default map: " + map_file)
        if not paths_file:
            paths_file = "src/maps/cpp/eybens_paths_roads.txt"
            print("No paths file specified, using default paths: " + paths_file)

        self.roads, self.nodes = read_map(map_file)
        read_paths(self.nodes, paths_file)

        if self.debug_mode:
            print("Debug mode enabled")
            print("press space to advance 10 steps")
        
        
        self.movables = []
        self.screen = pygame_init()
        pygame.display.set_caption("Simulation de réseau routier")
        self.clock = pygame.time.Clock()
        self.running = True

    def add_movables(self, count: int = 1):
        for _ in range(count):
            r = self.roads[randint(0, len(self.roads) - 1)]
            m = Movable(1, 2, random(), random() * (r.road_len), 2)
            if r.add_movable(m, 0):
                m.get_path(self.nodes[randint(0, len(self.nodes) - 1)])
                self.movables.append(m)

    def run(self):
        step = 0
        seed(0)
        self.add_movables(self.nb_movables)
        print("Start simulation \n ----------------------------------")

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN and self.debug_mode:
                    if event.key == pygame.K_SPACE:
                        step = 10

            if not self.debug_mode or step > 0:
                for _ in range(step or 1):
                    for r in self.roads:
                        r.update()
                    for n in self.nodes:
                        n.update(0)
                    for m in self.movables.copy():
                        if not m.update():
                            if m.road.add_movable(m, 0):
                                m.get_path(self.nodes[randint(0, len(self.nodes) - 1)])
                step -= 1 if step > 0 else 0

            self.screen.fill((255, 255, 255))
            # if self.debug_mode:
            #     draw_grid(self.roads, self.nodes, self.grid_size)
            for road in self.roads:
                draw_road(self.screen, road)
            for node in self.nodes:
                draw_node(self.screen, node)
            for movable in self.movables:
                draw_movable(movable, self.screen)

            pygame.display.flip()
            self.clock.tick(60)  # 60 FPS

        pygame.quit()

if __name__ == "__main__":
    simulation = Simulation(debug_mode=True)
    simulation.run()