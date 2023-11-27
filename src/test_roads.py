#!/usr/bin/env python3
# Cheat code for people who forgot PYTHONPATH
import sys
import os
import pygame
from engine.constants import TIME
from graphics.draw import draw_movable, draw_node, draw_road, draw_grid
from graphics.init_pygame import pygame_init
sys.path.append(os.path.dirname(__file__))

from random import randint, random, seed
from maps.maps import read_map, read_paths
from engine.movable.movable import Movable
from engine.road import Road
from engine.node import Node
from typing import List

def main():
    DEBUG_MODE = False
    step = 0
    grid_size = 50

    roads, nodes = read_map("src/maps/cpp/eybens_map.csv")
    roads: List[Road]
    nodes: List[Node]
    read_paths(nodes, "src/maps/cpp/eybens_paths_roads.txt")

    seed(0) #TODO check that random is still the same for everyone
    
    ms = []

    screen = pygame_init()
    pygame.display.set_caption("Simulation de réseau routier")
    clock = pygame.time.Clock()
    running = True


    for _ in range(300):
        r = roads[randint(0, len(roads)-1)]
        m = Movable(1, 2, random(), random()*(r.road_len), 2)
        if r.add_movable(m, 0):
            m.get_path(nodes[randint(0, 4)])
            ms.append(m)
    print("start")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE and DEBUG_MODE:
                        step = 10

        if not DEBUG_MODE or step > 0:
            for _ in range(step or 1):  # Exécute 1 ou 10 étapes
                for r in roads:
                    r.update()
                for n in nodes:
                    n.update(0)
                for m in ms.copy():
                    m: Movable

                    if not m.update():
                        if m.road.add_movable(m, 0):
                            m.get_path(nodes[randint(0, 4)])
            step -= 1 if step > 0 else 0

        screen.fill((255, 255, 255))
        draw_grid(screen, grid_size, screen.get_size())

        for road in roads:
            draw_road(screen, road)
        for node in nodes:
            draw_node(screen, node)
        for movable in ms:
            draw_movable(movable, screen)

        pygame.display.flip()
        clock.tick(60)  # 60 FPS

    pygame.quit()

if __name__ == "__main__":
    main()