#!/usr/bin/env python3
# Cheat code for people who forgot PYTHONPATH
import sys
import os
from engine.constants import TIME
sys.path.append(os.path.dirname(__file__))

from random import randint, random, seed
from maps.maps import read_map, read_paths
from engine.movable.movable import Movable
from engine.road import Road
from engine.node import Node
from typing import List

def main():

    roads, nodes = read_map("src/maps/cpp/test_roads0.txt")
    roads: List[Road]
    nodes: List[Node]
    read_paths(nodes, "src/maps/cpp/paths_roads0.txt")

    seed(0) #TODO check that random is still the same for everyone
    
    ms = []

    for _ in range(7000):
        r = roads[randint(0, 4)]
        m = Movable(1, 2, random(), random()*(r.road_len - 5), 2)
        if r.add_movable(m, 0):
            m.get_path(nodes[randint(0, 4)])
            ms.append(m)
    # print("start")
    time = 0
    from time import perf_counter
    t1 = perf_counter()
    while(time < 900):
        time += TIME
        print("time : ", time)
        for r in roads:
            r.update()
        for n in nodes:
            n.update(0)

        for m in ms.copy():
            m: Movable

            if not m.update():
                if m.road.add_movable(m, 0):
                    m.get_path(nodes[randint(0, 4)])

        #TODO supprimer cela (debug)
        for r in roads:
            for lane in r.lanes:
                prev = Movable(0, 0, 0, -1, 0)
                for mov in lane.iter():
                    if prev.pos > mov.pos:
                        print("erreur ici")
                    else:
                        prev = mov
    print(perf_counter() - t1)

if __name__ == "__main__":
    main()
