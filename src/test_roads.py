#!/usr/bin/env python3
# Cheat code for people who forgot PYTHONPATH
import sys
import os
from engine.constants import TIME
sys.path.append(os.path.dirname(__file__))

from maps.maps import read_map, read_paths
from engine.movable.movable import Movable
from engine.road import Road
from typing import List
# imports

def main():

    roads, nodes = read_map("maps/cpp/test_roads0.txt")
    roads: List[Road]
    read_paths(nodes, "maps/cpp/paths_roads0.txt")
    # for road in roads:
    #     print(road)
    m0 = Movable(0.0, 0.0, 0.1, 10, 1)
    m1 = Movable(4, 3.0, 0.1, 0, 1)
    # m0.set_road(roads[4])
    roads[4].add_movable(m1, 0)
    roads[4].add_movable(m0, 0)
    m0.get_path(nodes[0])
    m1.get_path(nodes[0])

    print("start")
    time = 0
    while(True):
        time += TIME
        for r in roads:
            r.update()
            for lane in r.lanes:
                for m in lane:
                    print(f"mid = {m._id}, rid = {m.road.start._id}, pos = {m.pos:0.1f}, speed = {m.speed:0.1f}, road_len = {m.road.length}")
        if time >= 30:
            break
    print(time)
    # print(nodes)


if __name__ == "__main__":
    main()
