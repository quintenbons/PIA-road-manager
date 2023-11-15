#!/usr/bin/env python3
# Cheat code for people who forgot PYTHONPATH
import sys
import os
from typing import List
from engine.constants import TIME
from engine.node import Node
sys.path.append(os.path.dirname(__file__))

from maps.maps import read_map, read_paths
from engine.movable.movable import Movable
from engine.road import Road

moveables = [
    lambda: Movable(0, 3, 0.1, 0, 1),
    lambda: Movable(0, 2, 0.1, 0, 1),
    lambda: Movable(1, 3, 0.1, 0, 1),
    lambda: Movable(0, 3, 0.1, 0, 1),
]

def main():
    nodes = [
        Node(1, 1), # Center
        Node(0, 0), # Top left
        Node(2, 0), # Top right
        Node(2, 2), # Bottom right
        Node(1, 2), # Bottom center
        Node(0, 1), # Left
    ]
    roads: List[Road] = []

    for i, node in enumerate(nodes[1:]):
        roads.append(Road(nodes[0], node, 13., 100.))
        roads.append(Road(node, nodes[0], 13., 100.))

    # wave = (time, node, destination, moveable_type)
    waves = [
        (0, 1, 2, 1),
        (1, 2, 2, 1),
        (30, 1, 2, 1),
    ]

    total_duration = 30 * 60 # 30 minutes
    time = 0
    wavepos = 0
    while(time < total_duration):
        time += TIME
        for r in roads:
            r.update()

        while waves[wavepos][0] > time:
            # spawn voiture
            wavepos += 1

if __name__ == "__main__":
    main()
