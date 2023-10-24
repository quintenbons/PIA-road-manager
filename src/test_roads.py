#!/usr/bin/env python3
# Cheat code for people who forgot PYTHONPATH
import sys
import os
sys.path.append(os.path.dirname(__file__))

# imports
import numpy as np
from engine.road import Road
from engine.node import Node
from engine.movable.movable import Movable
from typing import List

def main():

    nodes, roads = readMap("maps/test_roads.txt")

def readMap(name) -> (List[Node], List[Road]):
    # number of nodes
    # node1 node2 length
    with open(name, mode='r', encoding='utf-8') as f:
        nodes = [Node() for _ in range(int(f.readline()))]
        roads = []
        for line in f.readlines():
            n1, n2, length = line.split(' ')
            n1 = int(n1)
            n2 = int(n2)
            length = int(length)
            roads.append(Road(nodes[n1], nodes[n2], length, 30.0))
            roads.append(Road(nodes[n2], nodes[n1], length, 30.0))
    return nodes, roads

if __name__ == "__main__":
    main()
