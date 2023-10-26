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
    for road in roads:
        print(road)

    calculate_paths(nodes)
    u = find_path(nodes[0], nodes[2])
    for e in u:
        print(e)

def find_path(n1: Node, n2: Node):
    paths = n1.paths
    current = n2
    path = []
    while current != n1:
        path.append(current)
        current = paths[current]
    path.append(n1)
    return path

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

from math import inf
def calculate_paths(nodes: List[Node]):
    for node in nodes:
        P = []
        d = {n:inf for n in nodes}
        pred = {}
        d[node] = 0.0
        Q = [n for n in nodes]
        while len(Q) > 0:
            dist, n = find_min_dist(Q, d)
            P.append(n)
            Q.remove(n)
            for other, roadLen in n.neighbors():
                if other not in P:
                    d[other] = d[n] + roadLen
                    pred[other] = n
        node.paths = pred
        
def find_min_dist(Q, d):
    m = inf
    node = None

    for q in Q:
        if d[q] < m:
            m = d[q]
            node = q
    return m, node

if __name__ == "__main__":
    main()
