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
from time import perf_counter
def main():

    nodes, roads = readMap("maps/test_roads.txt")
    # for road in roads:
    #     print(road)

    # print(nodes)

    print("before calculate")
    t0 = perf_counter()
    calculate_paths(nodes)
    t1 = perf_counter()
    print("time :", t1 - t0)
    print("reading")
    t0 = perf_counter()
    # read_paths(nodes, "test.txt")
    for node in nodes:
        node.printPath()
    t1 = perf_counter()
    print("time :", t1 - t0)
    print("before find_path")
    t0 = perf_counter()
    for node in nodes:
        for other in nodes:
            find_path(node, other)
    t1 = perf_counter()
    print("time :", t1 - t0)

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
    nodes = []
    roads = []
    with open(name, mode='r', encoding='utf-8') as f:
        nodes = [Node() for _ in range(int(f.readline()))]
        roads = []
        for line in f:
            
            n1, n2, length = line.split(' ')
            n1 = int(n1)
            n2 = int(n2)
            length = float(length)
            roads.append(Road(nodes[n1], nodes[n2], length, 30.0))
            roads.append(Road(nodes[n2], nodes[n1], length, 30.0))
    return nodes, roads

from math import inf
def calculate_paths(nodes: List[Node]):
    # print("")
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
                    if d[other] > d[n] + roadLen:
                        d[other] = d[n] + roadLen
                        pred[other] = n
        node.paths = pred
        # print(f"\r{node._id/len(nodes)*100:.3f}", end='')

def read_paths(nodes: List[Node], name):
    currentNode :int = None
    with open(name, mode='r', encoding='utf-8') as file:
        for line in file:
            
            l = line.strip()
            l = l.split("#")
            if len(l) == 1:
                currentNode = int(l[0])
            else:
                # print(l)
                l.pop(-1)
                for n, pred in map(lambda x: x[1:-1].split(','), l):
                    nodes[currentNode].paths[nodes[int(n)]] = nodes[int(pred)]

            

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
