from __future__ import annotations
import sys
from math import inf

from typing import List

sys.path.append('../engine')
from engine.road import Road
from engine.node import Node

def read_map(name: str) -> (List[Road], List[Node]):

    with open(name, mode='r', encoding='utf-8') as f:
        nodes = [Node(0, 0) for _ in range(int(f.readline()))]
        roads = []

        for line in f:
            n1, n2, length, *_ = line.split()
            n1 = int(n1)
            n2 = int(n2)
            length = float(length)
            
            #TODO change speedlimit and remove second line
            roads.append(Road(nodes[n1], nodes[n2], 8, length))
            roads.append(Road(nodes[n2], nodes[n1], 8, length))
    return roads, nodes

def read_paths(nodes: List[Node], name: str):
    """ Read paths from a paths' file """
    currentNode :int = None
    with open(name, mode='r', encoding='utf-8') as f:
        for line in f:
            l = line.strip()
            l = l.split('|')
            if len(l) == 1:
                currentNode = int(l[0])
            else:
                #TODO optim ?
                l.pop(-1)
                for n, previous in map(lambda x: x.split(':'), l):
                    nodes[currentNode].paths[nodes[int(n)]] = nodes[int(previous)]

def find_path(n1: Node, n2: Node) -> List[Node]:
    paths = n1.paths
    current = n2
    path = []
    while current != n1:
        path.append(current)
        current = paths[current]
    path.append(n1)

    return path

"""
I don't knows how to mark function as deprecated

Don't use above functions as they are for test purpose only
"""
def calculatePath(nodes: List[Node]):
    for node in nodes:
        P = []
        d = {n:inf for n in nodes}
        pred = {}
        d[node] = 0.0
        Q = [n for n in nodes]
        while len(Q) > 0:
            dist, n = findMinDist(Q, d)
            P.append(n)
            Q.remove(n)
            for other, roadLen in n.neighbors():
                if other not in P:
                    if d[other] > d[n] + roadLen:
                        d[other] = d[n] + roadLen
                        pred[other] = n
        node.paths = pred

def findMinDist(Q, d):
    m = inf
    node = None

    for q in Q:
        if d[q] < m:
            m = d[q]
            node = q
    return m, node