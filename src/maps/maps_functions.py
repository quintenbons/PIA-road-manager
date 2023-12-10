from __future__ import annotations
import sys
from math import inf

from typing import List
from engine.strategies.cross_duplex_strategy import CrossDuplexStrategy
from engine.strategies.open_corridor_strategy import OpenCorridorStrategy
from engine.strategies.open_strategy import OpenStrategy
from engine.strategies.piece_of_cake_strategy import PieceOfCakeStrategy
from engine.strategies.strategies_manager import StrategyManager
from engine.strategies.strategy_mutator import StrategyTypes

from engine.traffic.traffic_light import TrafficLight
sys.path.append('../engine')

from engine.road import Road
from engine.node import Node

def normalize_coordinates(nodes, width, height):
    min_x = min(node.position[0] for node in nodes)
    min_y = min(node.position[1] for node in nodes)
    max_x = max(node.position[0] for node in nodes)
    max_y = max(node.position[1] for node in nodes)

    padding_x = width * 0.05
    padding_y = height * 0.05

    scaled_width = width - 2 * padding_x
    scaled_height = height - 2 * padding_y

    for node in nodes:
        x = (node.position[0] - min_x) / (max_x - min_x) * scaled_width + padding_x
        y = (node.position[1] - min_y) / (max_y - min_y) * scaled_height + padding_y
        node.position = (x, y)

def read_map(name: str) -> (List[Road], List[Node]):

    with open(name, mode='r', encoding='utf-8') as f:
        nodes = []
        roads = []

        for line in f:

            if line.strip() == "===":
                break
            x, y, *_ = line.split()
            nodes.append(Node(float(x), float(y)))

        normalize_coordinates(nodes, 1200, 800) # todo: remove hard coded values

        for line in f:
            n1, n2, *_ = line.split()
            n1 = int(n1)
            n2 = int(n2)
            
            #TODO change speedlimit and remove second line
            roads.append(Road(nodes[n1], nodes[n2], 8))
            roads.append(Road(nodes[n2], nodes[n1], 8))
    
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

def set_traffic_lights(nodes: List[Node]):
    for node in nodes:
        for road in node.road_in:
            trafficLight = TrafficLight(road, node.road_out)
            node.controllers.append(trafficLight)

def set_strategies(nodes: List[Node], strategy_manager: StrategyManager):
    for node in nodes:
        node.set_strategy(strategy_manager.get_random_strategy(node))
        

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
Deprecated function. It's our only one so no need to use a decorator, just print a warning
"""
def calculatePath(nodes: List[Node]):
    print("Warning: calculatePath is deprecated and used for testing purpose only, use read_paths instead")
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