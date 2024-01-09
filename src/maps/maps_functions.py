from __future__ import annotations
from io import TextIOWrapper
import sys
from math import inf

from typing import List, Tuple
from engine.spawners.spawner import Spawner
from engine.spawners.spawner_utils import spawner_frequencies
from engine.strategies.strategies_manager import StrategyManager
from engine.strategies.strategy_mutator import StrategyTypes

from graphics.constants import SCREEN_WIDTH, SCREEN_HEIGHT

from engine.traffic.traffic_light import TrafficLight
sys.path.append('../engine')

from engine.road import Road
from engine.node import Node

def normalize_coordinates(nodes, width, height):
    min_x = min(node.cnode.get_x() for node in nodes)
    min_y = min(node.cnode.get_y() for node in nodes)
    max_x = max(node.cnode.get_x() for node in nodes)
    max_y = max(node.cnode.get_y() for node in nodes)

    padding_x = width * 0.05
    padding_y = height * 0.05

    scaled_width = width - 2 * padding_x
    scaled_height = height - 2 * padding_y

    for node in nodes:
        x = (node.cnode.get_x() - min_x) / (max_x - min_x) * scaled_width + padding_x
        y = (node.cnode.get_y() - min_y) / (max_y - min_y) * scaled_height + padding_y
        node.cnode.set_position(x, y)

def read_spawner(io: TextIOWrapper, all_roads: List[Road], road_dict: dict) -> Spawner | None:
    """File format
    frequency (slow, medium fast)
    initial movables (int)
    picking type (uniform or picked)

    # if picked:
    source1
    source2
    =
    dest1
    ...
    """
    first_line = io.readline().strip()

    #EOF or parsing problem
    if len(first_line) <= 1:
        return None

    freq = spawner_frequencies(first_line)
    initial_rate = int(io.readline().strip())
    picker = io.readline().strip()

    if picker == "uniform":
        io.readline() # skip ==
        return Spawner(all_roads, all_roads, freq, initial_rate) # TODO: initial freq

    sources = []
    destinations = []
    sourceMode = True
    for line in io:
        match line.strip().split():
            case ["=="]:
                return Spawner(sources, destinations, freq, initial_rate)
            case ["="]:
                sourceMode = False
            case [n1, n2] if len(line.strip().split()) == 2:
                n1, n2 = int(n1), int(n2)
                if sourceMode:
                    sources.append(road_dict[(n1, n2)])
                else:
                    destinations.append(road_dict[(n1, n2)])
            case _ as any:
                raise Exception("Invalid line in spawner file", any)

def read_map(name: str) -> Tuple[List[Road], List[Node], List[Spawner]]:
    nodes: List[Node] = []
    roads: List[Road] = []
    roads_dictionnary: dict = {}
    spawners: List[Spawner] = []

    with open(name, mode='r', encoding='utf-8') as f:
        for line in f:
            if line.strip() == "===":
                break
            x, y, *_ = line.split()
            nodes.append(Node(float(x), float(y)))

        for line in f:
            if line.strip() == "===":
                break

            n1, n2, *_ = line.split()
            n1 = int(n1)
            n2 = int(n2)
            
            #TODO change speedlimit and remove second line
            road1 = Road(nodes[n1], nodes[n2], 8)
            road2 = Road(nodes[n2], nodes[n1], 8)
            roads.append(road1)
            roads.append(road2)

            roads_dictionnary[(n1, n2)] = road1
            roads_dictionnary[(n2, n1)] = road2

        # Spawners
        while (spawner := read_spawner(f, roads, roads_dictionnary)) is not None:
            spawners.append(spawner)

    return roads, nodes, spawners

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
                    # nodes[currentNode].paths[nodes[int(n)]] = nodes[int(previous)]
                    nodes[currentNode].cnode.set_path(nodes[int(n)].cnode, nodes[int(previous)].cnode)

def set_traffic_lights(nodes: List[Node]):
    for node in nodes:
        for road in node.cnode.get_road_in():
            trafficLight = TrafficLight(road, node.cnode.get_road_out())
            node.controllers.append(trafficLight)

def set_strategies(nodes: List[Node], strategy_manager: StrategyManager, benchmark: bool):
    for node in nodes:
        if benchmark:
            node.set_strategy(strategy_manager.get_strategy(node, StrategyTypes.PIECE_OF_CAKE, 0))
        else:
            node.set_strategy(strategy_manager.get_strategy(node, StrategyTypes.OPEN_CORRIDOR, 0))
            # node.set_strategy(strategy_manager.get_strategy(node, StrategyTypes.OPEN, 0))
        

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