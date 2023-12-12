from engine.road import Road
from engine.node import Node
from engine.movable.movable import Movable
from typing import List
from graphics.draw import get_rect


def get_clicked_movable( movables: List[Movable], pos: tuple[int, int]) -> Movable:
    for m in movables:
        if get_rect(m).collidepoint(pos):
            return m
    return None

def get_clicked_node(nodes: List[Movable], pos: tuple[int, int]) -> Node:
    for node in nodes:
        if get_rect(node).collidepoint(pos):
            return node
    return None