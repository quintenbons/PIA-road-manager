from engine.road import Road
from engine.node import Node
from engine.movable.movable import Movable

def get_clicked_movable(self, pos: tuple[int, int]) -> Movable:
    for m in self.movables:
        if m.get_rect().collidepoint(pos):
            return m
    return None

def get_clicked_node(self, pos: tuple[int, int]) -> Node:
    for node in self.nodes:
        print(node.position)
        if node.get_rect().collidepoint(pos):
            return node
    return None