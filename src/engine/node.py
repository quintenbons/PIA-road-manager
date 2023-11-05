from __future__ import annotations
from .road import Road
from .traffic.flow_controller import FlowController
from typing import List

nid = 0

class Node:
    roadIn: List[Road]
    roadOut: List[Road]
    controller: List[FlowController]

    paths: dict[Node: Node]
    _id: int

    def __init__(self):
        #TODO remove it later
        global nid
        self._id = nid
        nid += 1
        self.roadIn = []
        self.roadOut = []
        self.controller = []
        self.paths = {}

    def addRoadIn(self, road: Road):
        self.roadIn.append(road)

    def addRoadOut(self, road: Road):
        self.roadOut.append(road)

    def pathTo(self, node: Node) -> List[Node]:
        return self.paths[node]

    def neighbors(self) -> Node:
        for r in self.roadOut:
            yield r.end, r.length()

    def __str__(self) -> str:
        return str(self._id)

    def printPath(self):
        print(self._id)
        for p in self.paths:
            print(f"{p._id,self.paths[p]._id}", end='#')
        print("")