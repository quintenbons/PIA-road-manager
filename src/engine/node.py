from .road import Road
from .traffic.flow_controller import FlowController
from typing import List

nid = 0

class Node:
    roadIn: List[Road]
    roadOut: List[Road]
    controller: List[FlowController]

    _id: int

    def __init__(self):
        #TODO remove it later
        global nid
        self._id = nid
        nid += 1
        self.roadIn = []
        self.roadOut = []
        self.controller = []

    def addRoadIn(self, road: Road):
        self.roadIn.append(road)

    def addRoadOut(self, road: Road):
        self.roadOut.append(road)

    def __str__(self) -> str:
        return str(self._id)