from .road import Road
from .traffic.flow_controller import FlowController
from typing import List


class Node:
    roadIn: List[Road]
    roadOut: List[Road]
    controller: List[FlowController]
