#!/usr/bin/env python3
# Cheat code for people who forgot PYTHONPATH
import sys
import os
sys.path.append(os.path.dirname(__file__))

# imports
# import numpy as np
from engine.road import Road
from engine.movable.car import Car
from engine.tree import BinarySearchTree
from engine.simulation import Simulation
from engine.node import Node
from engine.traffic.traffic_light import TrafficLight
# from graphics.display import PygameDisplay

def main():
    simulation = Simulation()
    node1 = Node(1, 4)
    node2 = Node(4, 3)
    node3 = Node(1, 1)

    center = Node(2, 3)

    road1 = Road(node1, center, 50, True)
    road2 = Road(node2, center, 50, False)
    road3 = Road(node3, center, 50, True)

    simulation.add_node(node1);
    simulation.add_node(node2);
    simulation.add_node(node3);
    simulation.add_node(center);
    
    trafficLight1 = TrafficLight(road1, [road3])
    trafficLight2 = TrafficLight(road2, [road1, road3])
    trafficLight3 = TrafficLight(road3, [road1])

    center.controllers = [trafficLight1, trafficLight2, trafficLight3]

    car = Car()

    car.road = road1
    car.next_road = road3
    
    simulation.add_car(car)

    simulation.run()

    # road = Road(np.array([100, 100]), np.array([300, 200]))
    # simulation.add_road(road)
    # simulation.add_car(Car(road))

    # display = PygameDisplay(simulation)
    # display.run()
    car = Car(None)

if __name__ == "__main__":
    main()
