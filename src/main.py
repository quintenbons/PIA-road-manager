#!/usr/bin/env python3
# Cheat code for people who forgot PYTHONPATH
import sys
import os
sys.path.append(os.path.dirname(__file__))

# imports
import numpy as np
from engine.road import Road
from engine.movable.car import Car
from engine.tree import BinarySearchTree
# from engine.simulation import Simulation
# from graphics.display import PygameDisplay

def main():
    # simulation = Simulation()
    # road = Road(np.array([100, 100]), np.array([300, 200]))
    # simulation.add_road(road)
    # simulation.add_car(Car(road))

    # display = PygameDisplay(simulation)
    # display.run()
    car = Car(None)

if __name__ == "__main__":
    main()
