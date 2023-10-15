#!/usr/bin/env python3
from engine.simulation import Simulation
from engine.car import Car
from engine.road import Road
from graphics.display import PygameDisplay

def main():
    simulation = Simulation()
    simulation.add_car(Car(100, 100))
    simulation.add_road(Road((100, 100), (300, 200)))

    display = PygameDisplay(simulation)
    display.run()

if __name__ == "__main__":
    main()
