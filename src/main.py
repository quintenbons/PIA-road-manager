#!/usr/bin/python3
import argparse
import os
from engine.simulation import Simulation
from graphics.display import PygameDisplay

def main():
    parser = argparse.ArgumentParser(description='Lancez une simulation de réseau routier.')
    parser.add_argument('--debug', action='store_true', help='Mode debug')
    parser.add_argument('--nb_movables', type=int, default=1, help='Nombre de movables dans la simulation')
    parser.add_argument('--map_file', type=str, help='Fichier de la map')
    parser.add_argument('--paths_file', type=str, help='Fichier des chemins, en général pas besoin de le spécifier')
    args = parser.parse_args()

    if not args.paths_file:
        parent_dir = os.path.dirname(args.map_file)
        args.paths_file = os.path.join(parent_dir, "paths.csv")

    simulation = Simulation(map_file=args.map_file, paths_file=args.paths_file)
    simulation.add_movables(args.nb_movables)
    display = PygameDisplay(simulation, debug_mode=args.debug)
    display.run()

if __name__ == "__main__":
    main()