#!/usr/bin/python3
import argparse
from engine.simulation import Simulation

def main():
    parser = argparse.ArgumentParser(description='Lancez une simulation de réseau routier.')
    parser.add_argument('--nb_movables', type=int, default=1, help='Nombre de movables dans la simulation')

    args = parser.parse_args()

    simulation = Simulation(debug_mode=True, nb_movables=args.nb_movables)
    simulation.run()

if __name__ == "__main__":
    main()