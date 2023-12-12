from yaml import Node
from engine.constants import TIME
from engine.strategies.model.cross_duplex_strategy import CrossDuplexStrategy
from engine.strategies.model.open_corridor_strategy import OpenCorridorStrategy

from engine.strategies.model.open_strategy import OpenStrategy
from engine.strategies.model.piece_of_cake_strategy import PieceOfCakeStrategy


class StrategyTypes:
    length = 4

    CROSS_DUPLEX = 0
    OPEN_CORRIDOR = 1
    PIECE_OF_CAKE = 2
    OPEN = 3

class StrategyMutator:

    def __init__(self):
        pass

    def get_strategies_mutations(self, nb_controllers: int, default_cycle_duration: int):
        # Cross duplex
        cross_duplex_mutations = []
        # -> Default cycle
        cross_duplex_mutations.append([default_cycle_duration for _ in range(nb_controllers)])
        # -> 1 mutation for each road that takes 1.5 times the default cycle
        for i in range(nb_controllers):
            cross_duplex_mutations.append([default_cycle_duration for _ in range(nb_controllers)])
            cross_duplex_mutations[i + 1][i] = default_cycle_duration * 1.5
        
        # Open corridor
        # -> 1 mutation for each open road
        open_corridor_mutations = []
        for i in range(nb_controllers):
            open_corridor_mutations.append([default_cycle_duration for _ in range(nb_controllers)])
            open_corridor_mutations[i][i] = None

        # Piece of cake
        piece_of_cake_mutations = []
        piece_of_cake_mutations.append([default_cycle_duration for _ in range(nb_controllers)])
        # -> 1 mutation for each road that takes 1.5 times the default cycle
        for i in range(nb_controllers):
            if nb_controllers == 1:
                piece_of_cake_mutations.append([default_cycle_duration])
                continue
            piece_of_cake_mutations.append([default_cycle_duration for _ in range(nb_controllers)])
            piece_of_cake_mutations[i][i] = default_cycle_duration * 1.5
        
        # Open
        open_mutations = []
        open_mutations.append([None for _ in range(nb_controllers)])

        return {
            StrategyTypes.CROSS_DUPLEX: cross_duplex_mutations,
            StrategyTypes.OPEN_CORRIDOR: open_corridor_mutations,
            StrategyTypes.PIECE_OF_CAKE: piece_of_cake_mutations,
            StrategyTypes.OPEN: open_mutations
        }