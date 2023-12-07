from yaml import Node
from engine.constants import TIME
from engine.strategies.cross_duplex_strategy import CrossDuplexStrategy
from engine.strategies.open_corridor_strategy import OpenCorridorStrategy

from engine.strategies.open_strategy import OpenStrategy
from engine.strategies.piece_of_cake_strategy import PieceOfCakeStrategy


class StrategyTypes:
    length = 4

    CROSS_DUPLEX = 0
    OPEN_CORRIDOR = 1
    PIECE_OF_CAKE = 2
    OPEN = 3

class StrategyMutator:
    max_mutations = 0

    def __init__(self):
        pass

    def get_strategies(self, node: Node):
        strategies = []
        for i in range(StrategyTypes.length):
            strategies.append(self.get_mutation_handler(i)(node))

    def get_mutation_handler(self, strategy_type: int):
        handlers = [
            self.mutate_cross_duplex,
            self.mutate_open_corridor,
            self.mutate_piece_of_cake,
            self.mutate_open
        ]
        return handlers[strategy_type]
        
    def mutate_open_corridor(self, node: Node):
        return [OpenCorridorStrategy(node)]

    def mutate_cross_duplex(self, node: Node):
        return [CrossDuplexStrategy(node)]

    def mutate_piece_of_cake(self, node: Node):
        strategies = []
        for i in range(TIME, TIME * 3, TIME / 2):
            pieceOfCake = PieceOfCakeStrategy(node, i)
            pieceOfCake.time_per_state = i
            strategies.append(pieceOfCake)
        return strategies

    def mutate_open(self, node: Node):
        return [OpenStrategy(node)]
