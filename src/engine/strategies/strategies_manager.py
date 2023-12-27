from random import Random
from engine.node import Node
from engine.constants import DEFAULT_DURATION
from engine.strategies.model.cross_duplex_strategy import CrossDuplexStrategy
from engine.strategies.model.open_corridor_strategy import OpenCorridorStrategy

from engine.strategies.model.open_strategy import OpenStrategy
from engine.strategies.model.piece_of_cake_strategy import PieceOfCakeStrategy
from engine.strategies.strategy_mutator import StrategyMutator, StrategyTypes


class StrategyManager:
    mutations = {}
    random = None

    def __init__(self):
        self.random = Random(1)
        mutator = StrategyMutator()
        for i in range(10):
            self.mutations[i] = mutator.get_strategies_mutations(i, DEFAULT_DURATION)
            # total = 0
            # for j in range(4):
            #     print("nb controllers: ", i, " type: ", j, " mutations: ", len(self.mutations[i][j]))
            #     total += len(self.mutations[i][j])
            # print("total: ", total)

    def get_strategy(self, node:Node, type: StrategyTypes, mutation: int):
        controller_count = len(node.controllers)
        if len(self.mutations[controller_count][type]) <= mutation:
            return None
        mutation_parameters = self.mutations[controller_count][type][mutation]

        if type == StrategyTypes.CROSS_DUPLEX:
            return CrossDuplexStrategy(node.controllers, node.get_position(), mutation_parameters)
        elif type == StrategyTypes.OPEN_CORRIDOR:
            return OpenCorridorStrategy(node.controllers, mutation_parameters)
        elif type == StrategyTypes.OPEN:
            return OpenStrategy(node.controllers, mutation_parameters)
        elif type == StrategyTypes.PIECE_OF_CAKE:
            return PieceOfCakeStrategy(node.controllers, mutation_parameters)
        else:
            return None

    def enumerate_strategy_schemes(self, controller_count: int):
        for type in range(StrategyTypes.length):
            for mutation in range(len(self.mutations[controller_count][type])):
                yield type, mutation

    def enumerate_strategies(self, node: Node):
        for type, mutation in self.enumerate_strategy_schemes(len(node.controllers)):
            yield self.get_strategy(node, type, mutation)

    def get_random_strategy(self, node: Node):
        type = self.random.randint(0, StrategyTypes.length - 1)
        mutation = self.random.randint(0, len(self.mutations[len(node.controllers)][type]) - 1)
        return self.get_strategy(node, type, mutation)