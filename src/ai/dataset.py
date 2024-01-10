from dataclasses import dataclass
from os import PathLike
from typing import Tuple
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import time
from tqdm import tqdm
from engine.constants import GENERATION_SEGMENT_DURATION
import random

from engine.node import Node
from engine.simulation import Simulation
from ai.model_constants import *
from engine.strategies.strategies_manager import StrategyManager
from engine.strategies.strategy_mutator import STRAT_NAMES, StrategyTypes

@dataclass
class NodeDataset(Dataset):
    inputs: torch.TensorType
    outputs: torch.TensorType
    seeds: torch.TensorType
    result_scores: torch.TensorType

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

    def save(self, target: PathLike):
        torch.save([self.inputs, self.outputs, self.seeds, self.result_scores], target)

    def merge(self, other):
        self.inputs = torch.cat([self.inputs, other.inputs])
        self.outputs = torch.cat([self.outputs, other.outputs])
        self.seeds = torch.cat([self.seeds, other.seeds])
        self.result_scores = torch.cat([self.result_scores, other.result_scores])
        return self

    def merge_all(self, others):
        for other in others:
            self.merge(other)
        return self

    def get_output_shape(self) -> int:
        return self.outputs.shape[1]

    @classmethod
    def load(Cls, target: PathLike, device: str = "cpu"):
        data = torch.load(target, map_location=device)
        return Cls(*data)

    @classmethod
    def from_generation(Cls, size: int, map_folder: str, tqdm_disable=True):
        inputs, outputs, sim_seeds, result_scores = generate_batch(size, map_folder, tqdm_disable)
        return Cls(inputs, outputs, sim_seeds, result_scores)

class BenchNodeDataset(NodeDataset):
    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx], self.result_scores[idx]

def entry_from_node(node: Node, tqdm_disable=True):
    tensor = torch.zeros(MAX_ROADS * 2)
    for num in tqdm(range(MAX_ROADS), disable=tqdm_disable):
        if num >= 5:
            break
        # print(node.cnode)
        node_road_in = node.get_road_in()
        node_road_out = node.get_road_out()
        # print(node_road_in, node_road_out)
        if num < len(node_road_in):
            tensor[num * 2] = node_road_in[num].get_ai_flow_count_1()
        if num < len(node_road_out):
            tensor[num * 2 + 1] = node_road_out[num].get_ai_flow_count_0()
    return tensor

def score_tester(map_folder: str, nb_controllers: int):
    sim_seed = int(time.time())
    map_file = f"{map_folder}/map.csv"
    paths_file = f"{map_folder}/paths.csv"
    central_node = 0

    strategy_manager = StrategyManager()

    for typ, mutation in strategy_manager.enumerate_strategy_schemes(nb_controllers):
        random.seed(sim_seed)
        simulation = Simulation(map_file=map_file, paths_file=paths_file, nb_movables=15)
        simulation.set_node_strategy(central_node, typ, mutation)
        simulation.run(sim_duration=GENERATION_SEGMENT_DURATION)

        sim_score = simulation.get_total_score()
        print(f"{typ:3} {STRAT_NAMES[typ]:20} {mutation:3} {sim_score:15.5f}")

def simul_to_scores(central_node: int, second_seed: int, map_folder: str, nb_controllers: int):
    map_file = f"{map_folder}/map.csv"
    paths_file = f"{map_folder}/paths.csv"

    strategy_manager = StrategyManager()

    scores = []

    total_num_schemes = 0

    for typ, mutation in strategy_manager.enumerate_strategy_schemes(nb_controllers):
        total_num_schemes += 1
        random.seed(second_seed)
        simulation = Simulation(map_file=map_file, paths_file=paths_file, nb_movables=15)
        simulation.set_node_strategy(central_node, typ, mutation)
        simulation.run(sim_duration=GENERATION_SEGMENT_DURATION)

        sim_score = simulation.get_total_score()
        scores.append(sim_score)

    return scores, total_num_schemes

def seed_generator(meta_seed: int = None):
    if meta_seed is None:
        meta_seed = random.randrange(0, 2**32-1)

    local_random = random.Random(meta_seed)

    while True:
        yield local_random.randrange(0, 2**32)

def get_soft_scores_inverted(raw_scores):
    """This function insists on the worst score being 0, and the best score being 1"""
    scores = torch.tensor(raw_scores)
    scores = scores / torch.min(scores)
    soft_scores = F.softmax(scores, dim=0)
    soft_scores = torch.max(soft_scores) - soft_scores
    return soft_scores

def get_soft_scores(raw_scores):
    """
    This function insists on the best score being 0, and the worst score being 1
    The best score will be more noticeable this way
    """
    scores = torch.tensor(raw_scores)
    scores = scores / torch.min(scores)
    scores = torch.max(scores) - scores
    soft_scores = F.softmax(scores, dim=0)
    return soft_scores

def generate_batch(size: int, map_folder: str, tqdm_disable=True) -> Tuple[torch.TensorType, torch.TensorType, torch.TensorType, torch.TensorType]:
    map_file = f"{map_folder}/map.csv"
    paths_file = f"{map_folder}/paths.csv"
    central_node = 0

    batch = []
    expected = []
    sim_seeds = []
    result_scores = []

    seed_gen = seed_generator()

    try:
        for _ in tqdm(range(size), disable=tqdm_disable):
            sim_seed = next(seed_gen)
            random.seed(sim_seed)
            second_seed = random.randrange(0, 2**32)

            # Run first simulation
            simulation = Simulation(map_file=map_file, paths_file=paths_file, nb_movables=15)
            nb_controllers = len(simulation.nodes) - 1
            simulation.set_node_strategy(central_node, StrategyTypes.CROSS_DUPLEX, 0)
            simulation.run(sim_duration=GENERATION_SEGMENT_DURATION)

            # Run second range simulationS
            raw_scores, _ = simul_to_scores(central_node, second_seed, map_folder, nb_controllers)
            soft_scores = get_soft_scores(raw_scores)

            # pad with 0s until MAX_STRATEGIES
            raw_scores = torch.cat([torch.tensor(raw_scores), torch.zeros(MAX_STRATEGIES - len(raw_scores))])
            soft_scores = torch.cat([soft_scores, torch.zeros(MAX_STRATEGIES - len(soft_scores))])

            batch.append(entry_from_node(simulation.nodes[central_node]))
            expected.append(soft_scores)
            sim_seeds.append(sim_seed)
            result_scores.append(raw_scores)

            sim_seed += 1
    except KeyboardInterrupt:
        print("Keyboard interrupt, generated incomplete dataset...")
    except Exception as e:
        print("Unknown exception occured, generated incomplete dataset...")
        print(e)

    return torch.stack(batch), torch.stack(expected), torch.tensor(sim_seeds), torch.stack(result_scores)
