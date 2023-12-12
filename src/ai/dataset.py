from dataclasses import dataclass
from os import PathLike
from typing import Tuple
import torch
from torch.utils.data import Dataset
import time
from tqdm import tqdm
from engine.constants import GENERATION_SEGMENT_DUARTION
import random

from engine.node import Node
from engine.simulation import Simulation
from ai.model_constants import *

@dataclass
class NodeDataset(Dataset):
    inputs: torch.TensorType
    outputs: torch.TensorType
    seed: int

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

    def save(self, target: PathLike):
        torch.save([self.inputs, self.outputs], target)

    @classmethod
    def load(Cls, target: PathLike):
        data = torch.load(target)
        return Cls(*data)

    @classmethod
    def from_generation(Cls, size: int, tqdm_disable=True):
        inputs, outputs, sim_seeds = generate_batch(size, tqdm_disable)
        return Cls(inputs, outputs, sim_seeds)

def entry_from_node(node: Node, tqdm_disable=True):
    tensor = torch.zeros(MAX_ROADS * 2)
    for num in tqdm(range(MAX_ROADS), disable=tqdm_disable):
        if num >= 5:
            break

        if num < len(node.road_in):
            tensor[num * 2] = node.road_in[num].ai_flow_count[1]
        if num < len(node.road_out):
            tensor[num * 2 + 1] = node.road_out[num].ai_flow_count[0]
    return tensor

def generate_batch(size: int, tqdm_disable=True) -> Tuple[torch.TensorType, torch.TensorType, torch.TensorType]:
    sim_seed = int(time.time())
    map_file = "src/maps/build/GUI/Star/map.csv"
    paths_file = "src/maps/build/GUI/Star/paths.csv"
    central_node = 1

    batch = []
    sim_seeds = []

    for _ in tqdm(range(size), disable=tqdm_disable):
        random.seed(sim_seed)
        simulation = Simulation(map_file=map_file, paths_file=paths_file, nb_movables=15)
        simulation.run(sim_duration=GENERATION_SEGMENT_DUARTION)
        sim_seeds.append(sim_seed)
        batch.append(entry_from_node(simulation.nodes[central_node]))
        sim_seed += 1

    return torch.stack(batch), torch.tensor([1]), torch.tensor(sim_seeds)
