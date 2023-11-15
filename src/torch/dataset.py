from dataclasses import dataclass
from os import PathLike
from torch.utils.data import Dataset
import torch

@dataclass
class CrossRoad(Dataset):
    inputs: torch.TensorType
    outputs: torch.TensorType

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
        inputs, outputs = generate_batch(size, tqdm_disable=tqdm_disable)
        return Cls(inputs, outputs)

def generate_batch(size: int, tqdm_disable=True) -> torch.TensorType:
    """TODO"""
    pass
    # input_batch = []
    # output_batch = []
    # for _ in tqdm(range(size), disable=tqdm_disable):
    #     game_model = GameModel.random_grid()
    #     input_batch.append(game_model.torchify_input())
    #     output_batch.append(game_model.torchify_output())

    # return torch.stack(input_batch), torch.stack(output_batch)

def generate_batches(batch_number, batch_size) -> torch.TensorType:
    """TODO"""
    pass
    # batches = [generate_batch(batch_size) for _ in range(batch_number+1)]
    # input_batches, output_batches = zip(*batches)
    # return torch.stack(input_batches), torch.stack(output_batches)
