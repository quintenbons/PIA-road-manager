from os import PathLike
import torch
import torch.nn as nn
import torch.nn.functional as F
from ai.model_constants import *

class CrossRoadModel(nn.Module):
    inp: nn.Linear
    hidden_layers: nn.ModuleList
    out: nn.Linear

    input_size: int = INPUT_DIM
    output_size: int = OUTPUT_DIM

    def __init__(self, nb_strats=OUTPUT_DIM):
        super(CrossRoadModel, self).__init__()
        self.inp = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        self.hidden_layers = nn.ModuleList([nn.Linear(HIDDEN_DIM, HIDDEN_DIM) for _ in range(HIDDEN_AMOUNT)])
        self.out = nn.Linear(HIDDEN_DIM, nb_strats)

        self.input_size = INPUT_DIM
        self.output_size = nb_strats

    def forward(self, x):
        x = F.relu(self.inp(x))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        x = F.relu(self.out(x))
        return x

    def save(self, target: PathLike):
        if isinstance(self, nn.DataParallel):
            state_dict = self.module.state_dict()
        else:
            state_dict = self.state_dict()
        dimensions = (self.input_size, self.output_size)
        torch.save([dimensions, state_dict], target)

    @classmethod
    def load(Cls, target: PathLike, device="cpu"):
        dimensions, state_dict = torch.load(target, map_location=device)
        input_size, output_size = dimensions
        model = Cls(output_size)
        model.load_state_dict(state_dict)
        return model
