from os import PathLike
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ai.model_constants import *

class CrossRoadModel(nn.Module):
    def __init__(self, nb_strats=OUTPUT_DIM):
        super(CrossRoadModel, self).__init__()
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.fc3 = nn.Linear(HIDDEN_DIM, nb_strats)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

    def save(self, target: PathLike):
        if isinstance(self, nn.DataParallel):
            state_dict = self.module.state_dict()
        else:
            state_dict = self.state_dict()
        torch.save(state_dict, target)


    @classmethod
    def load(Cls, target: PathLike, device="cpu"):
        state_dict = torch.load(target, map_location=device)
        model = Cls()
        model.load_state_dict(state_dict)
        return model
