from os import PathLike
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from common.constants import *

INPUT_CHANNELS = 3  # as we have 3 channels: grid, player position, goal position
HIDDEN_DIM = 64  # example hidden dimension, can be tuned

MOVED_LEFT = 0
MOVED_RIGHT = 1
MOVED_UP = 2
MOVED_DOWN = 3
STAYED = 4

class PathFindingModel(nn.Module):
    def __init__(self):
        super(PathFindingModel, self).__init__()
        self.conv1 = nn.Conv2d(INPUT_CHANNELS, HIDDEN_DIM, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(HIDDEN_DIM, HIDDEN_DIM * 2, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(HIDDEN_DIM * 2 * W * H, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1) # flatten
        features = F.relu(self.fc1(x))
        next_move = self.fc2(features)
        return next_move

    def save(self, target: PathLike):
        torch.save(self.state_dict(), target)

    @classmethod
    def load(Cls, target: PathLike, device="cpu"):
        state_dict = torch.load(target, map_location=device)
        model = Cls()
        model.load_state_dict(state_dict)
        return model


def training_loop(model: PathFindingModel, dataloader: DataLoader, num_epochs: int):
    log_interval = 100
    criterion_next_move = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(num_epochs):
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            optimizer.zero_grad()

            # Forward pass ([batch, 3, W, H])
            result = model(inputs)

            # Calculate loss for next move prediction
            loss = criterion_next_move(result, targets)

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()

            if batch_idx % log_interval == 0:
                print(f"Epoch: {epoch+1:4} / {num_epochs:4}, Batch: {batch_idx+1:8d} / {len(dataloader):8d}, Loss: {loss.item():.4f}")


