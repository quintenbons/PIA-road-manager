from os import PathLike
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ai.model_constants import *

class CrossRoadModel(nn.Module):
    def __init__(self):
        super(CrossRoadModel, self).__init__()
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.fc3 = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

    def save(self, target: PathLike):
        torch.save(self.state_dict(), target)

    @classmethod
    def load(Cls, target: PathLike, device="cpu"):
        state_dict = torch.load(target, map_location=device)
        model = Cls()
        model.load_state_dict(state_dict)
        return model

def training_loop(model: CrossRoadModel, dataloader: DataLoader, num_epochs: int):
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


