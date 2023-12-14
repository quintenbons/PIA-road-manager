from os import PathLike
import torch
from torch.utils.data import DataLoader
from ai.dataset import NodeDataset
from ai.model import CrossRoadModel
import torch.nn as nn

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

def train(dataset_target: PathLike, model_target: PathLike, num_epochs):
    dataset = NodeDataset.load(dataset_target)
    model = CrossRoadModel.load(model_target)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    training_loop(model, dataloader, num_epochs)
    model.save(model_target)
