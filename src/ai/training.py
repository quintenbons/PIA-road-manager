from os import PathLike
import os
import torch
from torch.utils.data import DataLoader
from ai.dataset import NodeDataset
from ai.model import CrossRoadModel
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

def training_loop(model: CrossRoadModel, dataloader: DataLoader, num_epochs: int):
    log_interval = 100
    criterion_next_move = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
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
        scheduler.step()

def train(dataset_target: PathLike, model_target: PathLike, num_epochs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Used device:", device)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        print(torch.cuda.get_device_name(0))
    dataset = NodeDataset.load(dataset_target, device)

    # Create model if it does not exist
    if not os.path.exists(model_target):
        output_shape = dataset.get_output_shape()
        print(f"====== Generating model of output shape {output_shape}:")
        model = CrossRoadModel(output_shape)
        model.save(model_target)

    model = CrossRoadModel.load(model_target, device)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=False)
    training_loop(model, dataloader, num_epochs)
    model.save(model_target)


