#!/usr/bin/python3
from torch.utils.data import DataLoader
from os import PathLike
import game.game as game
import game.display as display
from common.constants import *
from neural.pathfinding import PathFindingModel, training_loop, INPUT_CHANNELS
from neural.dataset import PathfindingDataset, generate_batches
import argparse

def generate_dataset(target: PathLike, size: int):
  PathfindingDataset.from_generation(size, tqdm_disable=False).save(target)

def generate_model(target: PathLike):
  PathFindingModel().save(target)

def test_game():
  game_model = game.GameModel.random_grid()
  display.display(game_model)
  path = game_model.get_optimal_path()
  if path is None:
    print("No path")
  else:
    print(f"=== Next step: {path[1]} ===")
    print(game_model.torchify_output())
    display.display_mark(game_model, path)

def train(dataset_target: PathLike, model_target: PathLike, num_epochs):
  dataset = PathfindingDataset.load(dataset_target)
  model = PathFindingModel.load(model_target)
  dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
  training_loop(model, dataloader, num_epochs)
  model.save(model_target)

def main():
  parser = argparse.ArgumentParser(description='Run specific tasks with their respective arguments.')
  subparsers = parser.add_subparsers(dest='task', required=False, help='sub-command help')

  parser_dataset = subparsers.add_parser('gen-dataset')
  parser_dataset.add_argument('target', type=str)
  parser_dataset.add_argument('size', type=int)

  parser_model = subparsers.add_parser('gen-model')
  parser_model.add_argument('target', type=str)

  parser_train = subparsers.add_parser('train')
  parser_train.add_argument('dataset_target', type=str)
  parser_train.add_argument('model_target', type=str)
  parser_train.add_argument('epochs', type=int, default=10)

  subparsers.add_parser('test-game')
  subparsers.add_parser('test-complete')

  # Parse the arguments
  args = parser.parse_args()

  # Check which task to run and call the respective function with the parsed arguments
  if args.task == 'gen-dataset':
    generate_dataset(args.target, args.size)
  elif args.task == 'gen-model':
    generate_model(args.target)
  elif args.task == 'test-game':
    test_game()
  elif args.task == "test-complete":
    test_complete()
  elif args.task == "train":
    train(args.dataset_target, args.model_target, args.epochs)
  else:
    train("./data/dataset.pf", "./data/model.pf", 10)

def test_complete():
  game_model = game.GameModel.random_grid()
  display.display(game_model)
  path = game_model.get_optimal_path()
  if path is None:
    print("No path")
  else:
    print(f"=== Next step: {path[1]} ===")
    print(game_model.torchify_output())
    display.display_mark(game_model, path)

  # Input
  print("NEURAL NETWORK INPUT")
  dnn_input = game_model.torchify_input()
  print(dnn_input.size())
  print(dnn_input.shape)
  print()

  # Neural network
  print("NEURAL NETWORK EXECUTION")

  # Instantiate the model
  model = PathFindingModel()

  # Forward pass through the model to get the prediction
  input_batches = dnn_input.view(1, INPUT_CHANNELS, W, H)
  print(f"input shape {input_batches.shape}")
  move = model(input_batches)
  print(f"res {move}")
  print()

  # Generate data
  print("GENERATING BATCHES")
  dataset = PathfindingDataset.from_generation(10000)

  # Training
  print("TRAINING")
  dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
  training_loop(model, dataloader, 10000)


if __name__ == "__main__":
    main()
