from typing import List, Tuple
from common.constants import *
from game.game import GameModel

def display(game_model: GameModel):
  for y, line in enumerate(game_model.grid):
    for x, _cell in enumerate(line):
      if game_model.is_player(x, y):
        print(G_PLAYER, end="")
      elif game_model.is_goal(x, y):
        print(G_GOAL, end="")
      elif game_model.is_tree(x, y):
        print(G_TREE, end="")
      else:
        print(G_EMPTY, end="")
    print()

def display_mark(game_model: GameModel, path: List[Tuple[int, int]]):
  path = set(path)
  for y, line in enumerate(game_model.grid):
    for x, _cell in enumerate(line):
      if game_model.is_player(x, y):
        print(G_PLAYER, end="")
      elif game_model.is_goal(x, y):
        print(G_GOAL, end="")
      elif game_model.is_tree(x, y):
        print(G_TREE, end="")
      elif (x, y) in path:
        print(G_MARK, end="")
      else:
        print(G_EMPTY, end="")
    print()

