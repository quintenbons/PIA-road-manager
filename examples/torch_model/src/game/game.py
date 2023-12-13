from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from common.constants import *
import torch

dpos = [(-1, 0), (1, 0), (0, -1), (0, 1)]

@dataclass
class GameModel:
  grid: torch.TensorType
  player_pos: Tuple[int, int]
  goal_pos: Tuple[int, int]

  optimal_path: Optional[List[Tuple[int, int]]] = field(default=None, init=False)
  path_pos: Optional[int] = field(default=None, init=False)

  def is_player(self, x: int, y: int) -> bool:
    return self.player_pos == (x, y)

  def is_goal(self, x: int, y: int) -> bool:
    return self.goal_pos == (x, y)

  def is_tree(self, x: int, y: int):
    return self.grid[y][x] == TREE

  def _generate_optimal_path(self):
    queue = [(self.player_pos, [self.player_pos])]
    visited = set()
    while queue:
      (x, y), path = queue.pop(0)
      if (x, y) in visited:
        continue
      visited.add((x, y))
      if self.is_goal(x, y):
        self.optimal_path = path
        self.path_pos = 0
        return
      # Explore neighbors
      for dx, dy in dpos:
        next_x, next_y = x + dx, y + dy
        if 0 <= next_x < self.grid.shape[1] and 0 <= next_y < self.grid.shape[0]:
          if not self.is_tree(next_x, next_y) and (next_x, next_y) not in visited:
            new_path = path + [(next_x, next_y)]
            queue.append(((next_x, next_y), new_path))
      # If the loop ends with no path found
      self.optimal_path = None
      self.path_pos = -1

  def get_optimal_path(self) -> Optional[List[Tuple[int, int]]]:
    if self.path_pos == None:
      self._generate_optimal_path()

    return self.optimal_path

  def torchify_input(self) -> torch.Tensor:
    """
    This function generates a tensor representation of the game model suitable for
    input to a DNN. The output tensor has three channels:
      - Channel 0: The grid with obstacles
      - Channel 1: The player's position
      - Channel 2: The goal's position
    """
    # Create tensor with an additional dimension for channels
    channels = []

    # Channel 0: Grid
    channels.append(self.grid.unsqueeze(0))

    # Channel 1: Player's position
    player_channel = torch.zeros_like(self.grid)
    player_channel[self.player_pos[1], self.player_pos[0]] = 1
    channels.append(player_channel.unsqueeze(0))

    # Channel 2: Goal's position
    goal_channel = torch.zeros_like(self.grid)
    goal_channel[self.goal_pos[1], self.goal_pos[0]] = 1
    channels.append(goal_channel.unsqueeze(0))

    # Stack all channels to create a multi-channel tensor
    dnn_input_tensor = torch.cat(channels, dim=0)

    return dnn_input_tensor

  def torchify_output(self) -> torch.Tensor:
    out = torch.zeros(5)

    if self.get_optimal_path() is None:
      out[4] = 1
      return out

    optimal_path = self.get_optimal_path()
    diff = (optimal_path[1][0] - optimal_path[0][0], optimal_path[1][1] - optimal_path[0][1])
    idx = dpos.index(diff)
    out[idx] = 1
    return out

  @classmethod
  def random_grid(Cls):
    grid = torch.bernoulli(torch.full((W, H), 1/4))
    player_pos = (torch.randint(0, W, (1,)).item(), torch.randint(0, H, (1,)).item())
    goal_pos = (torch.randint(0, W, (1,)).item(), torch.randint(0, H, (1,)).item())
    new_model = Cls(grid, player_pos, goal_pos)

    while new_model.is_tree(*new_model.goal_pos):
      new_model.goal_pos = (torch.randint(0, W, (1,)).item(), torch.randint(0, H, (1,)).item())

    while new_model.is_goal(*new_model.player_pos) or new_model.is_tree(*new_model.player_pos):
      new_model.player_pos = (torch.randint(0, W, (1,)).item(), torch.randint(0, H, (1,)).item())

    return new_model
