from ast import Not
import torch

class Agent:
  def action(self, state : torch.Tensor):
    raise NotImplementedError
  def train(self, batch_size : int, epochs : int):
    raise NotImplementedError
  def net_action(self, obs : torch.Tensor):
    raise NotImplementedError
  def save_net(self):
    raise NotImplementedError