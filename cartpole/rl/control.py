import sys
sys.path.append('..')

from qlearn.torchq import Agent
import torch

class Network(torch.nn.Module):
  def __init__(self, learning_rate : float = 1e-4):
    super(Network, self).__init__()
    self._LossFunc_   = torch.nn.MSELoss()
    self._FLayer_     = torch.nn.Linear(4, 32)
    self._ILayer_     = torch.nn.Linear(32, 32)
    self._LLayer_     = torch.nn.Linear(32, 2)
    self._ALayer_     = torch.nn.LeakyReLU()
    self._Optimizer_  = torch.optim.Adam(self.parameters(), lr = learning_rate)
  
  def forward(self, x):
    x   = self._ALayer_(self._FLayer_(x))
    x   = self._ALayer_(self._ILayer_(x))
    x   = self._LLayer_(x)
    return x

class CartPole(Agent):
  def __init__(self, mem_size : int, discount_factor : float, eps_start : float, eps_mlp : float, learning_rate : float):
    super().__init__(
      mem_size = mem_size, obs_size = 4, act_size = 2, learning_rate = learning_rate, discount_factor = discount_factor, eps_start = eps_start, 
      eps_red = eps_mlp, network = Network
    )
