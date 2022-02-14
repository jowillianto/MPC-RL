from base.buffer import Buffer
from base.agent import Agent

import torch

class PPOAgent(Agent):
  def __init__(self, 
    mem_size : int, obs_size : int, act_size : int, eps : float, gamma : float, tau : float, 
    actor : torch.nn.Module, actorkwargs : dict, critic :torch.nn.Module, critickwargs : dict, 
    clip_factor : float = 0.2
  ):
    self._buffer  = Buffer(
      mem_size = mem_size, obs_size = obs_size, act_size = act_size, act_dtype = torch.float32
    )
    