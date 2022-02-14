import torch
import numpy as np

class Buffer:
  def __init__(self, mem_size : int, obs_size : int, act_size : int, act_dtype : type):
    self._pointer   = 0
    self._passed    = False
    self._cur_obs   = torch.zeros((mem_size, obs_size), dtype = torch.float32)
    self._next_obs  = torch.zeros((mem_size, obs_size), dtype = torch.float32)
    self._action    = torch.zeros((mem_size, act_size), dtype = act_dtype)
    self._reward    = torch.zeros(mem_size, dtype = torch.float32)
    self._terminal  = torch.zeros(mem_size, dtype = torch.bool)
    self._mem_size  = mem_size

  def save_state(self, obs : torch.Tensor, next_obs : torch.Tensor, action : torch.Tensor, reward : float, terminal : bool):
    idx   = self._pointer
    self._cur_obs[idx]  = obs
    self._next_obs[idx] = next_obs
    self._action[idx]   = action
    self._reward[idx]   = reward
    self._terminal[idx] = terminal
    self._pointer += 1
    if self._pointer == self._mem_size:
      if not self._passed:
        self._passed  = True
      self._pointer   = 0
  
  def sample(self, batch_size : int = 64):
    max_index   = self._mem_size if self._passed else self._pointer
    max_size    = min(batch_size, max_index)
    batch_idx   = torch.tensor(np.random.choice(range(max_index), size = max_size, replace = False), dtype = torch.int64)
    next_obs    = self._next_obs[batch_idx]
    obs         = self._cur_obs[batch_idx]
    action      = self._action[batch_idx]
    reward      = self._reward[batch_idx]
    terminal    = self._terminal[batch_idx]
    return max_size, obs, next_obs, action, reward, terminal
  
  def reset(self):
    self._pointer   = 0
    self._passed    = False
    self._cur_obs   = torch.zeros(self._cur_obs.shape, dtype = torch.float32)
    self._next_obs  = torch.zeros(self._next_obs.shape, dtype = torch.float32)
    self._action    = torch.zeros(self._action.shape, dtype = self._action.dtype)
    self._reward    = torch.zeros(self._reward.shape, dtype = torch.float32)
    self._terminal  = torch.zeros(self._terminal.shape, dtype =torch.bool)