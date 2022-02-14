from base.buffer import Buffer
from base.agent import Agent
import torch
import numpy as np

class DQNAgent(Agent):
  def __init__(self,
    mem_size : int, obs_size : int, gamma : float, eps : float, lr : float, eps_decay : float, act_space : int,
    network : torch.nn.Module, networkwargs : dict, 
  ):
    self._device  = 'cuda' if torch.cuda.is_available else 'cpu'
    self._buffer  = Buffer(mem_size = mem_size, obs_size = obs_size, act_size = 1, act_dtype = torch.int64)
    self._policy  = network(**networkwargs).to(self._device)
    self._target  = network(**networkwargs).to(self._device)
    self._target.load_state_dict(self._policy.state_dict())
    self._act_chc = list(range(act_space))

    self._gamma   = gamma
    self._eps     = eps
    self._eps_dec = eps_decay
    self._optim   = torch.optim.Adam(params = self._policy.parameters(), lr = lr)
    self._loss    = torch.nn.MSELoss()
  
  def action(self, obs : torch.Tensor):
    eps = np.random.rand()
    if self._eps > eps:
      return np.random.choice(self._act_chc)
    else:
      obs = obs.to(self._device).unsqueeze(0)
      self._policy.eval()
      act = self._policy(obs)
      return act.item()

  def net_action(self, obs : torch.Tensor):
    obs   = obs.to(self._device).unsqueeze(0)
    self._policy.eval()
    act   = self._policy(obs)
    return act.item()

  def train(self, epoch : int, batch_size : int):
    loss_accumulator  = 0
    for i in range(epoch):
      self._optim.zero_grad()
      batch_size, obs, next_obs, action, reward, terminal = self._buffer.sample(batch_size)
      arange        = torch.arange(0, batch_size)
      obs_val       = self._policy(obs)[action, arange]
      with torch.no_grad():
        next_obs_val  = self._target(next_obs)
      next_obs_val[terminal] = 0
      targ_obs_val  = reward + self._gamma * torch.max(next_obs_val, dim = -1)[0]
      val_delta     = self._loss(obs_val, targ_obs_val)
      val_delta.backward()
      loss_accumulator += val_delta.item()
      self._optim.step()
    return loss_accumulator
  
  def update_target(self):
    self._target.load_state_dict(self._policy.state_dict())
  
