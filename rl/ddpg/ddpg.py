import torch
import sys
sys.path.append('..')
from ..memory.buffer import Buffer

def soft_update_target(target : torch.nn.Module, pol : torch.nn.Module, coeff : float = 0.1):
  for target_param, param in zip(target.parameters(), pol.parameters()):
    target_param.data.copy_(
      target_param.data * (1 - coeff) + param.data * coeff
    )

class DDPGAgent:
  def __init__(self, 
    mem_size : int, obs_size : int, act_size : int, 
    tau : float, eps : float, gamma : float, actor_lr : float, critic_lr : float, eps_decay : float,
    actor : torch.nn.Module, actorkwargs : dict, critic : torch.nn.Module, critickwargs : dict
  ):
    self._buffer  = Buffer(
      mem_size = mem_size, obs_size = obs_size, act_size = act_size, act_dtype = torch.float32
    )
    self._tau     = tau
    self._gamma   = gamma
    self._eps     = eps
    self._device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self._actor   = actor(**actorkwargs).to(self._device)
    self._critic  = critic(**critickwargs).to(self._device)
    self._tactor  = actor(**actorkwargs).to(self._device)
    self._tcritic = critic(**critickwargs).to(self._device)
    self._aOptim  = torch.optim(self._actor.parameters(), lr = actor_lr)
    self._cOptim  = torch.optim(self._critic.parameters(), lr = critic_lr)
    self._cLoss   = torch.nn.MSELoss()
    self._epsDec  = eps_decay
  
  def save_state(self, obs : torch.Tensor, n_obs : torch.Tensor, action : torch.Tensor, reward : float, terminal : bool):
    Buffer.save_state(
      obs = obs, next_obs = n_obs, action = action, reward = reward, terminal = terminal 
    )
  
  def soft_update(self):
    soft_update_target(self._tactor, self._actor, coeff = self._tau)
    soft_update_target(self._tcritic, self._critic, coeff = self._tau)

  def train(self, batch_size : int):
    self._actor.train()
    self._critic.train()

    # Retrieve sample from buffer
    batch_size, obs, next_obs, action, reward, terminal = self._buffer.sample(batch_size)

    obs       = obs.to(self._device)
    next_obs  = next_obs.to(self._device)
    action    = action.to(self._device)
    reward    = torch.permute(reward.to(self._device).unsqueeze(0), (1, 0))
    terminal  = reward.to(self._device)
    
    # Train Critic Function 
    self._aOptim.zero_grad()
    self._cOptim.zero_grad()

    # Update Critic
    cur_state_value   = self._critic(torch.cat([obs, action], dim = 1))
    with torch.no_grad():
      next_action     = self._tactor(next_obs)
    next_state_value  = self._tcritic(torch.cat([next_obs, next_action], dim = 1))
    next_state_value[terminal]  = 0.0
    target_state_value= reward + self._gamma * next_state_value
    state_value_loss  = self._cLoss(cur_state_value, target_state_value)
    state_value_loss.backward()
    self._cOptim.step()

    # Update Actor
    net_action  = self._actor(obs)
    actor_loss  = - self._critic(torch.cat([obs, net_action], dim = 1)).mean()
    actor_loss.backward()
    self._aOptim.step()

    self.soft_update()
    return state_value_loss.item()

  def action(self, obs : torch.Tensor):
    # According to the paper, we add some noise to the data, let's use a gaussian distribution
    obs   = obs.to(self._device)
    avg   = self._actor(obs)[0]
    return torch.normal(avg, std = self._eps)
  
  def decay_epsilon(self):
    self._eps *= self._epsDec

  def net_action(self, obs : torch.Tensor):
    self._actor.eval()
    action  = self._action(obs)
    return action
  
  def save_net(self, path):
    torch.save(self._actor.state_dict(), f'{path}/actor.pt')
    torch.save(self._critic.state_dict(), f'{path}/critic.pt')


    
