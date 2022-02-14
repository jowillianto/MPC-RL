import torch
import sys
sys.path.append('..')
from rl.ddpg import DDPGAgent

class Actor(torch.nn.Module):
  def __init__(self, 
    net_width : int = 32, input_size : int = 4, output_size : int = 1
  ):
    super().__init__()
    # self.norm_layer   = torch.nn.BatchNorm1d(input_size)
    self.first_layer  = torch.nn.Linear(input_size, net_width)
    self.first_act    = torch.nn.LeakyReLU()
    self.inter_layer  = torch.nn.Linear(net_width, net_width)
    self.inter_act    = torch.nn.LeakyReLU()
    self.final_layer  = torch.nn.Linear(net_width, output_size)
  def forward(self, x):
    #x   = self.norm_layer(x)
    x   = self.first_layer(x)
    x   = self.first_act(x)
    x   = self.inter_layer(x)
    x   = self.inter_act(x)
    x   = self.final_layer(x)
    return x

class Critic(torch.nn.Module):
  def __init__(self, 
    net_width : int = 32, obs_size : int = 4, act_size : int = 1, output_size : int = 1
  ):
    super().__init__()
    self.norm_layer   = torch.nn.BatchNorm1d(obs_size)
    self.first_layer  = torch.nn.Linear(obs_size + act_size, net_width)
    self.first_act    = torch.nn.LeakyReLU()
    self.inter_layer  = torch.nn.Linear(net_width, net_width)
    self.inter_act    = torch.nn.LeakyReLU()
    self.final_layer  = torch.nn.Linear(net_width, output_size)
    self.obs_size     = obs_size
  def forward(self, obs, action):
    #x   = self.norm_layer(obs)
    x   = torch.cat((obs, action), dim = -1)
    x   = self.first_layer(x)
    x   = self.first_act(x)
    x   = self.inter_layer(x)
    x   = self.inter_act(x)
    x   = self.final_layer(x)
    return x


class ContinuousCartpole(DDPGAgent):
  def __init__(
    self, tau : float, eps : float, gamma : float, actor_lr : float, critic_lr : float, eps_decay : float, mem_size : int
  ):
    actor   = Actor
    actorkwargs = {
      'net_width' : 32, 'input_size' : 4, 'output_size' : 1
    }
    critic  = Critic
    critickwargs = {
      'net_width' : 32, 'obs_size' : 4, 'output_size' : 1, 'act_size' : 1
    }
    super().__init__(
      tau = tau, eps = eps, gamma = gamma, actor_lr = actor_lr, critic_lr = critic_lr, eps_decay = eps_decay, 
      actor = actor, actorkwargs = actorkwargs, critic = critic, critickwargs = critickwargs, 
      mem_size = mem_size, obs_size = 4, act_size = 1
    )




      