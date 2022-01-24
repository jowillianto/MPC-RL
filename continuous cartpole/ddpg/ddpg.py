import torch
import numpy as np

class Model(torch.nn.Module):
  def __init__(self, 
    net_width : int = 32, input_size : int = 4, output_size : int = 1
  ):
    super().__init__()
    self.norm_layer   = torch.nn.BatchNorm1d(input_size)
    self.first_layer  = torch.nn.Linear(input_size, net_width)
    self.first_act    = torch.nn.LeakyReLU()
    self.inter_layer  = torch.nn.Linear(net_width, net_width)
    self.inter_act    = torch.nn.LeakyReLU()
    self.final_layer  = torch.nn.Linear(net_width, output_size)
  def forward(self, x):
    x   = self.norm_layer(x)
    x   = self.first_layer(x)
    x   = self.first_act(x)
    x   = self.inter_layer(x)
    x   = self.inter_act(x)
    x   = self.final_layer(x)
    return x

class Buffer:
  def __init__(self, 
    mem_size : int = 10000, obs_size : int = 0, act_size : int = 10
  ):
    self._MemSize_  = mem_size
    self._ObsSize_  = obs_size
    self._ActSize_  = act_size
    
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    self._Device_   = device
    self._State_    = torch.zeros([mem_size, obs_size], dtype = torch.float32)
    self._NState_   = torch.zeros([mem_size, obs_size], dtype = torch.float32)
    self._Reward_   = torch.zeros([mem_size], dtype = torch.float32)
    self._Action_   = torch.zeros([mem_size, act_size], dtype = torch.int64)
    self._Terminal_ = torch.zeros([mem_size], dtype = torch.bool)

    self._Pointer_  = 0
    self._Passed_   = False

  def reset(self):
    mem_size        = self._MemSize_
    obs_size        = self._ObsSize_
    act_size        = self._ActSize_
    self._State_    = torch.zeros([mem_size, obs_size], dtype = torch.float32)
    self._NState_   = torch.zeros([mem_size, obs_size], dtype = torch.float32)
    self._Reward_   = torch.zeros([mem_size], dtype = torch.float32)
    self._Action_   = torch.zeros([mem_size, act_size], dtype = torch.int64)
    self._Terminal_ = torch.zeros([mem_size], dtype = torch.bool)
    self._Pointer_  = 0
    self._Passed_   = False

  def save_state(self, 
    state :torch.Tensor, next_state : torch.Tensor, action : torch.Tensor, reward : float, terminal : bool
  ):
    self._State_[self._Pointer_]    = state
    self._NState_[self._Pointer_]   = next_state

    self._Reward_[self._Pointer_]   = reward
    self._Action_[self._Pointer_]   = action
    self._Terminal_[self._Pointer_] = terminal

    self._Pointer_  += 1
    if self._Pointer_ == self._MemSize_:
      self._Pointer_  = 0
      if not self._Passed_:
        self._Passed_   = True
      
  def __getitem__(self, key : int):
    pointer   = abs(key % self._MemSize_)
    state     = self._State_[pointer]
    next_state= self._NState_[pointer]
    action    = self._Action_[pointer]
    reward    = self._Reward_[pointer]
    terminal  = self._Terminal_[pointer]
    return state, next_state, action, reward, terminal
  
  def __len__(self):
    if not self._Passed_:
      return self._Pointer_
    else:
      return self._MemSize_
  
  def sample(self, batch_size : int = 32):
    max_size  = min(self.__len__(), batch_size)
    batch_idx = torch.tensor(
      np.random.choice(np.array(range(max_size), dtype = np.int64), size = min(batch_size, max_size),replace = False), 
    dtype = torch.int64)
    state     = self._State_[batch_idx]
    next_state= self._NState_[batch_idx]
    reward    = self._Reward_[batch_idx]
    action    = self._Action_[batch_idx]
    terminal  = self._Terminal_[batch_idx]
    
    return state, next_state, action, reward, terminal, max_size

class DDPG:
  def __init__(
    self, actor : torch.nn.Module, actorkwargs : dict, 
    critic : torch.nn.Module, critickwargs : dict,
    obs_size : int = 4, act_size : int = 1, mem_size : int = 10000, 
    batch_size : int = 64, num_workers : int = 20, learning_rate : float = 1e-4, 
    eps : float = 0.9, eps_decay : float = 0.999
  ):
    self._MemoryBuffer_   = Buffer(
      obs_size = obs_size, act_size = act_size, mem_size = mem_size
    )

    device              = 'cuda' if torch.cuda.is_available() else 'cpu'
    self._Actor_        = actor(input_size = obs_size, output_size = act_size, **actorkwargs).to(device)
    self._Critic_       = critic(input_size = obs_size + act_size, output_size = 1, **critickwargs).to(device)
    self._TargetActor_  = actor(input_size = obs_size, output_size = act_size, **actorkwargs).to(device)
    self._TargetCritic_ = critic(input_size = obs_size + act_size, output_size = 1, **critickwargs).to(device)

    self._BatchSize_    = batch_size
    self._NumWorkers_   = num_workers

    self._LossFunc_     = torch.nn.MSELoss().to(device)
    self._ActorOpt_     = torch.optim.Adam(self._Actor_.parameters(), lr = learning_rate)
    self._CriticOpt_    = torch.optim.Adam(self._Critic_.parameters(), lr = learning_rate)

    self._Device_       = device

    self._Eps_          = eps
    self._EpsDecay_     = eps_decay

    self.update_target()
  def save_net(self, path):
    torch.save(self._Actor_.state_dict(), f'{path}/actor.pt')
    torch.save(self._Critic_.state_dict(), f'{path}/critic.pt')
  
  def load_net(self, path):
    self._Actor_.load_state_dict(torch.load(f'{path}/actor.pt'))
    self._Critic_.load_state_dict(torch.load(f'{path}/critic.pt'))
  
  def update_target(self):
    self._TargetCritic_.load_state_dict(self._Critic_.state_dict())
    self._TargetActor_.load_state_dict(self._Actor_.state_dict())

  def _soft_update_target(self, target : torch.nn.Module, pol : torch.nn.Module, coeff : float = 0.1):
    for target_param, param in zip(target.parameters(), pol.parameters()):
      target_param.data.copy_(
        target_param.data * (1 - coeff) + param.data * coeff
      )
  
  def soft_update_target(self, coeff : float = 0.1):
    self._soft_update_target(self._TargetCritic_, self._Critic_, coeff)
    self._soft_update_target(self._TargetActor_, self._Actor_, coeff)

  def save_state(self, state : torch.Tensor, next_state : torch.Tensor, action : torch.Tensor, reward : float, terminal : bool):
    self._MemoryBuffer_.save_state(
      state = state, next_state = next_state, action = action, reward = reward, terminal = terminal
    )
  
  def action(self, state : torch.Tensor):
    eps     = torch.rand(1)[0]
    if eps > self._Eps_: 
      state   = state.to(self._Device_)
      with torch.no_grad():
        self._Actor_.eval()
        action  = self._Actor_(state.unsqueeze(0))
        return action.item()
    else:
      return torch.distributions.uniform.Uniform(-10, 10).sample((1, 1)).item()

  def test_action(self, state : torch.Tensor):
    state   = state.to(self._Device_)
    self._Actor_.eval()
    with torch.no_grad():
      action  = self._Actor_(state.unsqueeze(0))
      return action.item()

  def decay_epsilon(self):
    self._Eps_ *= self._EpsDecay_

  def train(self, epochs : int = 3, gamma : float = 0.99):
    running_loss  = {
      'actor'   : 0, 
      'critic'  : 0
    }
    batch_size    = self._BatchSize_
    self._Actor_.train()
    self._Critic_.train()
    for epoch in range(epochs):
      self._ActorOpt_.zero_grad()
      self._CriticOpt_.zero_grad()
      state, next_state, action, reward, terminal, batch_size = self._MemoryBuffer_.sample(batch_size)
      device  = self._Device_
      state           = state.to(device)
      next_state      = next_state.to(device)
      action          = action.to(device)
      reward          = torch.permute(reward.to(device).unsqueeze(0), (1, 0))
      terminal        = terminal.to(device)
      current_state_value   = self._Critic_(torch.cat((state, action), dim = 1))
      with torch.no_grad():
        next_action           = self._TargetActor_(next_state)
      next_state_value        = self._TargetCritic_(torch.cat((next_state, next_action), dim = 1))

      next_state_value[terminal]  = 0.0
      target_state_value    = reward  + gamma * next_state_value
      #this is to update the Q network
      loss                  = self._LossFunc_(current_state_value, target_state_value)
      loss.backward()
      self._CriticOpt_.step()

      running_loss['critic']  += loss.item()
      #This is to update the actor network
      state_action  = self._Actor_(state)
      action_loss   = -self._Critic_(torch.cat((state, state_action), dim = 1)).mean()
      action_loss.backward()
      self._ActorOpt_.step()

      running_loss['actor']   += action_loss.item()
    return running_loss
      




      