import torch 
import numpy as np

class Memory:
  def __init__(self, 
    mem_size : int = 10000, obs_size : int = 0, act_space : int = 10
  ):
    self._MemSize_  = mem_size
    self._ObsSize_  = obs_size
    self._ActSpce_  = act_space
    
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    self._Device_   = device
    self._State_    = torch.zeros([mem_size, obs_size], dtype = torch.float32)
    self._NState_   = torch.zeros([mem_size, obs_size], dtype = torch.float32)
    self._Reward_   = torch.zeros([mem_size], dtype = torch.float32)
    self._Action_   = torch.zeros([mem_size], dtype = torch.int64)
    self._Terminal_ = torch.zeros([mem_size], dtype = torch.bool)

    self._Pointer_  = 0
    self._Passed_   = False

  def reset(self):
    mem_size        = self._MemSize_
    obs_size        = self._ActSpce_
    self._State_    = torch.zeros([mem_size, obs_size], dtype = torch.float32)
    self._NState_   = torch.zeros([mem_size, obs_size], dtype = torch.float32)
    self._Reward_   = torch.zeros([mem_size], dtype = torch.float32)
    self._Action_   = torch.zeros([mem_size], dtype = torch.int64)
    self._Terminal_ = torch.zeros([mem_size], dtype = torch.bool)
    self._Pointer_  = 0
    self._Passed_   = False

  def save_state(self, 
    state : np.ndarray, next_state : np.ndarray, action : int, reward : float, terminal : bool
  ):
    self._State_[self._Pointer_]    = torch.tensor(state, dtype = torch.float32)
    self._NState_[self._Pointer_]   = torch.tensor(next_state, dtype = torch.float32)

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

class Agent:
  def __init__(self, 
    network : torch.nn.Module, mem_size : int = 10000, obs_size : int = 0, act_size : int = 0, learning_rate : float = 1e-4, discount_factor : float = 0.8, eps_start : float = 1.0,
    eps_red : int = 0.99, **kwargs
  ):
    self._Memory_ = Memory(
      mem_size = mem_size, obs_size = obs_size, act_space = act_size
    )
    device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self._Device_ = device
    self._Policy_ = network(**kwargs).to(device)
    self._Target_ = network(**kwargs).to(device)

    self._Gamma_  = discount_factor
    self._Eps_    = eps_start
    self._Mpl_    = eps_red
  
  def reset(self):
    self._Memory_.reset()
    
  def update_target(self):
    self._Target_.load_state_dict(self._Policy_.state_dict())

  def save_state(self, 
    state : np.ndarray, next_state : np.ndarray, action : int, reward : float, terminal : bool
  ):
    self._Memory_.save_state(
      state = state, next_state = next_state, action = action, reward = reward, terminal = terminal
    )

  def action(self, state : np.ndarray):
    rnd_number  = np.random.rand()
    if rnd_number < self._Eps_:
      return np.random.choice(np.array(range(self._Memory_._ActSpce_), dtype = np.int64))
    else:
      state   = torch.tensor(state, dtype = torch.float32).to(self._Device_)
      with torch.no_grad():
        action  = self._Policy_(state).argmax(dim = 1)
        return action.item()

  def net_action(self, state : np.ndarray):
    state   = torch.tensor(state, dtype = torch.float32).to(self._Device_)
    with torch.no_grad():
      action  = self._Policy_(state).argmax(dim = 1)
    return action.item()

  def decay_explore(self):
    self._Eps_ *= self._Mpl_

  def save_net(self, path : str):
    torch.save(self._Policy_.state_dict(), path)
  
  def load_net(self, path : str):
    self._Policy_.load_state_dict(torch.load(path))

  def train(self, batch_size : int = 64):
    state, next_state, action, reward, terminal, batch_size = self._Memory_.sample(batch_size)
    device  = self._Device_
    state           = state.to(device)
    next_state      = next_state.to(device)
    action          = action.to(device)
    reward          = reward.to(device)
    terminal        = terminal.to(device)
    batch_idx       = torch.arange(batch_size)

    self._Policy_._Optimizer_.zero_grad()
    state_value     = self._Policy_(state)[batch_idx, action]
    with torch.no_grad():
      next_state_value= self._Target_(next_state)
      next_state_value[terminal] = 0
    
    correct_state_value   = reward + self._Gamma_ * torch.max(next_state_value, dim = 1)[0]
    
    loss            = self._Policy_._LossFunc_(state_value, correct_state_value)
    loss.backward()
    self._Policy_._Optimizer_.step()
    return loss.item()
