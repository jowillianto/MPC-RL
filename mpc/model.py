import numpy as np

class Model:
  def __init__(self, obs_size : int, act_size : int, prediction_horizon : int, control_step : int):
    self.obs_mem      = np.zeros((prediction_horizon, obs_size), dtype = np.float32)
    self.act_mem      = np.zeros((prediction_horizon, act_size), dtype = np.float32)
    self.cur_obs      = np.zeros(obs_size, dtype = np.float32)
    self.cur_act      = np.zeros(act_size, dtype = np.float32)
    self._PredHor_    = prediction_horizon    
    
    self.history      = {
      'obs'   : np.zeros((control_step, obs_size), dtype = np.float32),
      'r_obs' : np.zeros((control_step, obs_size), dtype = np.float32)
    }
    self.init_model()    
    pass
  def init_model(self):
    raise NotImplementedError

  def save_state(self, state : np.ndarray):
    self.cur_obs      = state

  def action(self):
    return self.cur_act

  def reset(self):
    self.obs_mem      = np.zeros(*self.obs_mem.shape, dtype = np.float32)
    self.act_mem      = np.zeros(*self.act_mem.shape, dtype = np.float32)

  def predict(self):
    raise NotImplementedError

  def action(self):
    return self.cur_act
  
  # Helper Function
  def cur_state(self):
    return self.cur_obs

  # Plot tools
  def save_data(self, step : int = 0):
    self.history['obs'][step]   = self.obs_mem[0]
    self.history['r_obs'][step] = self.cur_obs.flatten()
  
  def reset_history(self):
    self.history      = {
      'obs'   : np.zeros(self.history['obs'].shape, dtype = np.float32),
      'r_obs' : np.zeros(self.history['r_obs'].shape, dtype = np.float32)
    }