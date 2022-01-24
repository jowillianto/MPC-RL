from gym.envs.classic_control.cartpole import CartPoleEnv

class ContinuousCartPoleEnv(CartPoleEnv):
  def __init__(self):
    super().__init__()
    self.steps  = 0

  def reset(self):
    self.steps = 0
    return super().reset()
    
  def step(self, action : float):
    self.force_mag  = action
    obs, rew, done, info = super().step(action = 1)
    self.steps += 1
    if self.steps > 200:
      return obs, rew , True, info
    else:
      return obs, rew, done, info
  