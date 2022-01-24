from env.continuous_cartpole import ContinuousCartPoleEnv
from ddpg.ddpg import DDPG, Model
import torch

env   = ContinuousCartPoleEnv()
obj   = DDPG(
  actor = Model, actorkwargs = {'net_width' : 64}, 
  critic = Model, critickwargs = {'net_width' : 64}, 
  obs_size = 4, act_size = 1, mem_size = 10000, batch_size = 64, 
  num_workers = 0, learning_rate = 1e-6, eps = 0.3, eps_decay = 0.99999
)

train_epoch   = 1000000
test_freq     = 100
save_freq     = 100
targ_freq     = 1
print_freq    = 100
test_count    = 100
duration_arr  = []
for i in range(train_epoch):
  done          = False
  steps         = 0
  obs           = env.reset()
  action_loss   = 0
  critic_loss   = 0
  duration      = 0
  reward        = 0
  while not done:
    action  = obj.action(
      torch.tensor(obs, dtype = torch.float32)
    )
    nobs, _, done, info = env.step(action)
    rew   = 0.05 - nobs[2] ** 2
    obj.save_state(
      state   = torch.tensor(obs, dtype = torch.float32), 
      next_state = torch.tensor(nobs, dtype = torch.float32), 
      action  = torch.tensor(action, dtype = torch.float32), 
      reward  = rew,
      terminal = done 
    )
    duration += 1
    reward += rew
    obs   = nobs
    if i != 0:
      loss  = obj.train(epochs = 5, gamma = 0.99)
      action_loss += loss['actor']
      critic_loss += loss['critic']
      obj.soft_update_target(coeff = 0.3)
  obj.decay_epsilon()
  duration_arr.append(duration)
  if i % print_freq == 0:
    print("==============================================")
    print(f"Iteration {i}")
    print(f"Actor loss  : {round(action_loss, 3)}")
    print(f'Critic_loss : {round(critic_loss, 3)}')
    print(f'Duration    : {round(sum(duration_arr) / print_freq, 3)}')
    print(f'Reward      : {round(reward, 3)}')
    print(f'Explore rate: {round(obj._Eps_, 3)}')
    print('==============================================')
    action_loss   = 0
    critic_loss   = 0
    duration_arr  = []

  if i % test_freq == 0:
    test_duration_array   = []
    for j in range(test_count):
      test_dur  = 0
      obs       = env.reset()
      done      = False
      while not done:
        if j % 50 == 0:
          env.render()
        action  = obj.test_action(
          torch.tensor(obs, dtype = torch.float32)
        )
        obs, _, done, info = env.step(action)
        test_dur += 1
      test_duration_array.append(test_dur)
    print('Average time   : %s' % (sum(test_duration_array) / test_count))


  

