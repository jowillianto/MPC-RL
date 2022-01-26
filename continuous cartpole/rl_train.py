from env.continuous_cartpole import ContinuousCartPoleEnv
from ddpg.ddpg import ContinuousCartpole
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime

env   = ContinuousCartPoleEnv()
obj   = ContinuousCartpole(
  mem_size  = 30000, gamma = 0.99, eps_decay = 0.9999, eps = 2.0, tau = 1e-3,
  actor_lr  = 1e-4, critic_lr = 1e-4
)

train_epoch   = 1000000
test_freq     = 10
save_freq     = 10
targ_freq     = 0
print_freq    = 1
test_count    = 100
grace_count   = 1000
duration_arr  = []

# For graphing purpose
reward_graph= np.zeros(train_epoch, dtype = np.float32)
value_loss  = np.zeros(train_epoch, dtype = np.float32)
reward_accum= 0
steps       = 0
value_accum = 0

# Create folders
if not os.path.isdir('./rl_model'):
  os.mkdir('./rl_model')

folder_name   = f'train_{datetime.datetime.now().isoformat()}'
os.mkdir(f'./rl_model/{folder_name}')
os.mkdir(f'./rl_model/{folder_name}/model')

for i in range(train_epoch):
  done          = False
  obs           = env.reset()
  action_loss   = 0
  critic_loss   = 0
  duration      = 0
  reward        = 0
  reward_accum  = 0
  while not done:
    action  = obj.action(
      torch.tensor(obs, dtype = torch.float32)
    )
    nobs, _, done, info = env.step(action)
    rew   = 0.05 - obs[2]**2
    obj.save_state(
      obs   = torch.tensor(obs, dtype = torch.float32), 
      n_obs = torch.tensor(nobs, dtype = torch.float32), 
      action  = torch.tensor(action, dtype = torch.float32), 
      reward  = rew,  
      terminal = done 
    )
    duration += 1
    reward += rew
    reward_accum += rew
    obs   = nobs
    steps += 1
    if steps > grace_count:
      loss  = obj.train(epochs = 30, batch_size = 64)
      critic_loss += loss
      value_accum += loss
  obj.decay_epsilon()
  reward_graph[i] = reward_accum
  value_loss[i]   = value_accum
  reward_accum  = 0
  value_accum   = 0
  duration_arr.append(duration)
  if i % print_freq == 0:
    print("==============================================")
    print(f"Iteration {i}")
    print(f'Critic_loss : {round(critic_loss, 3)}')
    print(f'Duration    : {round(sum(duration_arr) / print_freq, 3)}')
    print(f'Reward      : {round(reward, 3)}')
    print(f'Explore rate: {round(obj._eps, 3)}')
    print('==============================================')

    duration_arr  = []
  critic_loss   = 0

  if i % test_freq == 0:
    test_duration_array   = []
    for j in range(test_count):
      test_dur  = 0
      obs       = env.reset()
      done      = False
      while not done:
        if j % 50 == 0:
          env.render()
          pass
        action  = obj.net_action(
          torch.tensor(obs, dtype = torch.float32)
        )
        obs, _, done, info = env.step(action)
        test_dur += 1
      test_duration_array.append(test_dur)
    print('Average time   : %s' % (sum(test_duration_array) / test_count))
  if i % save_freq == 0:
    obj.save_net(
      path_actor = f'./rl_model/{folder_name}/model/{i}_actor.pt', 
      path_critic = f'./rl_model/{folder_name}/model/{i}_critic.pt'
    )

    if os.path.isfile(f'./rl_model/{folder_name}/value.png'):
      os.remove(f'./rl_model/{folder_name}/value.png')
    if os.path.isfile(f'./rl_model/{folder_name}/reward.png'):
      os.remove(f'./rl_model/{folder_name}/reward.png')

    plt.clf()
    plt.plot(value_loss[:i])
    plt.ylabel('Loss of value function')
    plt.title('Value Loss')
    plt.xlabel('Episodes')
    plt.savefig(f'./rl_model/{folder_name}/value.png')

    plt.clf()
    plt.xlabel('Reward')
    plt.title('Reward')
    plt.ylabel('Episodes')
    plt.plot(reward_graph[:i])
    plt.savefig(f'./rl_model/{folder_name}/reward.png')


  

