from env.continuous_cartpole import ContinuousCartPoleEnv
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt
import os
import torch

env   = ContinuousCartPoleEnv()
obs   = env.reset()

import json
cfg_file  = open('./config.json')
config    = json.load(cfg_file)

render    = config['render']
render_n  = config['render_every']
eval_n    = config['test_n']
use_sleep = config['use_sleep']
eval_n    = 1

steps     = 0
time_arr  = []

suc_b     = 0
ang_b     = 0 
bou_b     = 0

from mpc_rl.control_refac import RLMPC, HighLevelMPC

obj       = RLMPC(pred_horizon = 1, control_step = 2000)
agent     = HighLevelMPC(tau = 1e-3, eps = 2, gamma = 0.999, actor_lr = 1e-8, critic_lr = 1e-8, eps_decay = 0.99, mem_size = 30000)
action_arr= np.zeros(2000, dtype = np.float32)
theta_arr = np.zeros((2000, 4), dtype = np.float32)
grace_pr  = 1000

if not os.path.isdir('./mpc_graph'):
  os.mkdir('./mpc_graph')

episode   = 1000000

# Make graphs and save models
rewards   = np.zeros(episode, dtype = np.float32)
losses    = np.zeros(episode, dtype = np.float32)
if not os.path.isdir('./rl_model'):
  os.mkdir('./rl_model')
folder_name   = f'train_{datetime.datetime.now().isoformat()}'
os.mkdir(f'./rl_model/{folder_name}')
os.mkdir(f'./rl_model/{folder_name}/model')

# Train Loop
for i in range(episode):
  done          = False
  steps         = 0
  obs           = env.reset()
  loss          = 0
  tot_rew       = 0
  # Get MPC Params
  params        = [0, 0, 0, 0]
  # Sample Episode
  while not done:
    # Save the state
    obj.save_state(np.array(obs, dtype = np.float32))
    # Get Params from rl
    params      = agent.action(torch.tensor(obs, dtype = torch.float32))
    # Predict Trajectory
    obj.predict(params.numpy().tolist())
    # Take an action
    action      = obj.action()[0]
    # Look at consequences of action
    next_obs, rew, done, info   = env.step(action)

    # Calculate reward from trajectory
    reward      = 0
    for nobs in obj._obs_mem:
      reward  += 0.15 - nobs[2]**2 - 1e-3 * nobs[1]**2 - 1e-2 * nobs[0]**2
    reward /= 1
    # Take state as input
    agent.save_state(
      obs   = torch.tensor(obs, dtype = torch.float32), 
      n_obs = torch.tensor(obs, dtype = torch.float32),
      action= params,
      reward= reward, terminal = True if abs(next_obs[2]) > 0.209 or abs(next_obs[0]) > 2.4 else False
    )
    obs         = next_obs
    tot_rew    += reward
    # Increment steps
    steps += 1
    if not grace_pr:
      loss  += agent.train(batch_size = 64, epochs = 20)
    if grace_pr:
      grace_pr-=1
  print(f"Episode {i}===============================================")
  print(f"Loss          : {round(loss, 3)}")
  print(f"Time          : {steps}")
  print(f"Total Reward  : {round(tot_rew, 3)}")
  print(f"Current Explore : {round(agent._eps, 3)}")
  print(f"Last action : {params}")
  print(f"End=======================================================")
  agent.decay_epsilon()
  rewards[i]  = tot_rew
  losses[i]   = loss
  plt.clf()
  plt.plot(rewards[:i])
  plt.ylabel('Rewards')
  plt.xlabel('Episodes')
  plt.savefig(f'./rl_model/{folder_name}/rewards.png')
  plt.clf()
  plt.plot(losses[:i])
  plt.ylabel('Loss value of function')
  plt.xlabel('Episodes')
  plt.savefig(f'./rl_model/{folder_name}/loss.png')
  if i % 10 == 0:
    agent.save_net(
      f'./rl_model/{folder_name}/model/actor_{i}.pt', f'./rl_model/{folder_name}/model/critic_{i}.pt'
    )
    test_duration_array   = []
    for j in range(30):
      test_dur  = 0
      obs       = env.reset()
      done      = False
      while not done:
        if j % 50 == 0:
          env.render()
          pass
        obj.save_state(np.array(obs, dtype = np.float32))
        params  = agent.net_action(torch.tensor(obs, dtype = torch.float32))
        obj.predict(params.numpy()[0].tolist())
        action  = obj.action()[0]
        next_obs, rew, done, info   = env.step(action)
        obs     = next_obs
        test_dur += 1
      test_duration_array.append(test_dur)
    print('Average Time : %s' % (sum(test_duration_array) / len(test_duration_array)))
