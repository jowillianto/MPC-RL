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

obj       = RLMPC(pred_horizon = 5, control_step = 2000)
agent     = HighLevelMPC(tau = 1e-3, eps = 2, gamma = 0.999, actor_lr = 1e-8, critic_lr = 1e-8, eps_decay = 0.99, mem_size = 30000)

import argparse
parser    = argparse.ArgumentParser()
parser.add_argument('actor_path')
parser.add_argument('critic_path')
args      = parser.parse_args()
agent.load_net(args.actor_path, args.critic_path)

test_duration_array   = []
for j in range(30):
  test_dur  = 0
  obs       = env.reset()
  done      = False
  while not done:
    if j % 10 == 0:
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
