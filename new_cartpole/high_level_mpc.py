from env.continuous_cartpole import ContinuousCartPoleEnv
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt
import os

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

from mpc_rl.control_refac import RLMPC

obj       = RLMPC(pred_horizon = 30, control_step = 2000)
action_arr= np.zeros(2000, dtype = np.float32)
theta_arr = np.zeros((2000, 4), dtype = np.float32)

if not os.path.isdir('./mpc_graph'):
  os.mkdir('./mpc_graph')

# Train Loop
episode   = 1000000
for i in range(episode):
  done          = False
  steps         = 0
  obs           = env.reset()
  obs_arr       = np.zeros((2000, 4), dtype = np.float32)
  next_obs_arr  = np.zeros(2000, dtype = np.float32)

  # Get MPC Params
  params        = [0, 0, 0, 0]
  # Sample Episode
  while not done:
    # Save the state
    obj.save_state(np.array(obs, dtype = np.float32))
    obs_arr[0]    = obs
    if render and i % render_n == 0:
      env.render()
    # Predict Trajectory
    obj.predict(params)
    # Take an action
    action      = obj.action()
    # Look at consequences of action
    next_obs, rew, done, info   = env.step(action)
    obj.save_state(np.array(next_obs, dtype = np.float32))

    # Save everything for RL
    next_obs_arr[steps] = np.array(next_obs, dtype = np.float32)
    obs         = next_obs

    # Increment steps
    steps += 1
  
  # Calculate Reward for Trajectory
  trajectory_reward   = 1e-3 * steps
  for i in range(steps):
    trajectory_reward += 0.05 - obs_arr[i, 2]**2
  
  # Save Everything to Reinforcement Learning Buffer
  
  # Train Reinforcement Learning Network
  

