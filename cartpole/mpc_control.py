import sys
import gym
import time
import numpy as np

from mpc.control import CartPole

env   = gym.make('CartPole-v0')

obj   = CartPole(
  cart_mass   = 1.0, 
  pole_mass   = 0.1, 
  gravity     = 9.8, 
  pole_length = 1.0, 
  control_dt  = 0.02,
  time_accur  = 0.02,
  pred_horizon= 0.1 #(10 step prediction horizon)
)
obs   = env.reset()
obj.save_state(np.array(obs, dtype = np.float32))
reward  = 0
accum   = []
eps     = 100
use_sleep = False
ang_b   = 0
bou_b   = 0
suc_b   = 0
render  = True
for i in range(eps):
  done = False
  reward = 0  
  while not done:
    if render and i % 20 == 0:
      env.render()
    obj.predict()
    action  = obj.action()[0]
    obs, rew, done, info = env.step(action)
    obj.save_state(np.array(obs, dtype = np.float32))
    reward += rew
    if use_sleep:
      time.sleep(0.1)
  if abs(obs[2]) > 0.209:
    ang_b += 1
  elif abs(obs[0]) > 2.4:
    bou_b += 1
  else:
    suc_b += 1
  obs   = env.reset()
  obj.reset() 
  obj.save_state(np.array(obs, dtype = np.float32))
  accum.append(reward)

print(sum(accum) / eps)
print("Dead by angle : %s" % ang_b)
print("Dead by bound : %s" %  bou_b)
print("Succesful iter: %s" % suc_b)
env.close()
