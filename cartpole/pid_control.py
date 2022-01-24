from pid.control import CartPole2, CartPole

import gym
import numpy as np
import time

env     = gym.make('CartPole-v0')

Kpp     = 160
PeriodP = 0.02 * 100
objP    = CartPole(
  truth_value   = 0, 
  Kp            = Kpp, 
  Kd            = 2.0 * Kpp / PeriodP, 
  Ki            = Kpp * PeriodP / 8, 
  Dt            = 0.02
)

obs   = env.reset()
objP.save_state(np.array(obs, dtype = np.float32))
eps     = 100
use_sleep = False
accum   = []
ang_b   = 0
bou_b   = 0
suc_b   = 0

for i in range(eps):
  done = False
  step = 0
  while not done:
    if i % 10 == 0:
      env.render()
    action  = objP.approximate_on_pid()
    obs, rew, done, info = env.step(action)
    objP.save_state(np.array(obs, dtype = np.float32))
    step += 1
    if use_sleep:
      time.sleep(0.1)
  if abs(obs[2]) > 0.209:
    ang_b += 1
    #print('angle dead')
  elif abs(obs[0]) > 2.4:
    bou_b += 1
    #print('bound dead')
  else:
    suc_b += 1
    #print('smart bot')
  obs   = env.reset()
  objP.reset()
  objP.save_state(obs)
  accum.append(step)

print("Dead by angle : %s" % ang_b)
print("Dead by bound : %s" %  bou_b)
print("Succesful iter: %s" % suc_b)
env.close()