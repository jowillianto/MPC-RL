import time
from env.continuous_cartpole import ContinuousCartPoleEnv
from pid.control import ContinuousCartPole

env   = ContinuousCartPoleEnv()

Kp    = -70
Tp    = 0.02 * 100

obj   = ContinuousCartPole(
  Kp = Kp, Kd = 0.6 * Kp / Tp, Ki = Kp * Tp / 8
)

import json
cfg_file  = open('./config.json')
config    = json.load(cfg_file)

render    = config['render']
render_n  = config['render_every']
eval_n    = config['test_n']
use_sleep = config['use_sleep']
eval_n    = 100

steps     = 0
time_arr  = []

suc_b     = 0
ang_b     = 0 
bou_b     = 0

for i in range(eval_n):
  done  = False
  steps = 0
  obs   = env.reset()
  obj.save_state(obs[2])
  while not done:
    if render and i % render_n == 0:
      env.render()
      if use_sleep:
        time.sleep(0.02)
    action  = obj.action()
    obs, rew, done, info = env.step(action)
    obj.save_state(obs[2])
    steps += 1
  time_arr.append(steps)
  if abs(obs[2]) > 0.209:
    ang_b += 1
    #print('angle dead')
  elif abs(obs[0]) > 2.4:
    bou_b += 1
    #print('bound dead')
  else:
    suc_b += 1

print("Dead by angle : %s" % ang_b)
print("Dead by bound : %s" % bou_b)
print("Succesful iter: %s" % suc_b)
env.close()