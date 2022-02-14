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

from mpc.control import ContinuousCartPole

obj   = ContinuousCartPole(
  pred_horizon = 30, 
  control_step = 200
)
action_arr  = np.zeros(100, dtype = np.float32)
theta_arr   = np.zeros((100, 4), dtype = np.float32)
begin_time  = datetime.datetime.now()
if not os.path.isdir('./mpc_graph'):
  os.mkdir('./mpc_graph')
for i in range(eval_n):
  done    = False
  steps   = 0
  obs     = env.reset()
  obj.save_state(np.array(obs, dtype = np.float32))
  while not done:
    if render and i % render_n == 0:
      env.render()
      if use_sleep:
        time.sleep(0.02)
    obj.predict()
    action  = obj.action()
    action_arr[steps]   = action[0]
    theta_arr[steps]    = obs.flatten()
    obs, rew, _, info = env.step(action)
    obj.save_state(np.array(obs, dtype = np.float32))
    steps += 1
    if abs(obs[2]) > 0.209 or abs(obs[0]) > 2.4 or steps == 100:
      break
  time_arr.append(steps)
  if abs(obs[2]) > 0.209:
    ang_b += 1
  elif abs(obs[0]) > 2.4:
    bou_b += 1
  else:
    suc_b += 1
  # Make plot and save and reset
  # os.mkdir(f'./mpc_graph/{i}_iter')
  # for j in range(4):
  #   plt.clf()
  #   plt.plot(obj.history['obs'][:, j] - obj.history['r_obs'][:, j])
  #   plt.savefig(f'./mpc_graph/{i}_iter/{j}_plot.png')
  obj.reset_history()

end_time  = datetime.datetime.now()
  
print("Dead by angle : %s" % ang_b)
print("Dead by bound : %s" % bou_b)
print("Succesful iter: %s" % suc_b)
print("Time taken : %s" % (end_time - begin_time))
print('Average time per iteration : %s' % ((end_time - begin_time).total_seconds() / 100))

name  = f'pred30vtFsmth{datetime.datetime.now()}'
os.mkdir(f'./mpc_graph/{name}')
plt.clf()
plt.plot(theta_arr[:,0])
plt.savefig(f'./mpc_graph/{name}/x.png')
plt.clf()
plt.plot(theta_arr[:,1])
plt.savefig(f'./mpc_graph/{name}/v.png')
plt.clf()
plt.plot(theta_arr[:,2])
plt.savefig(f'./mpc_graph/{name}/t.png')
plt.clf()
plt.plot(theta_arr[:,3])
plt.savefig(f'./mpc_graph/{name}/w.png')

plt.clf()
plt.plot(action_arr)
plt.savefig(f'./mpc_graph/{name}/F.png')

fig, grp= plt.subplots(2, 3)
grp[0, 0].plot(action_arr)
grp[0, 0].set_title('Force')
grp[1, 0].plot(theta_arr[:, 2])
grp[1, 0].set_title('Theta')
grp[1, 1].plot(theta_arr[:, 1])
grp[1, 1].set_title('Velocity')
grp[0, 2].plot(theta_arr[:, 3])
grp[0, 2].set_title('Omega')
grp[0, 1].plot(theta_arr[:, 0])
grp[0, 1].set_title('Displacement')
plt.show()

env.close()