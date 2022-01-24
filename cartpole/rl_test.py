import gym
import numpy as np
import time

from rl.control import CartPole

env = gym.make('CartPole-v0')
obj = CartPole(
  mem_size        = 10000,
  discount_factor = 0.9,
  eps_start       = 0.9, 
  eps_mlp         = 0.99999, 
  learning_rate   = 1e-3
)
obs = env.reset()

obj.load_net('./iter_7500.pt')
reward  = 0
accum   = []
eps     = 10
use_sleep = False
render  = True
ang_b   = 0
bou_b   = 0
suc_b   = 0
for i in range(eps):
  done = False
  reward = 0
  obs   = env.reset()  
  while not done:
    if render and i % 1 == 0:
      env.render()
    action  = obj.net_action(np.array([obs], dtype = np.float32))
    obs, rew, done, info = env.step(action)
    reward += rew
    if use_sleep:
      time.sleep(0.1)
    if reward >= 200 and abs(obs[2]) < 0.209 and abs(obs[0]) < 2.4:
      done = False
  if obs[2] > 0.209:
    ang_b += 1
  elif abs(obs[0]) > 2.4:
    bou_b += 1
  else:
    suc_b += 1
  obj.reset() 
  accum.append(reward)

print(sum(accum) / eps)
print("Dead by angle : %s" % ang_b)
print("Dead by bound : %s" % bou_b)
print("Succesful iter: %s" % suc_b)
env.close()