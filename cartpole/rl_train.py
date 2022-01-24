import gym
import numpy as np

from rl.control import CartPole
import os

env = gym.make('CartPole-v0')
obj = CartPole(
  mem_size        = 10000,
  discount_factor = 0.99,
  eps_start       = 0.7, 
  eps_mlp         = 0.99999, 
  learning_rate   = 1e-6
)
obs = env.reset()

reward  = 0
accum   = []
eps     = 10000000
step    = 0
loss    = 0
batch_size  = 64
target_update = 200
repeat  = 10

try:
  os.mkdir('./model')
except:
  pass

obj.update_target()

for i in range(eps):
  done    = False
  reward  = 0
  dur     = 0
  while not done:
    action  = obj.action(np.array([obs], dtype = np.float32))
    nobs, rew, done, info = env.step(action)
    rew     = 0.05 - (nobs[2] )** 2   
    obj.save_state(
      state = np.array(obs, dtype = np.float32), next_state = np.array(nobs, dtype = np.float32), 
      action = action, reward = rew, terminal = done
    )
    obs     = nobs
    step += 1
    dur += 1
    reward += rew
    if step > batch_size:
      for j in range(repeat):
        loss += obj.train(batch_size = batch_size)
    if dur == 200:
      done = False
    obj.decay_explore()

  #Reset env and observations
  obs   = env.reset()
  accum.append(reward)
  if i % 10 == 0:
    print("Iteration %.1f : %.3f, %.3f, %.3f, %.3f" % (i, sum(accum) / len(accum), loss, obj._Eps_, dur))
    loss = 0
    accum = []
  if i % target_update == 0:
    obj.update_target()

  #Testing
  if i % 100 == 0:
    accumulator = []
    for j in range(100):
      done    = False
      reward  = 0
      dur     = 0
      obs     = env.reset()
      while not done:
        action = obj.net_action(np.array([obs], dtype = np.float32))
        obs, rew, done, _ = env.step(action)
        dur += 1
      env.reset()
      accumulator.append(dur)
    print("Test at iteration %.1f : %.3f" % (i, sum(accumulator) / len(accumulator)))
    print('===========================================')
    obj.save_net(f'./model/iter_{i}.pt')
