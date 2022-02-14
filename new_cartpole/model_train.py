import torch
import random

class MLP(torch.nn.Module):
  def __init__(self, input_size : int, output_size : int, net_width : int, net_depth : int, learning_rate : float = 1e-4):
    super().__init__()
    net_cat   = [torch.nn.Linear(input_size, net_width)]
    net_cat.append(torch.nn.LeakyReLU())
    for i in range(net_depth):
      net_cat.append(torch.nn.Linear(net_width, net_width))
      net_cat.append(torch.nn.LeakyReLU())
    net_cat.append(torch.nn.Linear(net_width, output_size))
    self._net   = torch.nn.Sequential(*net_cat)
    self._loss  = torch.nn.MSELoss()
    self._optim = torch.optim.Adam(params = self.parameters(), lr = learning_rate)
  
  def forward(self, input):
    return self._net(input)
  
model   = MLP(input_size = 5, output_size = 4, net_width = 64, net_depth = 5, learning_rate = 1e-4)
model.to(torch.device('cuda'))
import gym

env = gym.make('CartPole-v0')

# Collect data for training
train_input_data  = torch.zeros((10000, 5), dtype = torch.float32)
train_output_data = torch.zeros((10000, 4), dtype = torch.float32)
obs = env.reset()
for i in range(10000):
  action                = random.choice([0, 1])
  temp_obs              = torch.tensor(obs, dtype = torch.float32)
  temp_act              = torch.tensor([action], dtype = torch.float32)
  train_input_data[i]   = torch.tensor(torch.cat([temp_obs, temp_act]))      
  next_obs, _, done, _  = env.step(action)
  train_output_data[i]  = torch.tensor(next_obs)
  obs   = next_obs
  if done:
    obs   = env.reset()

epoch   = 100

from torch.utils.data import Dataset, DataLoader
class TempDataset(Dataset):
  def __getitem__(self, x : int):
    return (train_input_data[x], train_output_data[i])
  
  def __len__(self):
    return 10000

trainloader   = DataLoader(dataset = TempDataset(), batch_size = 4, shuffle = True, num_workers = 4)
for i in range(epoch):
  run_loss  = 0
  for data, target in trainloader:
    data    = data.to(torch.device('cuda'))
    target  = target.to(torch.device('cuda'))
    output  = model(data)
    loss    = model._loss(output, target)
    run_loss += loss.item()
    model._optim.zero_grad()
    loss.backward()
    model._optim.step()
  print(f'Epoch {i} : {run_loss}')
torch.save(model.state_dict(), './env_model.pt')
    