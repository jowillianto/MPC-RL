import sys
sys.path.append('..')
import numpy as np

import casadi as cs
import os
from casadi import cos, sin
from rl.ddpg import DDPGAgent
import torch

class RLMPC:
  def __init__(self, 
    cart_mass : float = 1.0, pole_mass : float = 0.5, pole_length = 2.0, gravity : float = 9.8, init_angle : float = 0.0, 
    control_dt : float = 0.02, pred_horizon : int = 25, control_step : int = 200
  ):
    self._cart_mass   = cart_mass
    self._pole_mass   = pole_mass
    self._pole_length = pole_length
    self._gravity     = gravity
    self._angle       = init_angle
    self._control_dt  = control_dt
    self._pred_horizon= pred_horizon
    self._control_step= control_step
    
    act_size          = 1
    obs_size          = 4
    self._obs_size    = obs_size
    self._act_size    = act_size
    self._obs_mem     = np.zeros([pred_horizon, obs_size], dtype = np.float32)
    self._act_mem     = np.zeros([pred_horizon, act_size], dtype = np.float32)
    self._cur_act     = np.zeros(act_size, dtype = np.float32)

    self.init_model()

  def init_model(self):
    dt              = self._control_dt
    # Declare main variables
    pendulum_force  = cs.SX.sym('F')
    pendulum_disp   = cs.SX.sym('x')
    pendulum_vel    = cs.SX.sym('v')
    pendulum_angle  = cs.SX.sym('t')
    pendulum_ang_vel= cs.SX.sym('w')
    state_vars      = cs.vertcat(pendulum_disp, pendulum_vel, pendulum_angle, pendulum_ang_vel)
    control_vars    = cs.vertcat(pendulum_force)

    # Forward pass through formula
    ang_accel, lin_accel  = self.model(pendulum_angle, pendulum_ang_vel, pendulum_force)
    
    # Forward states
    velocity_next   = lin_accel * dt
    disp_next       = pendulum_vel * dt
    angle_next      = pendulum_ang_vel * dt
    ang_vel_next    = ang_accel * dt

    # Create State matrix
    next_state_vars = cs.vertcat(disp_next, velocity_next, angle_next, ang_vel_next)
    model           = cs.Function('m', [state_vars, control_vars], [next_state_vars])
    pred_horizon    = self._pred_horizon
    nlp_x           = cs.SX.sym('X', 4, pred_horizon + 1)
    nlp_control_var = cs.SX.sym('C', 1, pred_horizon)
    nlp_init_value  = cs.SX.sym('P', 12) # 4 Initial State, 4 Target State, 4 RL Coefficient
    boundary_func   = nlp_x[:, 0] - nlp_init_value[:4]

    cost_function   = 0
    for i in range(1, pred_horizon + 1):
      init_condition  = nlp_x[:, i - 1]
      end_condition   = model(init_condition, nlp_control_var[:, i - 1])
      boundary_func   = cs.vertcat(boundary_func, nlp_x[:, i] - end_condition)
      for j in range(4):
        cost_function += nlp_init_value[j + 8] * (end_condition[j, 0] - nlp_init_value[j + 4])**2 
    
    # Finalization of NLP
    optim_vars      = cs.vertcat(nlp_x.reshape((-1, 1)), nlp_control_var.reshape((-1, 1)))
    self._lbx       = cs.DM.zeros(4 * (pred_horizon + 1) + pred_horizon, 1)
    self._ubx       = cs.DM.zeros(4 * (pred_horizon + 1) + pred_horizon, 1)
    self._lbx[0 : pred_horizon+1]   = -2.4
    self._ubx[0 : pred_horizon+1]   = 2.4
    self._lbx[pred_horizon + 1 : 2 * (pred_horizon+1)]   = -cs.inf
    self._ubx[pred_horizon + 1 : 2 * (pred_horizon+1)]   = cs.inf
    self._lbx[2*(pred_horizon+1):3*(pred_horizon+1)]  = -0.2
    self._ubx[2*(pred_horizon+1):3*(pred_horizon+1)]  = 0.2
    self._lbx[3*(pred_horizon+1):4*(pred_horizon+1)]   = -cs.inf
    self._ubx[3*(pred_horizon+1):4*(pred_horizon+1)]   = cs.inf
    self._lbx[4*(pred_horizon+1):]  = -1
    self._ubx[4*(pred_horizon+1):]  = 1
    self._lbg = cs.DM.zeros((4 * (pred_horizon + 1), 1))
    self._ubg = cs.DM.zeros((4 * (pred_horizon + 1), 1))
    self._lbg[:]   = -1e-10
    self._ubg[:]   = 1e-10
    
    # Declare NLP
    parameters  = nlp_init_value.reshape((-1, 1))
    nlp_prob    = {
      'f' : cost_function, 'x' : optim_vars, 'g' : boundary_func, 'p' : parameters
    }
    nlp_options = {
      'ipopt' : { 
        'print_level' : 0, 
        'acceptable_tol'  : 1e-8, 
        'acceptable_obj_change_tol' : 1e-6
      }, 
      'print_time' : 0
    }
    self._nlp_solver  = cs.nlpsol('solver', 'ipopt', nlp_prob, nlp_options)
  
  def reset(self):
    pred_horizon      = self._pred_horizon
    obs_size          = self._obs_size
    act_size          = self._act_size
    self._obs_mem     = np.zeros([pred_horizon, obs_size], dtype = np.float32)
    self._act_mem     = np.zeros([pred_horizon, act_size], dtype = np.float32)
    self._cur_act     = np.zeros(act_size, dtype = np.float32)
  
  def save_state(self, state : np.ndarray):
    self._obs_mem[0]  = state
  
  def predict(self, rl_params : list):
    displacement, velocity, theta, omega  = self._obs_mem[0]
    parameters        = [displacement, velocity, theta, omega, 0, 0, 0, 0, *rl_params]
    pred_horizon      = self._pred_horizon
    x0                = cs.vertcat(self._obs_mem.flatten().tolist(), [0, 0, 0, 0], self._act_mem.flatten()[1:].tolist(), [0])
    solution          = self._nlp_solver(x0 = x0, lbx = self._lbx, ubx = self._ubx, lbg = self._lbg, ubg = self._ubg, p = parameters)
    solution_array    = np.array(solution['x'])
    observation_mem   = solution_array[:4 * (pred_horizon + 1)]
    action_mem        = solution_array[4 * (pred_horizon + 1):]
    for i in range(pred_horizon):
      self._obs_mem[i]  = np.array(observation_mem[4 * i : 4 * (i + 1)].flatten(), dtype = np.float32)
      self._act_mem[i]  = np.array(action_mem[i], dtype = np.float32)

  def action(self):
    return self._act_mem[0]

  def model(self, theta : float, omega : float, force : float):
    m = self._pole_mass
    M = self._cart_mass
    g = self._gravity
    t = theta
    w = omega
    l = self._pole_length
    F = force
    a = (6 * (m + M) * g * sin(t) - 6 * F * cos(t) + 3 * m * w**2 * l * sin(t) * cos(t) )/(4 * (m + M) + 3 * m * cos(t)**2 )
    x = (2 * F + m * l * (a * cos(t) - w**2 * sin(t) ) ) / (2 * (m + M) )
    return a, x

class Actor(torch.nn.Module):
  def __init__(self, 
    net_width : int = 32, input_size : int = 4, output_size : int = 1
  ):
    super().__init__()
    # self.norm_layer   = torch.nn.BatchNorm1d(input_size)
    self.first_layer  = torch.nn.Linear(input_size, net_width)
    self.first_act    = torch.nn.LeakyReLU()
    self.inter_layer  = torch.nn.Linear(net_width, net_width)
    self.inter_act    = torch.nn.LeakyReLU()
    self.final_layer  = torch.nn.Linear(net_width, output_size)
  def forward(self, x):
    #x   = self.norm_layer(x)
    x   = self.first_layer(x)
    x   = self.first_act(x)
    x   = self.inter_layer(x)
    x   = self.inter_act(x)
    x   = self.final_layer(x)
    return torch.nn.functional.normalize(x)

class Critic(torch.nn.Module):
  def __init__(self, 
    net_width : int = 32, obs_size : int = 4, act_size : int = 1, output_size : int = 1
  ):
    super().__init__()
    self.norm_layer   = torch.nn.BatchNorm1d(obs_size)
    self.first_layer  = torch.nn.Linear(obs_size + act_size, net_width)
    self.first_act    = torch.nn.LeakyReLU()
    self.inter_layer  = torch.nn.Linear(net_width, net_width)
    self.inter_act    = torch.nn.LeakyReLU()
    self.final_layer  = torch.nn.Linear(net_width, output_size)
    self.obs_size     = obs_size
  def forward(self, obs, action):
    #x   = self.norm_layer(obs)
    x   = torch.cat((obs, action), dim = -1)
    x   = self.first_layer(x)
    x   = self.first_act(x)
    x   = self.inter_layer(x)
    x   = self.inter_act(x)
    x   = self.final_layer(x)
    return x

class HighLevelMPC(DDPGAgent):
  def __init__(
    self, tau : float, eps : float, gamma : float, actor_lr : float, critic_lr : float, eps_decay : float, mem_size : int
  ):
    actor   = Actor
    actorkwargs   = {
      'net_width' : 64, 'input_size' : 4, 'output_size' : 4
    }
    critic  = Critic
    critickwargs  = {
      'net_width' : 64, 'obs_size' : 4, 'output_size' : 1, 'act_size' : 4
    }

    super().__init__(
      tau = tau, eps = eps, gamma = gamma, actor_lr = actor_lr, critic_lr = critic_lr, eps_decay = eps_decay, 
      actor = actor, actorkwargs = actorkwargs, critic = critic, critickwargs = critickwargs, 
      mem_size = mem_size, obs_size = 4, act_size = 4
    )