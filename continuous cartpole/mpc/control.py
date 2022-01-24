import sys

from casadi.casadi import vertcat
sys.path.append('..')

from mpc.model import Model
import numpy as np

import casadi as cs
import os
from casadi import cos, sin

class ContinuousCartPole(Model):
  def __init__(self,
    cart_mass : float = 1.0, pole_mass : float = 0.5, pole_length : float = 2.0, gravity : float = 9.8, init_angle : float = 0.0, 
    control_dt : float = 0.02, pred_horizon : int = 25, control_step : int = 200
  ):
    #parameters
    self._CartMass_   = cart_mass
    self._PoleMass_   = pole_mass
    self._PoleLength_ = pole_length
    self._Gravity_    = gravity
    self._Angle_      = init_angle
    self._ControlDt_  = control_dt

    #init  super class
    super().__init__(
      prediction_horizon = pred_horizon, obs_size = 4, act_size = 1, control_step = control_step
    )

  def init_model(self):
    #init the formula for control
    F   = cs.SX.sym('F')
    x   = cs.SX.sym('x')
    v   = cs.SX.sym('v')
    t   = cs.SX.sym('t')
    w   = cs.SX.sym('w')

    st  = cs.vertcat(x, v, t, w)
    ct  = cs.vertcat(F)
    #pass in the formula
    alpha, a  = self._CalcAcc_(t, w, F)
    vn        = v + a * self._ControlDt_
    wn        = w + alpha * self._ControlDt_
    xn        = x + v * self._ControlDt_
    tn        = t + w * self._ControlDt_

    nst = cs.vertcat(xn, vn, tn, wn)
    model     = cs.Function('m', [st, ct], [nst])

    l         = self._PredHor_
    X         = cs.SX.sym('X', 4, l+1)
    cl        = cs.SX.sym('C', 1, l)
    iv        = cs.SX.sym('P', 8)
    g         = X[:, 0] - iv[:4]
    c_f = 0
    for i in range(1, l + 1):
      init_condition    = X[:, i - 1]
      end_condition     = model(init_condition, cl[0, i - 1])
      g                 = cs.vertcat(g, X[:, i] - end_condition)
      c_f = c_f + end_condition[2, 0]**2 + end_condition[1, 0]**2
    #formulate nlp
    opt_vars  = cs.vertcat(
      X.reshape((-1, 1)), cl.reshape((-1, 1))
    )
    self.lbx  = cs.DM.zeros(4 * (l + 1) + l, 1)
    self.ubx  = cs.DM.zeros(4 * (l + 1) + l)

    self.lbx[0:l+1]   = -2.4
    self.ubx[0:l+1]   = 2.4
    self.lbx[l+1:2*(l+1)]   = -cs.inf
    self.ubx[l+1:2*(l+1)]   = cs.inf
    self.lbx[2*(l+1):3*(l+1)]  = -0.2
    self.ubx[2*(l+1):3*(l+1)]  = 0.2
    self.lbx[3*(l+1):4*(l+1)]   = -cs.inf
    self.ubx[3*(l+1):4*(l+1)]   = cs.inf
    self.lbx[4*(l+1):]  = -10
    self.ubx[4*(l+1):]  = 10
    self.lbg = cs.DM.zeros((4 * (l + 1), 1))
    self.ubg = cs.DM.zeros((4 * (l + 1), 1))
    self.lbg[:]   = -1e-10
    self.ubg[:]   = 1e-10

    params  = iv.reshape((-1, 1))
    nlp_prob  = {
      'f'   : c_f, 
      'x'   : opt_vars, 
      'g'   : g, 
      'p'   : params
    }

    opts  = {
      'ipopt' : { 
        'print_level' : 0, 
        'acceptable_tol'  : 1e-8, 
        'acceptable_obj_change_tol' : 1e-6
      }, 
      'print_time' : 0
    }
    self.solver   = cs.nlpsol('solver', 'ipopt', nlp_prob, opts)


  def reset(self):
    super().reset()
  
  def save_state(self, state : np.ndarray):
    super().save_state(state)

  def _CalcAcc_(self, 
    current_angle : float = 0.0, current_angular_velocity : float = 0.0, force : float = 0.0
  ):
    m = self._PoleMass_
    M = self._CartMass_
    g = self._Gravity_
    t = current_angle
    w = current_angular_velocity
    l = self._PoleLength_
    F = force
    a = (6 * (m + M) * g * sin(t) - 6 * F * cos(t) + 3 * m * w**2 * l * sin(t) * cos(t) )/(4 * (m + M) + 3 * m * cos(t)**2 )
    x = (2 * F + m * l * (a * cos(t) - w**2 * sin(t) ) ) / (2 * (m + M) )
    return a, x
  
  def predict(self):
    x, v, t, w  = self.cur_state()
    params  = [x, v, t, w, 0, 0, 0, 0]
    l   = self._PredHor_
    x0  = [x, v, t, w]
    x0  = cs.vertcat(x0, list(self.obs_mem.flatten()[4:]), [0, 0, 0, 0])
    x0  = cs.vertcat(x0, list(self.act_mem.flatten()[1:]), [0])

    solution = self.solver(
      x0 = x0, 
      lbx = self.lbx, ubx = self.ubx, 
      lbg = self.lbg, ubg = self.ubg, 
      p = params
    )
    x             = np.array(solution['x'])
    obs           = x[:4*(l + 1)]
    act           = x[4*(l + 1):]
    for i in range(l):
      self.obs_mem[i]   = np.array(obs[4*i:4*(i+1)].flatten(), dtype = np.float32)
      self.act_mem[i]   = np.array(act[i], dtype = np.float32)
    self.cur_act[0]   = act[0]
    