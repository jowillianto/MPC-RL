import sys
sys.path.append('..')

from mpc.model import Model as MPCModel
import numpy as np
from math import sin, cos

class CartPole(MPCModel):
  def __init__(self,
    cart_mass : float = 1.0, pole_mass : float = 0.5, pole_length : float = 2.0, gravity : float = 9.8, init_angle : float = 0.0, 
    control_dt : float = 0.02, time_accur : float = 0.01, pred_horizon = 0.1, mem_size : int = 10
  ):
    #parameters
    self._CartMass_   = cart_mass
    self._PoleMass_   = pole_mass
    self._PoleLength_ = pole_length
    self._Gravity_    = gravity
    self._Angle_      = init_angle

    #init  super class
    super().__init__(mem_size, 4, 1, np.int32)

    #properties
    self._CartX_      = np.zeros([mem_size, 3], dtype = np.float32)
    self._PoleT_      = np.zeros([mem_size, 3], dtype = np.float32)
    
    #Control properties
    self._ControlDt_  = control_dt
    self._TimeAccur_  = time_accur
    self._PredHor_    = pred_horizon

  def reset(self, init_angle : float = 0.0):
    super().reset()
    mem_size          = self._MemSize_
    self._Angle_      = init_angle
    self._CartX_      = np.zeros([mem_size, 3], dtype = np.float32)
    self._PoleT_      = np.zeros([mem_size, 3], dtype = np.float32)
  
  def save_state(self, state : np.ndarray):
    super().save_state(state)
    cart_x, cart_v, pole_t, pole_w = state
    self._Angle_  = pole_t
    cart_x   = np.array([cart_x, cart_v, 0], dtype = np.float32)
    pole_t   = np.array([pole_t, pole_w, 0], dtype = np.float32)
    self._CartX_  = np.concatenate([
      [cart_x], self._CartX_[:self._MemSize_ - 1]
    ])
    self._CartY_  = np.concatenate([
      [pole_t], self._PoleT_[:self._MemSize_ - 1]
    ])

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
    t   = self._Angle_
    w   = self._PoleT_[0, 1]
    dt  = self._ControlDt_
    ta  = self._TimeAccur_  
    ph  = self._PredHor_
    n   = int(dt // ta)
    k   = int(ph // dt)
    v   = self._CartX_[0, 1]
    p   = self._CartX_[0, 0]
    a1, x1  = self._CalcAcc_(t, w, 10)
    a2, x2  = self._CalcAcc_(t, w, -10)
    t1 = t2 = t
    w1 = w2 = w
    v1 = v2 = v
    p1 = p2 = p
    
    d1 = 0
    d2 = 0
    for j in range(k):
      #iterate for the first
      for i in range(n):
        t1      = t1 + w1 * dt + 0.5 * a1 * dt**2
        p1      = p1 + v1 * dt + 0.5 * x1 * dt**2
        w1      = w1 + a1 * dt
        v1      = v1 + x1 * dt
        a1, x1  = self._CalcAcc_(t1, w1, 10)

      for i in range(n):
        t2      = t2 + w2 * dt + 0.5 * a2 * dt**2
        p2      = p2 + v2 * dt + 0.5 * x2 * dt**2
        w2      = w2 + w2 * dt
        v2      = v2 + x2 * dt
        a2, x2  = self._CalcAcc_(t2, w2, -10)
      
      #factors 
      dir_fac   = 0
      omg_fac   = 0
      vel_fac   = 1e-3    
      vdr_fac   = 0
      t_fac     = 1

      c1  = t_fac * t1**2 + dir_fac * p1**2 + omg_fac * w1**2 + vel_fac * v1**2 + v1 * p1 * vdr_fac / abs(p1) / abs(v1)
      c2  = t_fac * t2**2 + dir_fac * p2**2 + omg_fac * w2**2 + vel_fac * v2**2 + v2 * p2 * vdr_fac / abs(p2) / abs(v2)

      if c1 < c2:
        d1 += 1
      else:
        d2 += 1

    action = 1 if d1 > d2 else 0
    super()._SaveAction_(np.array([action], dtype = np.int32))

    


