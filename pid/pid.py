class PID:
  def __init__(
    self, truth_value,
    Kp : float = 0.1, Ki : float = 0.1, Kd : float = 0.1, Dt : float = 0.1
  ):
    self._Kp_   = Kp
    self._Ki_   = Ki
    self._Kd_   = Kd
    self._Dt_   = Dt
    self._Obs_  = truth_value - truth_value
    self._PObs_ = truth_value - truth_value
    self._NObs_ = truth_value - truth_value
    self._TObs_ = truth_value

  def reset(self):
    truth_value = self._TObs_
    self._Obs_  = truth_value - truth_value
    self._PObs_ = truth_value - truth_value
    self._NObs_ = truth_value - truth_value
    self._TObs_ = truth_value

  def save_state(self, obs):
    self._PObs_ = self._Obs_
    self._Obs_  = self._NObs_
    self._NObs_ = self._TObs_ - obs
  
  def controlValue(self):
    prop  = self._NObs_ * self._Kp_

    #numerical differentiation
    diff  = (self._NObs_ - self._PObs_) / (2 * self._Dt_) * self._Kd_

    #numerical integration
    intgt = self._Ki_ * self._Dt_ / 3 * (self._PObs_ + 4 * self._Obs_ + self._NObs_)
  
    return prop + diff + intgt