import sys
sys.path.append('..')

from pid.pid import PID

class CartPole(PID):
  def __init__(self, 
    truth_value : float = 0, Kp : float = 0.1, Kd : float = 0, Ki : float = 0, Dt : float = 0.02
  ):
    super().__init__(truth_value, Kp, Ki, Kd, Dt)
  
  def approximate_on_pid(self):
    force   = self.controlValue()
    d1      = (force - 10) ** 2
    d2      = (force + 10) ** 2
    if d1[2] < d2[2]:
      return 0
    else:
      return 1

class CartPole2(PID):
  def __init__(self, 
    truth_value : float = 0, Kp : float = 0.1, Kd : float = 0, Ki : float = 0, Dt : float = 0.02
  ):
    super().__init__(truth_value, Kp, Ki, Kd, Dt)
  
  def approximate_on_pid(self):
    force   = self.controlValue()
    print(force)
    d1      = (force[0] + force[2] - 10) ** 2
    d2      = (force[0] + force[2] + 10) ** 2
    if d1 < d2:
      return 0
    else:
      return 1