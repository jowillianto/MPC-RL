import sys
sys.path.append('..')

from pid.pid import PID

class ContinuousCartPole(PID):
  def __init__(self, Kp : float = 0, Kd : float = 0, Ki : float = 0):
    super().__init__(
      truth_value = 0, Kp = Kp, Ki = Ki, Kd = Kd, Dt = 0.02
    )
  
  def action(self):
    return super().controlValue()