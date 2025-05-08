import time

class TimerContext:
   def __init__(self, data:Dict[str,float], key:str):
      self.data = data
      self.key  = key
   def __enter__(self):
      self.start = time.time()
   def __exit__(self, *args, **kwargs):
      self.data[self.key] = time.time() - self.start

class Controller:
   def __init__(self, max_steps:int=-1, launch_server:bool=False):
      self.max_steps = max_steps
      self.curr_step = -1
      self.timings = []

   def loop_start(self) -> bool:
      self.curr_step += 1
      self.timings.append({})
      return True

   def time_block(self, label:str) -> TimerContext:
      return TimerContext(self.timings[-1], label)
