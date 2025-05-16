from typing import Dict, List
import time, math

def format_digit_amount(x:float, amnt:int=4) -> str:
   log = math.log10(x) + 1
   if log > amnt:
      return f"{x:.0f}"
   elif log > amnt-1:
      return f"{x:.1f}"
   elif log > amnt-2:
      return f"{x:.2f}"
   elif log > amnt-3:
      return f"{x:.3f}"
   elif log > amnt-4:
      return f"{x:.4f}"
   elif log > amnt-5:
      return f"{x:.5f}"
   else:
      raise ValueError(f"Invalid inputs, x={x} log={log} amnt={amnt}")

class TimerContext:
   def __init__(self, data:Dict[str,float], key:str):
      self.data = data
      self.key  = key
   def __enter__(self):
      self.start = time.time()
   def __exit__(self, *args, **kwargs):
      self.data[self.key] = time.time() - self.start

class Controller:
   def __init__(self, max_steps:int=-1):
      self.max_steps = max_steps
      self.i = -1
      self.timings: Dict[str,float] = {}
      self.start_time = -1.0

   def loop_start(self, timed:bool=True) -> int|None:
      self.timings = {}
      if timed:
         self.start_time = time.time()
      self.i += 1
      if self.max_steps > 0 and self.i >= self.max_steps:
         return None
      return self.i

   def time_block(self, label:str) -> TimerContext:
      return TimerContext(self.timings, label)

   def print_step(self, loss:float|None=None, timings:bool=False) -> None:
      text = f"{self.i+1:8d}"
      if loss is not None:
         text += f" | {format_digit_amount(loss)} loss"
      if timings:
         items = list(self.timings.items())
         if self.start_time > 0:
            items.insert(0, ("step",time.time()-self.start_time))
         for key, value in items:
            text += f" | {format_digit_amount(value*1000, amnt=3)}ms {key}"
      print(text)
