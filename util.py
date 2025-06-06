import matplotlib.pyplot as plt
import numpy as np
import os
from enum import Enum, auto
from typing import Dict, List

class Schedules(Enum):
   LINEAR = auto()
   SQRT   = auto()
   SPLIT  = auto()
   ONRAMP = auto()

class Dictable:
   @classmethod
   def to_dict(cls) -> Dict:
      diff = set(dir(cls)) - set(dir(Dictable))
      return { k: getattr(cls, k) for k in diff }

def write_graph(train_data, test_data, save_dirpath, ylim=(0,None), segmented=False, delta=1, offset=0, x_label=None, y_label=None, show_train=True, title=None):
   if not os.path.exists( (parent:=os.path.dirname(save_dirpath)) ):
      os.makedirs(parent)

   entries = []

   if not segmented:
      scale = len(train_data) // len(test_data)
      if show_train:
         entries.append([np.arange(0, len(train_data))*delta+delta+offset, train_data, 'train'])
      entries.append([np.arange(scale-1, len(train_data), scale)*delta+delta+offset, test_data, 'test'])
   else:
      for data, label in [(train_data, 'train'), (test_data, 'test')]:
         for i, d in enumerate(data):
            entries.append([np.arange(0, len(d))*delta+delta+offset, d, f"{label}_{i}"])

   plt.clf()
   for x, y, label in entries:
      plt.plot(x, y, label=label)
   plt.ylim(ylim)
   plt.legend()
   if x_label is not None: plt.xlabel(x_label)
   if y_label is not None: plt.ylabel(y_label)
   if title is not None: plt.title(title)
   figure = plt.gcf()
   figure.set_size_inches(18/1.5, 10/1.5)
   plt.savefig(save_dirpath, dpi=100)

def write_probs(data, save_dirpath, ylim=(0,None), delta=1, offset=0, x_label=None, y_label=None, title=None):
   plt.clf()
   x = np.arange(0, len(data))*delta+delta+offset
   plt.plot(x, [d[0] for d in data], label="q1")
   plt.plot(x, [d[1] for d in data], label="med")
   plt.plot(x, [d[2] for d in data], label="q3")
   plt.ylim(ylim)
   plt.legend()
   if x_label is not None: plt.xlabel(x_label)
   if y_label is not None: plt.ylabel(y_label)
   if title is not None: plt.title(title)
   figure = plt.gcf()
   figure.set_size_inches(18/1.5, 10/1.5)
   plt.savefig(save_dirpath, dpi=100)

def shrink_format(x:float) -> str:
   return f"{x:.2f}" if x < 10 else (f"{x:.1f}" if x < 100 else str(int(x)))

def compress(x:float, labels:List[str], step:int=1000, fmt=shrink_format) -> str:
   i = 0
   padded_labels = [""] + labels
   while True:
      if x < step or i == len(padded_labels) - 1:
         return fmt(x) + padded_labels[i]
      x /= step
      i += 1

def fmt_digits(x:float, digits:int) -> str:
   r = f"{x:.8f}"
   if len(r.split(".", 1)[0]) == digits-1:
      return " " + r[:digits-1]
   return r[:digits]

def fmt_time(x:float, digits:int=5) -> str:
   if x > 3600*10:
      v, u = x/3600, "hr"
   elif x > 60*10:
      v, u = x/60, "mn"
   elif x > 1*10:
      v, u = x, "s"
   else:
      v, u = x*1000, "ms"
   return f"{fmt_digits(v, digits)} {u}" + " "*(2-len(u))

def fmt_percent(x:float, digits:int=4) -> str:
   return fmt_digits(x*100, digits) + "%"

if __name__ == "__main__":
   for v in [1, 500, 1000, 2000, 20000, 200000, 2000000, 20000000]:
      print(compress(v, ["k", "m", "b"]))
