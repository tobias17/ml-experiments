import matplotlib.pyplot as plt
import numpy as np
import os
from enum import Enum, auto
from typing import Dict

class Schedules(Enum):
   LINEAR = auto()
   SQRT   = auto()

class Dictable:
   @classmethod
   def to_dict(cls) -> Dict:
      diff = set(dir(cls)) - set(dir(Dictable))
      return { k: getattr(cls, k) for k in diff }

def write_graph(train_data, test_data, save_dirpath, ylim=(0,None), segmented=False, delta=1, offset=0, x_label=None, y_label=None):
   if not os.path.exists( (parent:=os.path.dirname(save_dirpath)) ):
      os.makedirs(parent)

   entries = []

   if not segmented:
      scale = len(train_data) // len(test_data)
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
   figure = plt.gcf()
   figure.set_size_inches(18/1.5, 10/1.5)
   plt.savefig(save_dirpath, dpi=100)
