import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict

class Dictable:
   @classmethod
   def to_dict(cls) -> Dict:
      diff = set(dir(cls)) - set(dir(Dictable))
      return { k: getattr(cls, k) for k in diff }

def write_graph(train_data, test_data, save_dirpath, ylim=(0,None), segmented=False):
   if not os.path.exists( (parent:=os.path.dirname(save_dirpath)) ):
      os.makedirs(parent)

   entries = []

   if not segmented:
      scale = len(train_data) // len(test_data)
      entries.append([np.arange(0, len(train_data)), train_data, 'train'])
      entries.append([np.arange(scale-1, len(train_data), scale), test_data, 'test'])
   else:
      for data, label in [(train_data, 'train'), (test_data, 'test')]:
         for i, d in enumerate(data):
            entries.append([np.arange(len(d)), d, f"{label}_{i}"])

   plt.clf()
   for x, y, label in entries:
      plt.plot(x, y, label=label)
   plt.ylim(ylim)
   plt.legend()
   plt.savefig(save_dirpath)
