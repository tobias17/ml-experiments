import matplotlib as plt
import numpy as np
import os

def write_graph(train_data, test_data, save_dirpath):
   if not os.path.exists( (parent:=os.path.dirname(save_dirpath)) ):
      os.makedirs(parent)

   scale = len(train_data) // len(test_data)
   x1, y1 = np.arange(0, len(train_data)), train_data
   x2, y2 = np.arange(scale-1, len(train_data), scale)

   plt.clf()
   plt.plot(x1, y1, label='train')
   plt.plot(x2, y2, label='test')
   plt.ylim(0, None)
   plt.legend()
   plt.savefig(save_dirpath)
