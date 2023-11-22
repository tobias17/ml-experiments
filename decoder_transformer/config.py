from typing import Dict

class Dictable:
   @classmethod
   def to_dict(cls) -> Dict:
      diff = set(dir(cls)) - set(dir(Dictable))
      return { k: getattr(cls, k) for k in diff }

class ModelParams(Dictable):
   vocab_size = 65
   layers = 4
   embed_dim = 256
   n_heads = 8
   ff_dim = 512
   max_context = 256

class Train(Dictable):
   lr = 2**-12
   batch_size = 1
   dataset = "datasets/shakespear.txt"
   split = 0.9
   test_every = 10
   save_every = 50

class Config:
   model_params = ModelParams
   train = Train
   save_name = "model_{0}.safetensor"
