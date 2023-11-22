from util import Dictable

class ModelParams(Dictable):
   vocab_size = 65
   layers = 4
   embed_dim = 256
   n_heads = 8
   ff_dim = 512
   max_context = 256

class Train(Dictable):
   lr = 2**-16
   batch_size = 12
   dataset = "datasets/shakespear.txt"
   split = 0.9
   test_every = 50
   save_every = 500

class Config:
   model_params = ModelParams
   train = Train
   save_name = "model_{0}.safetensor"
