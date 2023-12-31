from util import Dictable, Schedules

class ModelParams(Dictable):
   timesteps   = 400
   vocab_size  = 65
   n_layers    = 8
   embed_dim   = 256
   latent_dim  = 192
   timepos_dim = 64
   n_heads     = 8
   ff_dim      = embed_dim * 2
   max_context = 256
   assert embed_dim == latent_dim + timepos_dim

class Train(Dictable):
   dataset = "datasets/shakespear.txt"
   learning_rate = 2**-12
   batch_size = 64
   split = 0.9
   test_every = 50
   save_every = 1000
   gen_every  = 1000
   gen_count  = 512

class Config:
   model_params = ModelParams
   train = Train
   save_name = "model_{0}.safetensor"
   timestep_delta = 100
   schedule = Schedules.SQRT
