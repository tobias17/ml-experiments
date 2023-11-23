from util import Dictable

class ModelParams(Dictable):
   timesteps   = 400
   vocab_size  = 65
   n_layers    = 4
   embed_dim   = 256
   latent_dim  = 192
   timepos_dim = 64
   n_heads     = 8
   ff_dim      = embed_dim * 2
   max_context = 256
   assert embed_dim == latent_dim + timepos_dim

class Train(Dictable):
   learning_rate = 2**-13
   batch_size = 128
   dataset = "datasets/shakespear.txt"
   split = 0.9
   test_every = 50
   save_every = 500
   gen_every  = 500
   gen_count  = 512

class Config:
   model_params = ModelParams
   train = Train
   save_name = "model_{0}.safetensor"
