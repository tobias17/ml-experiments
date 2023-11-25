from util import Dictable, Schedules

# class ModelParams(Dictable):
#    vocab_size  = 65
#    timesteps   = 400
#    time_deltas = 100
#    ctx_size    = 6
#    den_size    = (timesteps // time_deltas) + 1
#    n_layers    = 4
#    ctx_dim     = 6
#    den_dim     = 4
#    latent_dim  = 3
#    timepos_dim = 1
#    ctx_heads   = 2
#    den_heads   = 2
#    ctx_ff_dim  = ctx_dim * 2
#    den_ff_dim  = den_dim * 2
#    assert den_dim == latent_dim + timepos_dim

class ModelParams(Dictable):
   vocab_size  = 65
   timesteps   = 400
   time_deltas = 50
   ctx_size    = 256
   den_size    = (timesteps // time_deltas) + 1
   n_layers    = 4
   ctx_dim     = 256
   den_dim     = 256
   latent_dim  = 192
   timepos_dim = 64
   ctx_heads   = 8
   den_heads   = 8
   ctx_ff_dim  = ctx_dim * 2
   den_ff_dim  = den_dim * 2
   assert den_dim == latent_dim + timepos_dim

class Train(Dictable):
   dataset = "datasets/shakespear.txt"
   learning_rate = 2**-13
   batch_size = 5
   # batch_size = 2
   split = 0.9
   test_every = 50
   save_every = 1000
   gen_every  = 1000
   gen_count  = 512

class Config:
   model_params = ModelParams
   train = Train
   save_name = "model_{0}.safetensor"
   schedule = Schedules.SQRT
