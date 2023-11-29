from util import Dictable, Schedules
from typing import Dict

class ModelParams(Dictable):
   vocab_size   = 65
   timesteps    = 400
   time_deltas  = 100
   pos_size     = (timesteps // time_deltas) + 1
   ctx_pos_size = 256
   den_pos_size = 8
   n_layers     = 4
   ctx_dim      = 256
   den_dim      = 256
   den_latent   = 192
   den_timepos  = 64
   ctx_heads    = 8
   den_heads    = 8
   ctx_ff_mult  = 2
   den_ff_mult  = 2
   assert den_dim == den_latent + den_timepos

class Train:
   learning_rate = 2**-12
   batch_size = 1
   test_every = 50
   save_every = 1000
   gen_every  = 1000
   gen_count  = 128

class Phase1Train(Train):
   batch_size = 128
   test_every = 25
   save_every = 500
   gen_every  = 500

class Phase2Train(Train):
   batch_size = 12
   gen_count  = 64*10

class Phase3Train(Train):
   batch_size = 10
   gen_count  = 64*10

class Config:
   dataset = "datasets/shakespear.txt"
   split = 0.9

   model_params = ModelParams
   schedule = Schedules.SQRT
   train: Dict[int, type[Train]] = {
      1: Phase1Train,
      2: Phase2Train,
      3: Phase3Train,
   }
   save_name = "p{0}_model_{1}.safetensors"
