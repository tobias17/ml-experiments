from util import Dictable, Schedules
from typing import Dict

vocab_size = 65

class CtxModelParams(Dictable):
   vocab_size = vocab_size
   pos_size   = 256
   n_layers   = 4
   embed_dim  = 256
   n_heads    = 8
   ff_dim     = embed_dim * 2

class DenModelParams(Dictable):
   vocab_size  = 65
   timesteps   = 400
   time_deltas = 50
   pos_size    = (timesteps // time_deltas) + 1
   n_layers    = 4
   ctx_dim     = CtxModelParams.embed_dim
   embed_dim   = 256
   latent_dim  = 192
   timepos_dim = 64
   n_heads     = 8
   ff_dim      = embed_dim * 2
   assert embed_dim == latent_dim + timepos_dim

class Train:
   learning_rate = 2**-12
   batch_size = 5
   test_every = 50
   save_every = 1000
   gen_every  = 1000
   gen_count  = 512

class Phase1Train(Train):
   batch_size = 108
   test_every = 25
   save_every = 500
   gen_every  = 500

class Phase2Train(Train):
   batch_size = 5

class Phase3Train(Train):
   batch_size = 4

class Config:
   dataset = "datasets/shakespear.txt"
   split = 0.9

   ctx_model_params = CtxModelParams
   den_model_params = DenModelParams
   train: Dict[int, type[Train]] = {
      1: Phase1Train,
      2: Phase2Train,
      3: Phase3Train,
   }
   ctx_save_name = "p{0}_ctx_{1}.safetensors"
   den_save_name = "p{0}_den_{1}.safetensors"
   schedule = Schedules.SQRT
