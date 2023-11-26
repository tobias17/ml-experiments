from util import Dictable
from typing import Dict

class ModelParams(Dictable):
   vocab_size   = 65
   ctx_pos_size = 256
   dec_pos_size = 8
   n_layers     = 4
   ctx_dim      = 256
   dec_dim      = 256
   ctx_heads    = 8
   dec_heads    = 8
   ctx_ff_dim   = ctx_dim * 2
   dec_ff_dim   = dec_dim * 2

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
   gen_every  = 500000

class Phase3Train(Train):
   batch_size = 4
   gen_every  = 500000

class Config:
   dataset = "datasets/shakespear.txt"
   split = 0.9

   model_params = ModelParams
   train: Dict[int, type[Train]] = {
      1: Phase1Train,
      2: Phase2Train,
      3: Phase3Train,
   }
   save_name = "p{0}_model_{1}.safetensors"
