from util import Dictable, Schedules
from typing import Dict

class ModelParams(Dictable):
   vocab_size   = 65
   timesteps    = 200
   time_deltas  = 100
   ctx_pos_size = 256
   den_pos_size = (timesteps // time_deltas) + 1
   n_layers     = 4
   ctx_dim      = 256//2
   den_dim      = 256//2
   time_dim     = 320//2
   fusion_mult  = 1
   ctx_heads    = 8
   den_heads    = 8
   ctx_ff_mult  = 2
   den_ff_mult  = 2

class Train:
   learning_rate = 2**-12
   batch_size = 1
   test_every = 25
   save_every = 1000
   gen_every  = 1000
   gen_count  = 128

   grad_ctx = False
   grad_den = False

   ctx_tok_loss = False
   den_tok_loss_orig = False
   den_tok_loss_pred = False
   den_tok_noise_loss = False

class Phase1Train(Train):
   batch_size = 128
   save_every = 500
   gen_every  = 500

   grad_ctx = True

   ctx_tok_loss = True

class Phase2Train(Train):
   learning_rate = 2**-13
   batch_size = 20
   save_every = 100
   gen_count  = 64*10

   grad_den = True

   den_tok_loss_orig = True
   den_tok_loss_pred = True
   den_tok_noise_loss = True

class Phase3Train(Train):
   learning_rate = 2**-13
   batch_size = 40
   gen_count  = 64*10

   grad_ctx = True
   grad_den = True

   den_tok_loss_orig = True
   den_tok_loss_pred = True
   den_tok_noise_loss = True

class Config:
   dataset = "datasets/shakespear.txt"
   split = 0.9
   dropout = 0.2

   model_params = ModelParams
   schedule = Schedules.SQRT
   train: Dict[int, type[Train]] = {
      1: Phase1Train,
      2: Phase2Train,
      3: Phase3Train,
   }
   start_phase = 3
   save_name = "p{0}_model_{1}.safetensors"
