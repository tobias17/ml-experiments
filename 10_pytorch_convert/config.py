from util import Dictable, Schedules
from typing import Dict

class ModelParams(Dictable):
   vocab_size   = 50304
   timesteps    = 256
   time_deltas  = 64
   ctx_pos_size = 64
   den_pos_size = (timesteps // time_deltas) + 1
   n_layers     = 16
   ctx_dim      = 768
   den_dim      = 768
   time_dim     = 768//2
   fusion_mult  = 1
   ctx_heads    = 12
   den_heads    = 12
   ctx_ff_mult  = 2
   den_ff_mult  = 2

class Train:
   learning_rate = 2**-16
   batch_size = 1
   test_every = 400
   deep_every = 4000
   save_every = 20000
   gen_every  = 20000
   gen_count  = 64

   grad_ctx = False
   detach_ctx = False
   grad_den = False

   ctx_tok_loss = False
   den_tok_loss_orig = False
   den_tok_loss_pred = False
   den_tok_noise_loss = False

class Phase1Train(Train):
   batch_size = 36
   test_every = 200
   deep_every = 1000
   save_every = 10000
   gen_every  = 10000

   grad_ctx = True

   ctx_tok_loss = True

class Phase2Train(Train):
   batch_size = 1

   detach_ctx = True
   grad_den = True

   den_tok_loss_orig = True
   den_tok_loss_pred = True
   den_tok_noise_loss = True

class Phase3Train(Train):
   learning_rate = 2**-17
   batch_size = 1

   grad_ctx = True
   grad_den = True

   den_tok_loss_orig = True
   den_tok_loss_pred = True
   den_tok_noise_loss = True

class Config:
   dataset = "datasets/openweb_{0}.bin"
   dropout = 0.1

   model_params = ModelParams
   schedule = Schedules.SQRT
   train: Dict[int, type[Train]] = {
      1: Phase1Train,
      2: Phase2Train,
      3: Phase3Train,
   }
   start_phase = 1
   save_name = "p{0}_model_{1}.safetensors"
