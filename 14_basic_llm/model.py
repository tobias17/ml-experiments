from tinygrad import Tensor, TinyJit, dtypes, nn
from tinygrad.helpers import prod
import numpy as np

from training_controller import Controller
from util import compress

class ModelConfig:
   n_layers: int = 24
   dim: int = 512
   vocab_size: int = 32000
   ctx_length: int = 1024

   n_heads: int = 16
   @property
   def head_d(self) -> int: return self.dim // self.n_heads

   ff_mult: float = 3.0
   @property
   def ff_dim(self) -> int: return int(self.dim * self.ff_mult)
   @property
   def ff_act(self): return Tensor.leaky_relu

class Attention:
   def __init__(self, cfg:ModelConfig):
      self.proj_in  = nn.Linear(cfg.dim, cfg.dim*3)
      self.proj_out = nn.Linear(cfg.dim, cfg.dim)
      self.n_heads  = cfg.n_heads
      self.head_d   = cfg.head_d
   def __call__(self, x:Tensor) -> Tensor:
      q,k,v = self.proj_in(x).chunk(3, dim=-1)
      q,k,v = [y.rearrange('b c (h d) -> b h c d', h=self.n_heads, d=self.head_d) for y in (q,k,v)]
      h = Tensor.scaled_dot_product_attention(q,k,v)
      h = h.rearrange('b h c d -> b c (h d)')
      return x + self.proj_out(h)

class FeedForward:
   def __init__(self, cfg:ModelConfig):
      self.sequence = [
         nn.Linear(cfg.dim, cfg.ff_dim),
         cfg.ff_act,
         nn.Linear(cfg.ff_dim, cfg.dim),
         cfg.ff_act,
      ]
   def __call__(self, x:Tensor) -> Tensor:
      return x + x.sequential(self.sequence)

class TransformerBlock:
   def __init__(self, cfg:ModelConfig):
      self.attn = Attention(cfg)
      self.ff = FeedForward(cfg)
   def __call__(self, x:Tensor) -> Tensor:
      return x.sequential([self.attn, self.ff])

class Model:
   def __init__(self, cfg:ModelConfig=ModelConfig()):
      self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
      self.pos_emb = nn.Embedding(cfg.ctx_length, cfg.dim)
      self.layers = [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
      self.proj_out = nn.Linear(cfg.dim, cfg.vocab_size)
   def __call__(self, tok:Tensor) -> Tensor:
      x = self.tok_emb(tok) + self.pos_emb(Tensor.arange(tok.shape[-1]))
      for layer in self.layers:
         x = layer(x)
      return self.proj_out(x)

class Keys:
   TRAIN = "train"

BS = 4
LR = 2**-18

def train():
   Tensor.training = True

   X_train, _ = [np.memmap(f"/raid/datasets/fineweb/tokenized/fineweb_{split}.bin", dtype=np.uint16, mode='r') for split in ('train', 'val')]

   ctr = Controller()
   mdl = Model(cfg := ModelConfig())
   opt = nn.optim.AdamW(params := nn.state.get_parameters(mdl), LR)

   print(f"\nModel Parameters: {compress(sum(prod(p.shape) for p in params), ['k','m','b'])}\n") # type: ignore

   CHUNK_SIZE = BS*(cfg.ctx_length+1)

   @TinyJit
   @Tensor.train()
   def step(chk:Tensor) -> Tensor:
      opt.zero_grad()
      loss = mdl(chk[:,:-1]).sparse_categorical_crossentropy(chk[:,1:]).backward()
      opt.step()
      return loss

   while (i := ctr.loop_start()) is not None:

      with ctr.time_block("train"):
         Tensor.manual_seed(i)
         loss = step(Tensor(np.asarray(X_train[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE]), dtype=dtypes.int32).reshape(BS, -1))

      ctr.print_step(loss.item(), timings=True)


if __name__ == "__main__":
   train()
