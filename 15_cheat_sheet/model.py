from tinygrad import Tensor, TinyJit, dtypes, nn
from typing import List, Tuple
from tinygrad.helpers import prod
from dataclasses import dataclass
from util import compress

@dataclass
class ModelConfig:
   n_layers: int = 20
   dim: int = 768
   vocab_size: int = 32000
   ctx_length: int = 512

   cross_attn: bool = False

   n_heads: int = 16
   @property
   def head_d(self) -> int: return self.dim // self.n_heads

   ff_mult: float = 4.0
   @property
   def ff_dim(self) -> int: return int(self.dim * self.ff_mult)
   @property
   def ff_act(self): return Tensor.leaky_relu

class SelfAttention:
   def __init__(self, cfg:ModelConfig, is_causal:bool=True):
      self.proj_in   = nn.Linear(cfg.dim, cfg.dim*3)
      self.proj_out  = nn.Linear(cfg.dim, cfg.dim)
      self.n_heads   = cfg.n_heads
      self.head_d    = cfg.head_d
      self.is_causal = is_causal
   def __call__(self, x:Tensor) -> Tensor:
      q,k,v = self.proj_in(x).chunk(3, dim=-1)
      q,k,v = [y.rearrange('b c (h d) -> b h c d', h=self.n_heads, d=self.head_d) for y in (q,k,v)]
      h = Tensor.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal)
      h = h.rearrange('b h c d -> b c (h d)')
      return x + self.proj_out(h)

class CrossAttention:
   def __init__(self, cfg:ModelConfig):
      self.proj_q   = nn.Linear(cfg.dim, cfg.dim)
      self.proj_kv  = nn.Linear(cfg.dim, cfg.dim*2)
      self.proj_out = nn.Linear(cfg.dim, cfg.dim)
      self.n_heads  = cfg.n_heads
      self.head_d   = cfg.head_d
   def __call__(self, x:Tensor, z:Tensor) -> Tensor:
      q   = self.proj_q(x)
      k,v = self.proj_kv(z).chunk(2, dim=-1)
      q,k,v = [y.rearrange('b c (h d) -> b h c d', h=self.n_heads, d=self.head_d) for y in (q,k,v)]
      h = Tensor.scaled_dot_product_attention(q, k, v, is_causal=False)
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
      self.x_attn     = SelfAttention(cfg)
      self.z_attn     = SelfAttention(cfg, is_causal=False) if cfg.cross_attn else None
      self.cross_attn = CrossAttention(cfg) if cfg.cross_attn else None
      self.ff         = FeedForward(cfg)
   def __call__(self, x:Tensor, z:Tensor|None) -> Tuple[Tensor,Tensor|None]:
      x = self.x_attn(x)
      if (self.z_attn is not None) and (self.cross_attn is not None):
         assert z is not None
         z = self.z_attn(z)
         x = self.cross_attn(x, z)
      x = self.ff(x)
      return x, z

class Model:
   def __init__(self, cfg:ModelConfig):
      self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
      self.pos_emb = nn.Embedding(cfg.ctx_length, cfg.dim)
      self.layers = [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
      self.proj_out = nn.Linear(cfg.dim, cfg.vocab_size)
      self.cross_attn = cfg.cross_attn
   def __call__(self, tok:Tensor, ctx:Tensor|None) -> Tensor:
      x = self.tok_emb(tok) + self.pos_emb(Tensor.arange(tok.shape[-1], device=tok.device))
      z = self.tok_emb(ctx) + self.pos_emb(Tensor.arange(ctx.shape[-1], device=tok.device)) if (ctx is not None) else None
      for i, layer in enumerate(self.layers):
         x, z = layer(x, z)
      return self.proj_out(x)
   @property
   def device(self) -> str|Tuple[str,...]:
      return self.tok_emb.weight.device


