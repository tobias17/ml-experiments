from tinygrad import Tensor, TinyJit, dtypes, nn
from typing import Tuple, Dict
from tinygrad.helpers import prod
from dataclasses import dataclass
from util import compress

@dataclass
class ModelConfig:
   n_layers: int = 20
   dim: int = 1024
   vocab_size: int = 32000
   ctx_length: int = 512
   norm_eps: float = 1e-5
   rope_theta: float = 500000

   cross_attn: bool = False

   n_heads: int = 32
   @property
   def head_d(self) -> int: return self.dim // self.n_heads

   n_kv_heads: int = 8
   @property
   def n_rep(self) -> int: return self.n_heads // self.n_kv_heads

   ff_mult: float = 3.0
   @property
   def ff_dim(self) -> int: return int(self.dim * self.ff_mult)
   @property
   def ff_act(self): return Tensor.silu

   def validate(self):
      assert self.head_d * self.n_heads == self.dim
      assert self.n_rep * self.n_kv_heads == self.n_heads

# https://github.com/facebookresearch/llama/blob/1076b9c51c77ad06e9d7ba8a4c6df775741732bd/llama/model.py#L47
class FreqCis:
   _cache: Dict[Tuple[int,int,float],Tensor] = {}
   @staticmethod
   def get(dim:int, end:int, theta:float) -> Tensor:
      key = (dim,end,theta)
      value = FreqCis._cache.get(key)
      if value is None:
         freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2)[:(dim // 2)] / dim))
         freqs = Tensor.arange(end).unsqueeze(dim=1) * freqs.unsqueeze(dim=0)
         value = Tensor.stack(freqs.cos(), freqs.sin(), dim=-1).reshape(1, end, 1, dim//2, 2)
         FreqCis._cache[key] = value
      return value

# matches meta, non hugging face weights
# (a+i*b) * (c+i*d) = (ac-bd) + i*(ad+bc)
def complex_mult(A, c, d):
   a,b = A[..., 0:1], A[..., 1:2]
   ro = a*c - b*d
   co = a*d + b*c
   return ro.cat(co, dim=-1)

def apply_rotary_emb(xq:Tensor, xk:Tensor, freqs_cis:Tensor) -> tuple[Tensor, Tensor]:
   assert freqs_cis.shape[1] == xq.shape[1] == xk.shape[1], f"freqs_cis shape mismatch {freqs_cis.shape} xq:{xq.shape} xk:{xk.shape}"
   xq = xq.reshape(*xq.shape[0:-1], -1, 2)
   xk = xk.reshape(*xk.shape[0:-1], -1, 2)
   assert len(xq.shape) == len(xk.shape) == len(freqs_cis.shape) == 5
   c, d = freqs_cis[..., 0:1], freqs_cis[..., 1:2]
   xq_out = complex_mult(xq, c, d)
   xk_out = complex_mult(xk, c, d)
   return xq_out.flatten(3), xk_out.flatten(3)

class Attention:
   def __init__(self, cfg:ModelConfig, is_causal:bool=True, cross_attn:bool=False):
      self.pre_norm_x = nn.RMSNorm(cfg.dim, cfg.norm_eps)
      self.pre_norm_z = nn.RMSNorm(cfg.dim, cfg.norm_eps) if cross_attn else None
      self.proj_q    = nn.Linear(cfg.dim, cfg.dim)
      self.proj_kv   = nn.Linear(cfg.dim, 2*cfg.n_kv_heads*cfg.head_d)
      self.proj_out  = nn.Linear(cfg.dim, cfg.dim)
      self.is_causal = is_causal
      self.n_rep = cfg.n_rep
      self.rara  = { "h":cfg.n_heads, "kvh":cfg.n_kv_heads, "d":cfg.head_d }
      self.head_d = cfg.head_d
   def __call__(self, x:Tensor, freqs_cis:Tensor, z:Tensor|None=None) -> Tensor:
      x = self.pre_norm_x(x)
      z = x if z is None else self.pre_norm_z(z) # type: ignore

      q = self.proj_q(x)
      k,v = self.proj_kv(z).chunk(2, dim=-1)

      q,k,v = [y.rearrange('b c (h d) -> b c h d', d=self.head_d) for y in (q,k,v)]
      q,k   = apply_rotary_emb(q, k, freqs_cis)
      k,v   = [y.repeat((1, 1, 1, self.n_rep)).rearrange('b c h (d r) -> b c (h r) d', r=self.n_rep) for y in (k,v)]
      q,k,v = [y.rearrange('b c h d -> b h c d') for y in (q,k,v)]

      h = Tensor.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal)
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
      self.x_attn     = Attention(cfg)
      self.z_attn     = Attention(cfg, is_causal=False) if cfg.cross_attn else None
      self.cross_attn = Attention(cfg, cross_attn=True) if cfg.cross_attn else None
      self.x_ff       = FeedForward(cfg)
      self.z_ff       = FeedForward(cfg) if cfg.cross_attn else None
   def __call__(self, x:Tensor, freqs_cis:Tensor, z:Tensor|None) -> Tuple[Tensor,Tensor|None]:
      x = self.x_attn(x, freqs_cis)
      if self.cross_attn is not None:
         z = self.z_ff(self.z_attn(z, freqs_cis)) # type: ignore
         x = self.cross_attn(x, freqs_cis, z)
      x = self.x_ff(x)
      return x, z

class Model:
   def __init__(self, cfg:ModelConfig):
      self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
      self.layers = [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
      self.proj_out = nn.Linear(cfg.dim, cfg.vocab_size)
      self.cross_attn = cfg.cross_attn
      self.freqs_cis_args = (cfg.head_d, cfg.ctx_length*2, cfg.rope_theta)
   def __call__(self, tok:Tensor, ctx:Tensor|None) -> Tensor:
      x = self.tok_emb(tok)
      z = self.tok_emb(ctx) if (ctx is not None) else None
      freqs_cis = FreqCis.get(*self.freqs_cis_args).cast(x.dtype).to(tok.device)[:,:int(tok.shape[1]),:,:,:]
      for layer in self.layers:
         x, z = layer(x, freqs_cis, z)
      return self.proj_out(x)
   @property
   def device(self) -> str|Tuple[str,...]:
      return self.tok_emb.weight.device
