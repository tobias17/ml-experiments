from tinygrad import Tensor, nn

class ModelConfig:
   n_layers: int = 8
   dim: int = 512
   vocab_size: int = 32000
   ctx_length: int = 1024

   n_heads: int = 16
   @property
   def head_d(self) -> int: return self.dim // self.n_heads

   ff_mult: float = 4.0
   @property
   def ff_dim(self) -> int: return int(self.dim * self.ff_mult)
   @property
   def ff_act(self): return Tensor.leakyrelu

class Attention:
   def __init__(self, cfg:ModelConfig):
      self.proj_in  = nn.Linear(cfg.dim, cfg.dim*3)
      self.proj_out = nn.Linear(cfg.dim, cfg.dim)
      self.n_heads  = cfg.n_heads
      self.head_d   = cfg.head_d
   def __call__(self, x:Tensor) -> Tensor:
      q,k,v = self.proj_in(x).chunk(3)
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
