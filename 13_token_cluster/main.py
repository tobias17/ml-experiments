from tinygrad import Tensor, nn
from examples.llama import TikToken # type: ignore
from extra.models.llama import TransformerBlock, precompute_freqs_cis # type: ignore

from typing import Callable

TOKEN_DIMS   = 768
CLUSTER_SIZE = 8
CLUSTER_DIMS = 2048

MAX_CLUSTER_CONTEXT = 32
MAX_TOKEN_CONTEXT = (MAX_CLUSTER_CONTEXT + 1) * CLUSTER_SIZE

NORM_EPS = 1e-5

class Encoder:
   def __init__(self, vocab_size:int, max_context:int, n_layers:int, token_dim:int, cluster_dim:int, cluster_size:int, d_head:int, ff_dim:int, rope_theta:int=10000):
      self.model_dim = token_dim * cluster_size
      self.cluster_size = cluster_size

      self.tok_embeddings = nn.Embedding(vocab_size, token_dim)
      self.cluster_embed = nn.Linear(self.model_dim, self.model_dim)
      self.layers = [TransformerBlock(self.model_dim, ff_dim, self.model_dim // d_head, None, NORM_EPS, max_context) for _ in range(n_layers)]
      self.out_proj = nn.Linear(self.model_dim, cluster_dim)
      self.freqs_cis = precompute_freqs_cis(d_head, max_context*2, rope_theta).contiguous()
   
   def __call__(self, tokens:Tensor) -> Tensor:
      h = self.tok_embeddings(tokens)

      B, T, _ = h.shape
      assert T % self.cluster_size == 0
      C = T // self.cluster_size
      x = self.cluster_embed(h.reshape(B, C, self.model_dim))

      mask = Tensor.full((1, 1, C, C), float("-inf"), dtype=x.dtype, device=x.device).triu(1).realize()
      for layer in self.layers: x = layer(x, 0, self.freqs_cis, mask)
      return self.out_proj(x)

class Decoder:
   def __init__(self, vocab_size:int, max_context:int, n_layers:int, token_dim:int, cluster_dim:int, cluster_size:int, d_head:int, ff_dim:int, rope_theta:int=10000):
      self.model_dim = token_dim * cluster_size
      self.cluster_size = cluster_size
      
      self.in_proj = nn.Linear(cluster_dim, self.model_dim)
      self.layers = [TransformerBlock(self.model_dim, ff_dim, self.model_dim // d_head, None, NORM_EPS, max_context) for _ in range(n_layers)]
      self.freqs_cis = precompute_freqs_cis(d_head, max_context*2, rope_theta).contiguous()
      self.output = nn.Linear(token_dim, vocab_size, bias=False)
   
   def __call__(self, c:Tensor) -> Tensor:
      x = self.in_proj(c)
      B, C, _ = x.shape
      T = C * self.cluster_size

      mask = Tensor.full((1, 1, C, C), float("-inf"), dtype=x.dtype, device=x.device).triu(1).realize()
      for layer in self.layers: x = layer(x, 0, self.freqs_cis, mask)
   
      logits = self.output(x.reshape(B, T, self.model_dim // self.cluster_size))
      return logits

class Generator:
   def __init__(self, max_context:int, n_layers:int, dim:int, d_head:int, ff_dim:int, rope_theta:int=10000):
      self.layers = [TransformerBlock(dim, ff_dim, dim // d_head, None, NORM_EPS, max_context)]
      self.freqs_cis = precompute_freqs_cis(d_head, max_context*2, rope_theta).contiguous()

def main():
   pass

if __name__ == "__main__":
   main()
