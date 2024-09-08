from tinygrad import Tensor, nn
from tinygrad.nn.state import get_parameters
from tinygrad.helpers import prod
from extra.models.llama import TransformerBlock, precompute_freqs_cis # type: ignore

from sentencepiece import SentencePieceProcessor

TOKEN_DIMS   = 256
CLUSTER_SIZE = 8
CLUSTER_DIMS = 1024

MAX_CLUSTER_CONTEXT = 32
MAX_TOKEN_CONTEXT = int(MAX_CLUSTER_CONTEXT * CLUSTER_SIZE)

NORM_EPS = 1e-5

class Encoder:
   def __init__(self, vocab_size:int, max_context:int, n_layers:int, token_dim:int, cluster_dim:int, cluster_size:int, d_head:int, ff_mult:float=4.0, rope_theta:int=10000):
      self.model_dim = token_dim * cluster_size
      self.cluster_size = cluster_size

      self.tok_embeddings = nn.Embedding(vocab_size, token_dim)
      self.cluster_embed = nn.Linear(self.model_dim, self.model_dim)
      self.layers = [TransformerBlock(self.model_dim, int(self.model_dim*ff_mult), self.model_dim // d_head, None, NORM_EPS, max_context) for _ in range(n_layers)]
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
   def __init__(self, vocab_size:int, max_context:int, n_layers:int, token_dim:int, cluster_dim:int, cluster_size:int, d_head:int, ff_mult:float=4.0, rope_theta:int=10000):
      self.model_dim = token_dim * cluster_size
      self.cluster_size = cluster_size
      
      self.in_proj = nn.Linear(cluster_dim, self.model_dim)
      self.layers = [TransformerBlock(self.model_dim, int(self.model_dim*ff_mult), self.model_dim // d_head, None, NORM_EPS, max_context) for _ in range(n_layers)]
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
   def __init__(self, max_context:int, n_layers:int, dim:int, d_head:int, ff_mult:float=4.0, rope_theta:int=10000):
      self.layers = [TransformerBlock(dim, int(dim*ff_mult), dim // d_head, None, NORM_EPS, max_context) for _ in range(n_layers)]
      self.freqs_cis = precompute_freqs_cis(d_head, max_context*2, rope_theta).contiguous()
   
   def __call__(self, x:Tensor) -> Tensor:
      C = x.shape[1]
      mask = Tensor.full((1, 1, C, C), float("-inf"), dtype=x.dtype, device=x.device).triu(1).realize()
      for layer in self.layers: x = layer(x, 0, self.freqs_cis, mask)
      return x

def main():
   VOCAB_SIZE = 32000
   D_HEAD = 32

   layers = {
      "enc": 4,
      "gen": 48,
      "dec": 4,
   }

   enc = Encoder  (VOCAB_SIZE, MAX_CLUSTER_CONTEXT+1, layers["enc"], TOKEN_DIMS, CLUSTER_DIMS, CLUSTER_SIZE, D_HEAD)
   gen = Generator(            MAX_CLUSTER_CONTEXT,   layers["gen"],             CLUSTER_DIMS,               D_HEAD)
   dec = Decoder  (VOCAB_SIZE, MAX_CLUSTER_CONTEXT+1, layers["dec"], TOKEN_DIMS, CLUSTER_DIMS, CLUSTER_SIZE, D_HEAD)

   counts = {}
   MULT = 1.0 / 1024 / 1024 / 1024
   for name, model in { "enc":enc, "gen":gen, "dec":dec }.items():
      counts[name] = sum(prod(w.shape) for w in get_parameters(model))
      print(f"{name}: {counts[name] * MULT:.3f} B")
   print(f"all: {sum(counts.values()) * MULT:.3f} B")

if __name__ == "__main__":
   main()
