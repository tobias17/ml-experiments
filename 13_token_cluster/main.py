from tinygrad import Tensor, nn, Variable
from tinygrad.nn.state import get_parameters
from tinygrad.helpers import prod
from extra.models.llama import TransformerBlock, Attention, precompute_freqs_cis, apply_rotary_emb, repeat_kv # type: ignore

from sentencepiece import SentencePieceProcessor # type: ignore
from typing import List, Dict, Union, Optional
import datetime, os, time
import matplotlib.pyplot as plt
import numpy as np

TOKEN_DIMS   = 256
CLUSTER_SIZE = 8
CLUSTER_DIMS = 1024

MAX_CLUSTER_CONTEXT = 32

NORM_EPS = 1e-5

def __call__(self:Attention, x:Tensor, start_pos:Union[Variable,int], freqs_cis:Tensor, mask:Optional[Tensor]) -> Tensor:
   xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

   xq = xq.reshape(xq.shape[0], xq.shape[1], self.n_heads, self.head_dim)
   xk = xk.reshape(xk.shape[0], xk.shape[1], self.n_kv_heads, self.head_dim)
   xv = xv.reshape(xv.shape[0], xv.shape[1], self.n_kv_heads, self.head_dim)

   xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
   bsz, seqlen, _, _ = xq.shape

   keys = xk
   values = xv

   keys, values = repeat_kv(keys, self.n_rep), repeat_kv(values, self.n_rep)
   xq, keys, values = xq.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2)
   attn = xq.scaled_dot_product_attention(keys, values, mask).transpose(1, 2)
   attn = attn.reshape(bsz, seqlen, -1)
   return self.wo(attn)
Attention.__call__ = __call__

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

      freqs_cis = self.freqs_cis.shrink((None,(0,C),None,None,None))
      mask = Tensor.full((1, 1, C, C), float("-inf"), dtype=x.dtype, device=x.device).triu(1).realize()
      for layer in self.layers: x = layer(x, 0, freqs_cis, mask)
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

      freqs_cis = self.freqs_cis.shrink((None,(0,C),None,None,None))
      mask = Tensor.full((1, 1, C, C), float("-inf"), dtype=x.dtype, device=x.device).triu(1).realize()
      for layer in self.layers: x = layer(x, 0, freqs_cis, mask)
   
      logits = self.output(x.reshape(B, T, self.model_dim // self.cluster_size))
      return logits

class Generator:
   def __init__(self, max_context:int, n_layers:int, dim:int, d_head:int, ff_mult:float=4.0, rope_theta:int=10000):
      self.layers = [TransformerBlock(dim, int(dim*ff_mult), dim // d_head, None, NORM_EPS, max_context) for _ in range(n_layers)]
      self.freqs_cis = precompute_freqs_cis(d_head, max_context*2, rope_theta).contiguous()
   
   def __call__(self, x:Tensor) -> Tensor:
      C = x.shape[1]
      freqs_cis = self.freqs_cis.shrink((None,(0,C),None,None,None))
      mask = Tensor.full((1, 1, C, C), float("-inf"), dtype=x.dtype, device=x.device).triu(1).realize()
      for layer in self.layers: x = layer(x, 0, freqs_cis, mask)
      return x

def main():
   # Define Models
   VOCAB_SIZE = 32000
   D_HEAD = 32
   layers = {
      "enc": 6,
      "gen": 32,
      "dec": 6,
   }
   enc = Encoder  (VOCAB_SIZE, MAX_CLUSTER_CONTEXT+1, layers["enc"], TOKEN_DIMS, CLUSTER_DIMS, CLUSTER_SIZE, D_HEAD, ff_mult=2.0)
   gen = Generator(            MAX_CLUSTER_CONTEXT,   layers["gen"],             CLUSTER_DIMS,               D_HEAD, ff_mult=3.0)
   dec = Decoder  (VOCAB_SIZE, MAX_CLUSTER_CONTEXT+1, layers["dec"], TOKEN_DIMS, CLUSTER_DIMS, CLUSTER_SIZE, D_HEAD, ff_mult=2.0)

   # Compute and Print Parameter Counts
   params = []
   counts = {}
   MULT = 1.0 / 1024 / 1024 / 1024
   print("\nModel Parameters:")
   for name, model in { "enc":enc, "gen":gen, "dec":dec }.items():
      model_params = get_parameters(model)
      counts[name] = sum(prod(w.shape) for w in model_params)
      params += model_params
      print(f"{name}: {counts[name] * MULT:.3f} B")
   print(f"all: {sum(counts.values()) * MULT:.3f} B")
   print("")

   # Define the Optimizer
   LEARNING_RATE = 2e-13
   optim = nn.optim.AdamW(params, LEARNING_RATE)

   # Define some Globals
   GLOBAL_BS = 1
   TOKENS_CONTEXT_SIZE = (MAX_CLUSTER_CONTEXT + 1) * CLUSTER_SIZE

   GRAPH_EVERY = 20

   # Define some Tracking Variables
   weights_folder = f"weights/{os.path.basename(os.path.dirname(__file__))}/{datetime.datetime.now()}".replace(" ", "_").replace(":", "_").replace("-", "_").replace(".", "_")
   train_losses = { "a": [5.0] }
   train_losses.pop("a")

   step_i = 0
   dataset_i = 0
   with Tensor.train():
      while True:
         start_time = time.time()

         orig_tokens = Tensor.randint(GLOBAL_BS, TOKENS_CONTEXT_SIZE, low=0, high=VOCAB_SIZE)

         enc_clusters = enc(orig_tokens)
         prd_clusters = gen(enc_clusters[:, :-1])
         dec_tokens   = dec(enc_clusters)
         prd_tokens   = dec(prd_clusters)

         losses = {
            "cluster": (enc_clusters[:, 1:] - prd_clusters).square().mean().realize(),
            "decoded": dec_tokens.sparse_categorical_crossentropy(orig_tokens).realize(),
            "predict": prd_tokens.sparse_categorical_crossentropy(orig_tokens[:, CLUSTER_SIZE:]).realize(),
         }
         loss = sum(losses.values()).realize()
         optim.zero_grad()
         loss.backward()
         optim.step()

         acc = (prd_tokens.argmax(axis=-1) == orig_tokens[:, CLUSTER_SIZE:]).mean().numpy().item()

         delta_time = time.time() - start_time
         print(f"| {step_i:05d} | {1000.0*delta_time:.0f} ms | {loss.numpy().item():.4f} Train Loss | {100.0*acc:.2f}% Train Acc |")

         for k,v in losses.items():
            if k not in train_losses:
               train_losses[k] = []
            train_losses[k].append(v.numpy().item())

         step_i += 1
         dataset_i += GLOBAL_BS

         if step_i > 0 and step_i % GRAPH_EVERY == 0:
            x = np.arange(step_i)
            for label, y in train_losses.items():
               plt.plot(x, y, label=label)
            plt.ylim((0,None))
            plt.title("Loss")
            plt.legend()
            figure = plt.gcf()
            figure.set_size_inches(18/1.5, 10/1.5)
            if not os.path.exists(weights_folder): os.makedirs(weights_folder)
            plt.savefig(os.path.join(weights_folder, "graph_loss.png"), dpi=100)

if __name__ == "__main__":
   main()
