from tinygrad import Tensor, nn, Variable, dtypes, TinyJit, Device
from tinygrad.nn.state import get_parameters
from tinygrad.helpers import prod, BEAM, Context
from extra.models.llama import TransformerBlock, Attention, precompute_freqs_cis, apply_rotary_emb, repeat_kv # type: ignore

from typing import List, Dict, Union, Optional, Tuple
import datetime, os, time
import matplotlib.pyplot as plt
import numpy as np

TOKEN_DIMS   = 512
CLUSTER_SIZE = 4
CLUSTER_DIMS = 2048

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
      # self.cluster_embed = nn.Linear(self.model_dim, self.model_dim)
      # self.layers = [nn.Linear(self.model_dim, self.model_dim) for _ in range(n_layers)]
      # self.out_proj = nn.Linear(self.model_dim, cluster_dim)

   def __call__(self, tokens:Tensor) -> Tensor:
      h = self.tok_embeddings(tokens)

      # B, T, _ = h.shape
      # assert T % self.cluster_size == 0
      # C = T // self.cluster_size
      # x = self.cluster_embed(h.reshape(B, C, self.model_dim))

      # for layer in self.layers: x = layer(x).silu()

      # return self.out_proj(x)

      return h

class Decoder:
   def __init__(self, vocab_size:int, max_context:int, n_layers:int, token_dim:int, cluster_dim:int, cluster_size:int, d_head:int, ff_mult:float=4.0, rope_theta:int=10000):
      self.model_dim = token_dim * cluster_size
      self.cluster_size = cluster_size

      # self.in_proj = nn.Linear(cluster_dim, self.model_dim)
      # self.layers = [nn.Linear(self.model_dim, self.model_dim) for _ in range(n_layers)]
      self.output = nn.Linear(token_dim, vocab_size, bias=False)

   def __call__(self, c:Tensor) -> Tensor:
      # x = self.in_proj(c)
      # B, C, _ = x.shape
      # T = C * self.cluster_size

      # for layer in self.layers: x = layer(x).silu()

      # h = x.reshape(B, T, self.model_dim // self.cluster_size)
      # logits = self.output(h)
      # return logits
   

      logits = self.output(c)
      return logits



def main():
   BEAM_VALUE = BEAM.value
   BEAM.value = 0

   MULTI_GPU = False
   TRAIN_DTYPE = dtypes.float32

   # Define Models
   VOCAB_SIZE = 32000
   D_HEAD = 32
   layers = {
      "enc": 8,
      "dec": 8,
   }
   enc = Encoder(VOCAB_SIZE, MAX_CLUSTER_CONTEXT+1, layers["enc"], TOKEN_DIMS, CLUSTER_DIMS, CLUSTER_SIZE, D_HEAD, ff_mult=2.0)
   dec = Decoder(VOCAB_SIZE, MAX_CLUSTER_CONTEXT+1, layers["dec"], TOKEN_DIMS, CLUSTER_DIMS, CLUSTER_SIZE, D_HEAD, ff_mult=2.0)

   # Load Dataset
   X_train, X_val = [np.memmap(f"/raid/datasets/fineweb/tokenized/fineweb_{split}.bin", dtype=np.uint16, mode='r') for split in ('train', 'val')]

   GPUS = [f"{Device.DEFAULT}:{i}" for i in range(6)]
   if not MULTI_GPU:
      GPUS =  GPUS[:1]

   params = []
   counts = {}
   MULT = 1.0 / 1024 / 1024 / 1024
   print("\nModel Parameters:")
   for name, model in {
      "enc":enc,
      "dec":dec
   }.items():
      model_params = get_parameters(model)
      for w in model_params:
         if MULTI_GPU:
            w.replace(w.cast(TRAIN_DTYPE).shard(GPUS, axis=None)).realize()
         else:
            w.replace(w.cast(TRAIN_DTYPE)).realize()
      params += model_params
      counts[name] = sum(prod(w.shape) for w in model_params)
      print(f"{name}: {counts[name] * MULT:.3f} B")
   print(f"all: {sum(counts.values()) * MULT:.3f} B")
   print("")

   # Define the Optimizer
   LEARNING_RATE = 2e-10
   optim = nn.optim.Adam(params, LEARNING_RATE)

   # Define some Globals
   DEVICE_BS = 32
   GLOBAL_BS = DEVICE_BS * len(GPUS)
   TOKENS_CONTEXT_SIZE = (MAX_CLUSTER_CONTEXT + 1) * CLUSTER_SIZE

   GRAPH_EVERY = 100


   # Define some Tracking Variables
   weights_folder = f"weights/{os.path.basename(os.path.dirname(__file__))}/{datetime.datetime.now()}".replace(" ", "_").replace(":", "_").replace("-", "_").replace(".", "_")
   train_losses = { "a": [5.0] }
   train_losses.pop("a")

   # @TinyJit
   def train_step(orig_tokens:Tensor) -> Tuple[Tensor,Dict[str,Tensor],Tensor]:
      enc_clusters = enc(orig_tokens).realize()
      # prd_clusters = gen(enc_clusters[:, :-1]).realize()
      dec_tokens   = dec(enc_clusters).realize()
      # prd_tokens   = dec(prd_clusters).realize()

      losses = {
         "decoded": dec_tokens.sparse_categorical_crossentropy(orig_tokens).realize(),
      }
      loss = losses["decoded"].realize() # sum(losses.values()).realize()
      optim.zero_grad()
      loss.backward()
      optim.step()

      acc = (dec_tokens.argmax(axis=-1) == orig_tokens).mean().realize()

      return loss, losses, acc

   step_i = 0
   dataset_i = 0
   with Tensor.train():
      while True:
         start_time = time.time()
         Tensor.manual_seed(step_i)

         with Context(BEAM=BEAM_VALUE):
            orig_batches = [Tensor(np.asarray(X_train[dataset_i + batch_i*TOKENS_CONTEXT_SIZE :dataset_i + (batch_i+1)*TOKENS_CONTEXT_SIZE]), dtype=dtypes.int32) for batch_i in range(GLOBAL_BS)]
            orig_tokens = Tensor.stack(*orig_batches)
            if MULTI_GPU:
               orig_tokens = orig_tokens.shard(GPUS, axis=0)
            loss, losses, acc = train_step(orig_tokens.realize())

            delta_time = time.time() - start_time
            print(f"| {step_i:05d} | {1000.0*delta_time:.0f} ms | {loss.item():.4f} Train Loss | {100.0*acc.item():.2f}% Train Acc |")

         for k,v in losses.items():
            if k not in train_losses:
               train_losses[k] = []
            train_losses[k].append(v.numpy().item())

         step_i += 1
         dataset_i += TOKENS_CONTEXT_SIZE * GLOBAL_BS

         if step_i > 0 and step_i % GRAPH_EVERY == 0:
            plt.clf()
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