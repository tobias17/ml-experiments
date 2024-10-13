from torch import Tensor, nn, Module
from torch import optim as nn_optim
import time
import numpy as np

TOKEN_DIMS   = 512
CLUSTER_SIZE = 4
CLUSTER_DIMS = 2048

MAX_CLUSTER_CONTEXT = 32



class Encoder(Module):
   def __init__(self, vocab_size:int, max_context:int, n_layers:int, token_dim:int, cluster_dim:int, cluster_size:int, d_head:int, ff_mult:float=4.0, rope_theta:int=10000):
      super().__init__()
      self.model_dim = token_dim * cluster_size
      self.cluster_size = cluster_size

      self.tok_embeddings = nn.Embedding(vocab_size, token_dim)

   def __call__(self, tokens:Tensor) -> Tensor:
      h = self.tok_embeddings(tokens)
      return h

class Decoder(Module):
   def __init__(self, vocab_size:int, max_context:int, n_layers:int, token_dim:int, cluster_dim:int, cluster_size:int, d_head:int, ff_mult:float=4.0, rope_theta:int=10000):
      super().__init__()
      self.model_dim = token_dim * cluster_size
      self.cluster_size = cluster_size

      self.output = nn.Linear(token_dim, vocab_size, bias=False)

   def __call__(self, c:Tensor) -> Tensor:
      logits = self.output(c)
      return logits



def main():
   Tensor.manual_seed(42)
   Tensor.training = True

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

   GPUS =  ["CUDA"]

   params = []
   counts = {}
   MULT = 1.0 / 1024 / 1024 / 1024
   print("\nModel Parameters:")
   for name, model in {
      "enc": enc,
      "dec": dec,
   }.items():
      model_params = model.parameters()
      params += model_params
      counts[name] = sum(prod(w.shape) for w in model_params)
      print(f"{name}: {counts[name] * MULT:.3f} B")
   print(f"all: {sum(counts.values()) * MULT:.3f} B")
   print("")

   # Define the Optimizer
   LEARNING_RATE = 2e-8
   optim = nn_optim.SGD(params, LEARNING_RATE)

   # Define some Globals
   DEVICE_BS = 256
   GLOBAL_BS = DEVICE_BS * len(GPUS)
   TOKENS_CONTEXT_SIZE = MAX_CLUSTER_CONTEXT * CLUSTER_SIZE

   step_i = 0
   dataset_i = 0
   while True:
      start_time = time.time()
      Tensor.manual_seed(step_i)

      orig_batches = [Tensor(np.asarray(X_train[dataset_i + batch_i*TOKENS_CONTEXT_SIZE :dataset_i + (batch_i+1)*TOKENS_CONTEXT_SIZE])).long() for batch_i in range(GLOBAL_BS)]
      orig_tokens = Tensor.stack(*orig_batches)

      enc_clusters = enc(orig_tokens).realize()
      dec_tokens   = dec(enc_clusters).realize()

      loss = dec_tokens.sparse_categorical_crossentropy(orig_tokens).realize()
      optim.zero_grad()
      loss.backward()
      optim.step()

      acc = (dec_tokens.argmax(axis=-1) == orig_tokens).mean().realize()

      delta_time = time.time() - start_time
      print(f"| {step_i:05d} | {dataset_i:08d} | {1000.0*delta_time:.0f} ms | {loss.item():.4f} Train Loss | {100.0*acc.item():.2f}% Train Acc |")

      step_i += 1
      dataset_i += TOKENS_CONTEXT_SIZE * GLOBAL_BS

if __name__ == "__main__":
   main()
