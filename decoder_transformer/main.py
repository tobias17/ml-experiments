from tinygrad.tensor import Tensor # type: ignore
from tinygrad.nn import Embedding, Linear # type: ignore
from tinygrad.nn.optim import Adam # type: ignore
from tinygrad.nn.state import get_parameters, get_state_dict, safe_save # type: ignore
from tinygrad.helpers import dtypes # type: ignore
import numpy as np
from config import Config
from util import write_graph
from typing import Dict
import time, datetime, os

class TransformerBlock:
   def __init__(self, embed_dim, num_heads, ff_dim, act=lambda x: x.relu(), dropout=0.1):
      assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

      self.num_heads = num_heads
      self.head_size = embed_dim // num_heads
      self.act = act
      self.dropout = dropout

      self.query = (Tensor.scaled_uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))
      self.key   = (Tensor.scaled_uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))
      self.value = (Tensor.scaled_uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))

      self.out = (Tensor.scaled_uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))

      self.ff1 = (Tensor.scaled_uniform(embed_dim, ff_dim), Tensor.zeros(ff_dim))
      self.ff2 = (Tensor.scaled_uniform(ff_dim, embed_dim), Tensor.zeros(embed_dim))

      self.ln1 = (Tensor.ones(embed_dim), Tensor.zeros(embed_dim))
      self.ln2 = (Tensor.ones(embed_dim), Tensor.zeros(embed_dim))

   def attn(self, x:Tensor) -> Tensor:
      # x: (bs, time, embed_dim) -> (bs, time, embed_dim)
      query, key, value = [x.linear(*y).reshape(shape=(x.shape[0], -1, self.num_heads, self.head_size)).transpose(1,2) for y in [self.query, self.key, self.value]]
      attention = Tensor.scaled_dot_product_attention(query, key, value, is_causal=True).transpose(1,2)
      return attention.reshape(shape=(x.shape[0], -1, self.num_heads * self.head_size)).linear(*self.out)

   def __call__(self, x:Tensor) -> Tensor:
      x = x + self.attn(x).dropout(self.dropout)
      x = x.layernorm().linear(*self.ln1)
      x = x + self.act(x.linear(*self.ff1)).linear(*self.ff2).dropout(self.dropout)
      x = x.layernorm().linear(*self.ln2)
      return x

class Transformer:
   def __init__(self, vocab_size: int, max_context: int, layers: int, embed_dim: int, n_heads: int, ff_dim: int):
      self.tok_embed = Embedding(vocab_size,  embed_dim)
      self.pos_embed = Embedding(max_context, embed_dim)
      self.layers = [TransformerBlock(embed_dim, n_heads, ff_dim) for _ in range(layers)]
      self.class_head = Linear(embed_dim, vocab_size)
   def __call__(self, x:Tensor) -> Tensor:
      x = self.tok_embed(x) + self.pos_embed(Tensor.arange(0, x.shape[1], requires_grad=False).reshape((1,-1)))
      x = x.sequential(self.layers)
      return self.class_head(x).log_softmax()

def load_train_test():
   with open(Config.train.dataset) as f:
      all_text = f.read()
   chars = set(all_text)
   c_to_t = { c:i for i,c in enumerate(chars) }
   t_to_c = { v:k for k,v in c_to_t.items() }
   global encode, decode
   encode = lambda chars: [c_to_t[c] for c in chars]
   decode = lambda toks:  [t_to_c[t] for t in toks ]

   tokens = encode(all_text)
   split_i = int(Config.train.split * len(tokens))
   return tokens[:split_i], tokens[split_i:]

def train():
   model = Transformer(**Config.model_params.to_dict())
   opt = Adam(get_parameters(model), Config.train.lr)
   X_train, X_test = load_train_test()
   weights_folder = f"weights/{datetime.datetime.now()}".replace(" ", "_").replace(":", "_").replace("-", "_").replace(".", "_")

   s_time = time.time()
   step, is_test = 0, False
   train_loss, test_loss = [], []
   train_acc,  test_acc  = [], []
   train_context = Config.model_params.max_context
   while True:
      if not is_test:
         np.random.seed(step)
         data = X_train
         index = np.random.randint(0, len(X_train)-train_context)
      else:
         data = X_test
         index = 0

      X = Tensor(X_train[index  :index+train_context  ], dtype=dtypes.float32, requires_grad=False).reshape(1,-1)
      Y = Tensor(X_train[index+1:index+train_context+1], dtype=dtypes.float32).reshape(1,-1)
      
      output = model(X)
      loss = output.sparse_categorical_crossentropy(Y)
      acc = (output.argmax(axis=-1)==Y).mean()

      loss_l, acc_l = (test_loss, test_acc) if is_test else (train_loss, train_acc)
      loss_l.append(loss.numpy().item())
      acc_l.append(acc.numpy().item())

      if not is_test:
         loss.realize()
         opt.zero_grad()
         loss.backward()
         opt.step()

      if (step+1) % Config.train.test_every == 0:
         if is_test:
            step += 1
            write_graph(train_loss, test_loss, f"{weights_folder}/graph_loss.png")
            write_graph(train_acc,  test_acc,  f"{weights_folder}/graph_acc.png")
            te = Config.train.test_every
            print(f"Step {str(step): >5} | Train Loss: {sum(train_loss[-te:])/te:.4f} | Train Accuracy: {100.0*sum(train_acc[-te:])/te:.2f}% | Test Loss: {test_loss[-1]:.4f} | Test Accuracy: {100.0*test_acc[-1]:.2f}% | {(time.time() - s_time) / float(te):.2f} sec/iter")
            s_time = time.time()
         is_test = not is_test
      else:
         step += 1

      if step % Config.train.save_every == 0:
         if not os.path.exists(weights_folder):
            os.makedirs(weights_folder)
         safe_save(get_state_dict(model), os.path.join(weights_folder, Config.save_name.format(step)))

if __name__ == "__main__":
   train()
