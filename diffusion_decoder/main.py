from tinygrad.tensor import Tensor # type: ignore
from tinygrad.nn import Embedding, Linear, LayerNorm # type: ignore
from tinygrad.nn.optim import Adam # type: ignore
from tinygrad.nn.state import get_parameters, get_state_dict, load_state_dict, safe_save, safe_load # type: ignore
from tinygrad.helpers import dtypes # type: ignore
import numpy as np
from config import Config
from util import write_graph
from typing import Dict
import time, datetime, os, shutil, math
from tqdm import tqdm, trange # type: ignore

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
   def __init__(self, timesteps: int, vocab_size: int, max_context: int, n_layers: int, embed_dim: int, latent_dim: int, timepos_dim: int, n_heads: int, ff_dim: int):
      self.latent_dim, self.timepos_dim = latent_dim, timepos_dim
      
      self.tok_embed = [Embedding(vocab_size, latent_dim), LayerNorm(latent_dim)]
      self.pos_embed = Embedding(max_context, timepos_dim)
      self.time_embed = Embedding(timesteps, timepos_dim)

      self.n_layers = [TransformerBlock(embed_dim, n_heads, ff_dim) for _ in range(n_layers)]

      self.class_head = Linear(latent_dim, vocab_size)

   def make_x_0_from(self, tok:Tensor) -> Tensor:
      return tok.sequential(self.tok_embed)

   def __call__(self, latent:Tensor, timesteps:Tensor) -> Tensor:
      assert latent.shape[2] == self.latent_dim
      timepos = self.time_embed(timesteps) + self.pos_embed(Tensor.arange(0, latent.shape[1], requires_grad=False).reshape((1,-1)))
      x = latent.cat(timepos, dim=-1)
      x = x.sequential(self.n_layers)
      B,T,C = x.shape
      assert C > self.latent_dim
      return x.shrink( ((0,B), (0,T), (0,self.latent_dim)) )

   def estimate(self, latent:Tensor) -> Tensor:
      return self.class_head(latent).log_softmax()

def make_alphas(show=False) -> np.ndarray:
   T = Config.model_params.timesteps
   a = np.zeros((T,), dtype=np.float32)
   for i in range(T):
      a[i] = 1.0 - (i / (T-1))
   if show:
      import matplotlib.pyplot as plt
      plt.plot(np.arange(T), a)
      plt.show()
   return a

def load_train_test():
   with open(Config.train.dataset) as f:
      all_text = f.read()
   chars = list(set(all_text))
   chars = sorted(chars)
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
   opt = Adam(get_parameters(model), Config.train.learning_rate)
   all_alphas = make_alphas()
   X_train, X_test = load_train_test()
   type_name = os.path.basename(os.path.dirname(__file__))
   weights_folder = f"weights/{type_name}/{datetime.datetime.now()}".replace(" ", "_").replace(":", "_").replace("-", "_").replace(".", "_")

   BS = Config.train.batch_size
   CS = Config.model_params.max_context

   s_time = time.time()
   step, is_test = 0, False
   train_loss, test_loss = [], []
   train_acc,  test_acc  = [], []
   while True:
      if not is_test:
         data  = X_train
         np.random.seed(step)
      else:
         data  = X_test
         np.random.seed(1337)
      
      index = np.random.randint(0, len(data)-CS, size=BS)
      max_diffuse = math.ceil(Config.model_params.timesteps / Config.timestep_delta)
      diff_start_index  = np.random.randint(1, Config.model_params.max_context - max_diffuse - 1)
      diff_start_amount = np.random.randint(1, Config.model_params.timesteps - 1)
      diff_ladder_size  = 1 + math.floor((Config.model_params.timesteps - diff_start_amount) / Config.timestep_delta)

      amnt = diff_start_index + diff_ladder_size
      X_tok = np.zeros((BS,CS))
      X_tok[:,:amnt] = np.array([data[i:i+amnt] for i in index], dtype=np.float32)

      alphas = np.ones((BS,CS), dtype=np.float32)
      timesteps = np.zeros((BS,CS), dtype=np.float32)
      for i in range(diff_ladder_size):
         ts = diff_start_amount + i*Config.timestep_delta
         alphas[:,diff_start_index+i] = all_alphas[int(ts)]
         timesteps[:,diff_start_index+i] = ts
      alphas = Tensor(alphas, dtype=dtypes.float32, requires_grad=False).reshape(BS,CS,1)

      x_0 = model.make_x_0_from(Tensor(X_tok, dtype=dtypes.float32, requires_grad=False))
      x_t = x_0*alphas + ((1-alphas)*Tensor.randn(BS,CS,Config.model_params.latent_dim)).detach()

      e_t = model(x_t, Tensor(timesteps, dtype=dtypes.float32, requires_grad=False))
      pred_x_0 = x_t[:,diff_start_index:diff_start_index+diff_ladder_size] - e_t[:,diff_start_index:diff_start_index+diff_ladder_size]
      output = model.estimate(pred_x_0)

      Y = Tensor([data[i+diff_start_index:i+amnt] for i in index], dtype=dtypes.float32).reshape(BS,-1)
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
            te = Config.train.test_every
            print(f"Step {str(step): >5} | Train Loss: {sum(train_loss[-te:])/te:.4f} | Train Accuracy: {100.0*sum(train_acc[-te:])/te:.2f}% | Test Loss: {test_loss[-1]:.4f} | Test Accuracy: {100.0*test_acc[-1]:.2f}% | {(time.time() - s_time) / float(te):.2f} sec/iter")
            write_graph(train_loss, test_loss, f"{weights_folder}/graph_loss.png")
            write_graph(train_acc,  test_acc,  f"{weights_folder}/graph_acc.png", ylim=(0,1))
            s_time = time.time()
         is_test = not is_test
      else:
         step += 1

      if step % Config.train.gen_every == 0:
         g_time = time.time()
         text = generate(Config.train.gen_count, False, True, model)
         gen_folder = f"{weights_folder}/gens"
         if not os.path.exists(gen_folder):
            os.makedirs(gen_folder)
         with open(f"{gen_folder}/text_{step}.txt", "w") as f:
            f.write(text)
         s_time += (time.time() - g_time)

      if step % Config.train.save_every == 0:
         if not os.path.exists(weights_folder):
            os.makedirs(weights_folder)
         safe_save(get_state_dict(model), os.path.join(weights_folder, Config.save_name.format(step)))
         config_filepath = f"{weights_folder}/config.py"
         if not os.path.exists(config_filepath):
            shutil.copyfile(f"{os.path.dirname(__file__)}/config.py", config_filepath)

def generate(count=20, print_output=True, use_trange=False, model=None):
   load_train_test()
   if model is None:
      model = Transformer(**Config.model_params.to_dict())
      root = f"weights/{os.path.basename(os.path.dirname(__file__))}"
      last_folder = [f"{root}/{f}" for f in os.listdir(root)]
      last_folder = max([f for f in last_folder if os.path.isdir(f)], key=os.path.getmtime)
      last_weight = [f"{last_folder}/{f}" for f in os.listdir(last_folder) if f.startswith("model_")]
      last_weight = max(last_weight, key=os.path.getmtime)
      print(f"Using {last_weight}")
      load_state_dict(model, safe_load(last_weight))

   CONTEXT = Config.model_params.max_context
   output = ""
   all_output = ""

   X = Tensor(encode("\n"), dtype=dtypes.float32, requires_grad=False).reshape(1,-1).pad( ((0,0), (0,CONTEXT-1)) )
   for i in (trange(count) if use_trange else range(count)):
      assert X.shape == (1,CONTEXT,)
      pull_i = min(i, CONTEXT-1)
      pred = model(X.realize())[:,pull_i:pull_i+1].argmax(axis=-1)
      char = decode([pred.numpy().item()])[0]
      if char == "\n":
         if print_output: print(output)
         output = ""
      else:
         output += char
      all_output += char

      X_np = np.zeros((1,CONTEXT+1))
      X_np[:,:-1] = X.numpy()
      if i + 1 < CONTEXT:
         X_np[:,i+1:i+2] = pred.numpy()
         X = Tensor(X_np[:,:-1], dtype=dtypes.float32, requires_grad=False)
      else:
         X_np[:,-1:] = pred.numpy()
         X = Tensor(X_np[:,1:], dtype=dtypes.float32, requires_grad=False)
   
   if print_output and output: print(output)
   return all_output

if __name__ == "__main__":
   train()
