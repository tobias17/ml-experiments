from tinygrad.tensor import Tensor # type: ignore
from tinygrad.nn import Embedding, Linear, LayerNorm # type: ignore
from tinygrad.nn.optim import Adam # type: ignore
from tinygrad.nn.state import get_parameters, get_state_dict, load_state_dict, safe_save, safe_load # type: ignore
from tinygrad.helpers import dtypes # type: ignore
import numpy as np
from config import Config
from util import write_graph, Schedules
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
      if Config.schedule == Schedules.LINEAR:
         a[i] = 1.0 - (i / (T-1))
      elif Config.schedule == Schedules.SQRT:
         a[i] = 1.0 - (i / (T-1))**0.5
      else:
         raise NotImplementedError()
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
   TS = Config.model_params.timesteps

   s_time = time.time()
   step, test_index = 0, 0
   train_loss, test_loss = [], []
   train_accs, test_accs = [[] for _ in range(4)], [[] for _ in range(4)]
   while True:
      np.random.seed(step if test_index == 0 else 1337)
      data = X_train if test_index <= 1 else X_test
      
      index = np.random.randint(0, len(data)-CS, size=BS)
      max_diffuse = math.ceil(TS / Config.timestep_delta)
      diff_start_index  = np.random.randint(1, Config.model_params.max_context - max_diffuse - 1)
      diff_start_amount = np.random.randint(1, TS - 1) if test_index==0 else Config.timestep_delta - 1
      diff_ladder_size  = 2 + math.floor((TS - diff_start_amount - 1) / Config.timestep_delta)

      amnt = diff_start_index + diff_ladder_size
      X_tok = np.zeros((BS,CS))
      X_tok[:,:amnt] = np.array([data[i:i+amnt] for i in index], dtype=np.float32)

      alphas = np.ones((BS,CS), dtype=np.float32)
      timesteps = np.zeros((BS,CS), dtype=np.float32)
      for i in range(diff_ladder_size):
         ts = min(diff_start_amount + i*Config.timestep_delta, TS-1)
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

      if test_index == 0:
         loss.realize()
         opt.zero_grad()
         loss.backward()
         opt.step()
      else:
         loss_l, accs_l = (test_loss, test_accs) if test_index==2 else (train_loss, train_accs)
         loss_l.append(loss.numpy().item())
         for i in range(4):
            accs_l[i].append((output[:,i:i+1].argmax(axis=-1)==Y[:,i:i+1]).mean().numpy().item())

      if (step+1) % Config.train.test_every == 0:
         if test_index == 2:
            step += 1
            tc = 4
            print(f"Step {str(step): >5} | Train Loss: {train_loss[-1]:.4f} | Train Accuracy: {100.0*sum(train_accs[i][-1] for i in range(tc))/tc:.2f}% | Test Loss: {test_loss[-1]:.4f} | Test Accuracy: {100.0*sum(test_accs[i][-1] for i in range(tc))/tc:.2f}% | {(time.time() - s_time) / float(Config.train.test_every):.2f} sec/iter")
            write_graph(train_loss, test_loss, f"{weights_folder}/graph_loss.png")
            write_graph(train_accs, test_accs, f"{weights_folder}/graph_acc.png", ylim=(0,1), segmented=True)
            s_time = time.time()
            test_index = 0
         else:
            test_index += 1
      else:
         step += 1

      if step % Config.train.save_every == 0:
         if not os.path.exists(weights_folder):
            os.makedirs(weights_folder)
         safe_save(get_state_dict(model), os.path.join(weights_folder, Config.save_name.format(step)))
         config_filepath = f"{weights_folder}/config.py"
         if not os.path.exists(config_filepath):
            shutil.copyfile(f"{os.path.dirname(__file__)}/config.py", config_filepath)
         main_filepath = f"{weights_folder}/{os.path.basename(__file__)}"
         if not os.path.exists(main_filepath):
            shutil.copyfile(__file__, main_filepath)

      if step % Config.train.gen_every == 0:
         g_time = time.time()
         text = generate(Config.train.gen_count, use_trange=True, model=model)
         gen_folder = f"{weights_folder}/gens"
         if not os.path.exists(gen_folder):
            os.makedirs(gen_folder)
         with open(f"{gen_folder}/text_{step}.txt", "w") as f:
            f.write(text)
         s_time += (time.time() - g_time)

def generate(count=20, timestep_reduce=100, use_trange=True, model=None, start="\n", archive=False):
   load_train_test()
   all_alphas = make_alphas()
   if model is None:
      model = Transformer(**Config.model_params.to_dict())
      root = f"weights/{os.path.basename(os.path.dirname(__file__))}" if not archive else f"archive/{os.path.basename(os.path.dirname(os.path.dirname(__file__)))}"
      last_folder = [f"{root}/{f}" for f in os.listdir(root)]
      last_folder = max([f for f in last_folder if os.path.isdir(f)], key=os.path.getmtime)
      last_weight = [f"{last_folder}/{f}" for f in os.listdir(last_folder) if f.startswith("model_")]
      last_weight = max(last_weight, key=os.path.getmtime)
      print(f"Using {last_weight}")
      load_state_dict(model, safe_load(last_weight))

   BS = 1
   CS = Config.model_params.max_context
   TS = Config.model_params.timesteps
   all_output = start

   x_0 = model.make_x_0_from(Tensor(encode(all_output), dtype=dtypes.float32, requires_grad=False).reshape(1,-1).pad( ((0,0), (0,CS-len(all_output))) ))
   diff_start_index = 1
   diff_start_amount = Config.model_params.timesteps - 1

   for i in (trange(count) if use_trange else range(count)):

      diff_ladder_size = 2 + math.floor((TS - diff_start_amount - 1) / Config.timestep_delta)
      while diff_start_index + diff_ladder_size > CS:
         amnt = (diff_start_index + diff_ladder_size) - CS
         x_0_np = x_0.shrink( ((0,BS), (amnt,CS), (0,x_0.shape[2])) ).pad( ((0,0), (0,amnt), (0,0)) ).numpy()
         del x_0
         x_0 = Tensor(x_0_np, dtype=dtypes.float32, requires_grad=False)
         diff_start_index -= 1

      alphas = np.ones((BS,CS), dtype=np.float32)
      timesteps = np.zeros((BS,CS), dtype=np.float32)
      for i in range(diff_ladder_size):
         ts = min(diff_start_amount + i*Config.timestep_delta, TS-1)
         alphas[:,diff_start_index+i] = all_alphas[int(ts)]
         timesteps[:,diff_start_index+i] = ts
      alphas = Tensor(alphas, dtype=dtypes.float32, requires_grad=False).reshape(BS,CS,1)

      x_t = x_0*alphas + Tensor.randn(BS,CS,Config.model_params.latent_dim)*(1-alphas)
      e_t = model(x_t, Tensor(timesteps, dtype=dtypes.float32, requires_grad=False))
      x_0_np = x_0.numpy()
      del x_0
      x_0_np[:,diff_start_index:diff_start_index+diff_ladder_size] = (x_t[:,diff_start_index:diff_start_index+diff_ladder_size] - e_t[:,diff_start_index:diff_start_index+diff_ladder_size]).numpy()
      x_0 = Tensor(x_0_np, requires_grad=False)

      while diff_start_amount < timestep_reduce:
         pred = model.estimate(x_0[:,diff_start_index:diff_start_index+1])
         all_output += decode([pred.argmax(axis=-1).numpy().item()])[0]
         x_0[:,diff_start_index:diff_start_index+1] = model.make_x_0_from(pred.argmax(axis=-1).reshape(1,1))
         del pred
         diff_start_index += 1
         diff_start_amount += Config.timestep_delta
      diff_start_amount -= timestep_reduce

   return all_output

text = """SEBASTIAN:
Bate, I beseech you, widow Dido.

ANTONIO:
"""

if __name__ == "__main__":
   train()
   # print(generate(count=128, start=text, timestep_reduce=20))
