from tinygrad.tensor import Tensor # type: ignore
from tinygrad.nn import Embedding, Linear, LayerNorm # type: ignore
from tinygrad.nn.optim import Adam # type: ignore
from tinygrad.nn.state import get_parameters, get_state_dict, load_state_dict, safe_save, safe_load # type: ignore
from tinygrad.helpers import dtypes, all_int # type: ignore
import numpy as np
from config import Config
from util import write_graph, Schedules
from typing import Dict, Optional
import time, datetime, os, shutil, math
from tqdm import tqdm, trange # type: ignore

# copied from tensor.py
def scaled_dot_product_attention(self:Tensor, key:Tensor, value:Tensor, attn_mask:Optional[Tensor]=None, dropout_p:float=0.0, is_causal:bool=False) -> Tensor:
   assert all_int(self.shape), f"does not support symbolic shape {self.shape}"
   if is_causal: attn_mask = Tensor.ones(self.shape[-2], key.shape[-2], requires_grad=False, device=self.device).tril(0).cast(dtypes.bool)
   if attn_mask is not None and attn_mask.dtype == dtypes.bool: attn_mask = (attn_mask == 0).where(-float("inf"), attn_mask)
   a = self @ key.transpose(-2,-1) / math.sqrt(self.shape[-1])
   b = a + attn_mask
   return (self @ key.transpose(-2,-1) / math.sqrt(self.shape[-1]) + attn_mask).softmax(-1).dropout(dropout_p) @ value

class CrossAttention:
   def __init__(self, query_dim, context_dim, n_heads, d_head, dropout=0.1, is_causal=False):
      self.to_q = Linear(query_dim,   n_heads*d_head, bias=False)
      self.to_k = Linear(context_dim, n_heads*d_head, bias=False)
      self.to_v = Linear(context_dim, n_heads*d_head, bias=False)
      self.num_heads = n_heads
      self.head_size = d_head
      self.to_out = [Linear(n_heads*d_head, query_dim)]
      self.dropout = dropout
      self.is_causal = is_causal
   def __call__(self, x:Tensor, context:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None) -> Tensor:
      context = x if context is None else context
      q,k,v = self.to_q(x), self.to_k(context), self.to_v(context)
      q,k,v = [y.reshape(*x.shape[0:-2], -1, self.num_heads, self.head_size).transpose(-3,-2) for y in (q,k,v)]
      if attn_mask is not None: attn_mask = attn_mask.reshape(attn_mask.shape[0], 1, *attn_mask.shape[1:]).expand((attn_mask.shape[0], self.num_heads, *attn_mask.shape[1:]))
      attention = scaled_dot_product_attention(q, k, v, is_causal=self.is_causal, attn_mask=attn_mask).dropout(self.dropout).transpose(-3,-2)
      h_ = attention.reshape(shape=(*x.shape[0:-2], -1, self.num_heads * self.head_size))
      return h_.sequential(self.to_out)

class GEGLU:
   def __init__(self, dim_in, dim_out):
      self.proj = Linear(dim_in, dim_out * 2)
      self.dim_out = dim_out
   def __call__(self, x:Tensor) -> Tensor:
      x, gate = self.proj(x).chunk(2, dim=-1)
      return x * gate.gelu()

class FeedForward:
   def __init__(self, dim, mult=4):
      self.net = [
         GEGLU(dim, dim*mult),
         Linear(dim*mult, dim),
      ]
   def __call__(self, x:Tensor) -> Tensor:
      return x.sequential(self.net)

class CrossAttentionBlock:
   def __init__(self, dim, context_dim, n_heads, d_head, ff_dim, dropout=0.1, is_causal=False, cross_attn=True):
      self.norm1 = LayerNorm(dim)
      self.attn1 = CrossAttention(dim, dim, n_heads, d_head, is_causal=is_causal)
      self.cross_attn = cross_attn
      if self.cross_attn:
         self.norm2 = LayerNorm(dim)
         self.attn2 = CrossAttention(dim, context_dim, n_heads, d_head, is_causal=is_causal)
      self.norm3 = LayerNorm(dim)
      self.ff    = FeedForward(dim)
      self.dropout = dropout
   def __call__(self, x, context:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None) -> Tensor:
      x = self.attn1(self.norm1(x)) + x
      if self.cross_attn:
         x = self.attn2(self.norm2(x), context=context, attn_mask=attn_mask) + x
      x = self.ff(self.norm3(x)).dropout(self.dropout) + x
      return x

class Transformer:
   def __init__(self,
      vocab_size:int, timesteps:int, time_deltas:int, n_layers:int,
      ctx_dim:int, ctx_heads:int, ctx_ff_dim:int, ctx_size:int,
      den_dim:int, latent_dim:int, timepos_dim:int, den_heads:int, den_ff_dim:int, den_size:int
   ):
      self.latent_dim, self.timepos_dim = latent_dim, timepos_dim

      self.ctx_tok_embed = Embedding(vocab_size, ctx_dim)
      self.ctx_pos_embed = Embedding(ctx_size, ctx_dim)

      self.den_tok_embed = [Embedding(vocab_size, latent_dim), LayerNorm(latent_dim)]
      self.den_pos_embed = Embedding(den_size, timepos_dim)
      self.den_time_embed = Embedding(timesteps, timepos_dim)

      self.layers = [
         [
            CrossAttentionBlock(ctx_dim, ctx_dim, ctx_heads, ctx_dim//ctx_heads, ctx_ff_dim, is_causal=True, cross_attn=False),
            CrossAttentionBlock(den_dim, ctx_dim, den_heads, den_dim//den_heads, den_ff_dim),
         ] for _ in range(n_layers)
      ]

      self.class_head = Linear(latent_dim, vocab_size)

   def make_context_from(self, tok:Tensor) -> Tensor:
      return self.ctx_tok_embed(tok) + self.ctx_pos_embed(Tensor.arange(0, tok.shape[1], requires_grad=False).reshape((1,-1)))

   def make_x_0_from(self, tok:Tensor) -> Tensor:
      x = self.den_tok_embed[0](tok.reshape((-1,tok.shape[-1])))
      x = x.reshape((*tok.shape,x.shape[-1]))
      x = self.den_tok_embed[1](x)
      return x

   def __call__(self, den_latent:Tensor, ctx_latent:Tensor, timesteps:Tensor, attn_mask:Tensor) -> Tensor:
      den_timepos = self.den_time_embed(timesteps.reshape(1,timesteps.shape[0])) + self.den_pos_embed(Tensor.arange(0, timesteps.shape[0], requires_grad=False).reshape((1,-1)))
      den_latent = den_latent.cat(den_timepos.reshape((1,1,*den_timepos.shape[-2:])).expand((*den_latent.shape[:2],*den_timepos.shape[-2:])), dim=-1)

      for ctx_block, den_block in self.layers:
         ctx_latent = ctx_block(ctx_latent)
         a = den_latent.reshape((-1,*den_latent.shape[-2:]))
         b = ctx_latent.reshape((ctx_latent.shape[0],1,*ctx_latent.shape[1:])).expand((ctx_latent.shape[0],den_latent.shape[1],*ctx_latent.shape[1:])).reshape((-1,*ctx_latent.shape[1:]))
         c = attn_mask
         d = den_block(a, b, c)
         e = d.reshape(den_latent.shape)
         den_latent = e

      B,D,T,C = den_latent.shape
      assert C > self.latent_dim
      return den_latent.shrink( ((0,B), (0,D), (0,T), (0,self.latent_dim)) )

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
   TS = Config.model_params.timesteps
   CS = Config.model_params.ctx_size
   DS = Config.model_params.den_size

   s_time = time.time()
   step, test_index = 0, 0
   train_loss, test_loss = [], []
   train_accs, test_accs = [[] for _ in range(DS)], [[] for _ in range(DS)]
   while True:
      np.random.seed(step if test_index == 0 else 1337)
      data = X_train if test_index <= 1 else X_test
      
      index = np.random.randint(0, len(data)-CS-DS, size=BS)
      diff_start_amount = np.random.randint(1, TS - 1) if test_index==0 else Config.model_params.time_deltas // 2

      X_tok = np.array([data[i:i+CS] for i in index], dtype=np.float32)
      Y_tok = np.array([[data[i+j:i+j+DS] for j in range(CS)] for i in index], dtype=np.float32)
      Y = Tensor(Y_tok, dtype=dtypes.float32)

      alphas = np.ones((DS,), dtype=np.float32)
      timesteps = np.zeros((DS,), dtype=np.float32)
      for i in range(DS):
         ts = min(diff_start_amount + i*Config.model_params.time_deltas, TS-1)
         alphas[i] = all_alphas[int(ts)]
         timesteps[i] = ts
      alphas = Tensor(alphas, dtype=dtypes.float32, requires_grad=False).reshape(1,1,DS,1).expand(BS,CS,DS,1)

      attn_mask = Tensor.ones(CS,CS).tril(0).cast(dtypes.bool).reshape(1,CS,1,CS).expand(BS,CS,DS,CS).reshape(-1,DS,CS)
      context = model.make_context_from(Tensor(X_tok, dtype=dtypes.float32, requires_grad=False))
      x_0 = model.make_x_0_from(Y.detach())
      x_t = x_0*alphas + ((1-alphas)*Tensor.randn(*x_0.shape)).detach()

      e_t = model(x_t, context, Tensor(timesteps, dtype=dtypes.float32, requires_grad=False), attn_mask)
      pred_x_0 = x_t - e_t
      output = model.estimate(pred_x_0)
      loss = output.sparse_categorical_crossentropy(Y)

      if test_index == 0:
         loss.realize()
         opt.zero_grad()
         loss.backward()
         opt.step()
      else:
         loss_l, accs_l = (test_loss, test_accs) if test_index==2 else (train_loss, train_accs)
         loss_l.append(loss.numpy().item())
         for i in range(DS):
            accs_l[i].append((output[:,i:i+1].argmax(axis=-1)==Y[:,i:i+1]).mean().numpy().item())

      if (step+1) % Config.train.test_every == 0:
         if test_index == 2:
            step += 1
            print(f"Step {str(step): >5} | Train Loss: {train_loss[-1]:.4f} | Train Accuracy: {100.0*sum(train_accs[i][-1] for i in range(DS))/DS:.2f}% | Test Loss: {test_loss[-1]:.4f} | Test Accuracy: {100.0*sum(test_accs[i][-1] for i in range(DS))/DS:.2f}% | {(time.time() - s_time) / float(Config.train.test_every):.2f} sec/iter")
            write_graph(train_loss, test_loss, f"{weights_folder}/graph_loss.png", delta=Config.train.test_every)
            write_graph(train_accs, test_accs, f"{weights_folder}/graph_acc.png", ylim=(0,1), segmented=True, delta=Config.train.test_every)
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
         text = generate(Config.train.gen_count, model=model)
         gen_folder = f"{weights_folder}/gens"
         if not os.path.exists(gen_folder):
            os.makedirs(gen_folder)
         with open(f"{gen_folder}/text_{step}.txt", "w") as f:
            f.write(text)
         s_time += (time.time() - g_time)

text = """SEBASTIAN:
Bate, I beseech you, widow Dido.

ANTONIO:
"""

def generate(count=20, timestep_reduce=25, use_trange=True, model=None, start=text, archive=False):
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
   TS = Config.model_params.timesteps
   CS = Config.model_params.ctx_size
   DS = Config.model_params.den_size
   all_output = start

   def make_context(toks):
      if len(toks) > CS:
         toks = toks[-CS:]
      data = np.zeros((CS,))-1
      data[:len(toks)] = np.array(encode(toks))
      data_i = len(toks)       
      context = model.make_context_from(Tensor(data, dtype=dtypes.float32, requires_grad=False).reshape(1,-1))
      return context, data_i
   context, data_i = make_context(all_output)
   x_0 = Tensor.randn(BS,1,DS,Config.model_params.latent_dim)
   diff_start_amount = Config.model_params.timesteps - 1

   for i in (trange(count) if use_trange else range(count)):

      # while diff_start_index + diff_ladder_size > CS:
      #    amnt = (diff_start_index + diff_ladder_size) - CS
      #    x_0_np = x_0.shrink( ((0,BS), (amnt,CS), (0,x_0.shape[2])) ).pad( ((0,0), (0,amnt), (0,0)) ).numpy()
      #    del x_0
      #    x_0 = Tensor(x_0_np, dtype=dtypes.float32, requires_grad=False)
      #    diff_start_index -= 1

      alphas = np.ones((DS,), dtype=np.float32)
      timesteps = np.zeros((DS,), dtype=np.float32)
      for i in range(DS):
         ts = min(diff_start_amount + i*Config.model_params.time_deltas, TS-1)
         alphas[i] = all_alphas[int(ts)]
         timesteps[i] = ts
      alphas = Tensor(alphas, dtype=dtypes.float32, requires_grad=False).reshape(1,1,DS,1)
      attn_mask = Tensor.ones(CS,CS).tril(0).cast(dtypes.bool).reshape(1,CS,1,CS).expand(BS,CS,DS,CS)[:,data_i-1:data_i,:,:].reshape(-1,DS,CS)

      x_t = x_0*alphas + Tensor.randn(BS,1,Config.model_params.latent_dim)*(1-alphas)
      e_t = model(x_t, context, Tensor(timesteps, dtype=dtypes.float32, requires_grad=False), attn_mask)
      x_0 = (x_t - e_t).realize().detach()

      while diff_start_amount < timestep_reduce:
         pred = model.estimate(x_0)[0,0,0,:]
         all_output += decode([pred.argmax(axis=-1).numpy().item()])[0]
         context, data_i = make_context(all_output)
         diff_start_amount += Config.model_params.time_deltas
      diff_start_amount -= timestep_reduce

   return all_output

if __name__ == "__main__":
   train()
   # print(generate(count=512, timestep_reduce=50))
