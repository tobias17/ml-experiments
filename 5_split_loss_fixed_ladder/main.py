from tinygrad.tensor import Tensor # type: ignore
from tinygrad.nn import Embedding, Linear, LayerNorm # type: ignore
from tinygrad.nn.optim import Adam # type: ignore
from tinygrad.nn.state import get_parameters, get_state_dict, load_state_dict, safe_save, safe_load # type: ignore
from tinygrad.helpers import dtypes, all_int, prod # type: ignore
import numpy as np
from config import Config
from util import write_graph, Schedules
from typing import Dict, Optional, List
import time, datetime, os, shutil, math
from tqdm import tqdm, trange # type: ignore

# copied from tensor.py
def scaled_dot_product_attention(self:Tensor, key:Tensor, value:Tensor, attn_mask:Optional[Tensor]=None, dropout_p:float=0.0, is_causal:bool=False) -> Tensor:
   assert all_int(self.shape), f"does not support symbolic shape {self.shape}"
   if is_causal: attn_mask = Tensor.ones(self.shape[-2], key.shape[-2], requires_grad=False, device=self.device).tril(0).cast(dtypes.bool)
   if attn_mask is not None and attn_mask.dtype == dtypes.bool: attn_mask = (attn_mask == 0).where(-float("inf"), attn_mask)
   # if not is_causal and attn_mask is not None:
   #    print("\n\nBefore attention mask")
   #    print((self @ key.transpose(-2,-1) / math.sqrt(self.shape[-1])).numpy())
   #    print("\n\nAfter attention mask")
   #    print((self @ key.transpose(-2,-1) / math.sqrt(self.shape[-1]) + attn_mask).numpy())
   #    z = 0
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

class CtxTransformer:
   def __init__(self, vocab_size:int, pos_size:int, n_layers:int, embed_dim:int, n_heads:int, ff_dim:int):
      self.tok_embed = Embedding(vocab_size, embed_dim)
      self.pos_embed = Embedding(pos_size, embed_dim)
      self.layers = [CrossAttentionBlock(embed_dim, embed_dim, n_heads, embed_dim//n_heads, ff_dim, is_causal=True, cross_attn=False) for _ in range(n_layers)]
      self.class_head = Linear(embed_dim, vocab_size)

   def get_parameters(self, pretraining:bool=False) -> List[Tensor]:
      params = []
      params += get_parameters(self.tok_embed)
      params += get_parameters(self.pos_embed)
      params += get_parameters(self.layers)
      if pretraining:
         params += get_parameters(self.class_head)
      return params

   def make_context_from(self, tok:Tensor) -> Tensor:
      x = self.tok_embed(tok) + self.pos_embed(Tensor.arange(0, tok.shape[1], requires_grad=False).reshape((1,-1)))
      return x.sequential(self.layers)
   
   def estimate(self, x:Tensor) -> Tensor:
      return self.class_head(x).log_softmax()

class DenTransformer:
   def __init__(self, vocab_size:int, timesteps:int, time_deltas:int, pos_size:int, n_layers:int, ctx_dim:int, embed_dim:int, latent_dim:int, timepos_dim:int, n_heads:int, ff_dim:int):
      self.latent_dim = latent_dim
      self.tok_embed  = Embedding(vocab_size, latent_dim)
      self.tok_norm   = LayerNorm(latent_dim)
      self.pos_embed  = Embedding(pos_size, timepos_dim)
      self.time_embed = Embedding(timesteps, timepos_dim)
      self.layers = [CrossAttentionBlock(embed_dim, ctx_dim, n_heads, embed_dim//n_heads, ff_dim) for _ in range(n_layers)]
      self.class_head = Linear(latent_dim, vocab_size)
   
   def get_parameters(self) -> List[Tensor]:
      return get_parameters(self)

   def make_x_0_from(self, tok:Tensor) -> Tensor:
      x = self.tok_embed(tok.reshape((-1,tok.shape[-1])))
      x = x.reshape((*tok.shape,x.shape[-1]))
      return self.tok_norm(x)

   def __call__(self, den_latent:Tensor, ctx_latent:Tensor, timesteps:Tensor, attn_mask:Tensor) -> Tensor:
      den_timepos = self.time_embed(timesteps.reshape(1,timesteps.shape[0])) + self.pos_embed(Tensor.arange(0, timesteps.shape[0], requires_grad=False).reshape((1,-1)))
      den_latent = den_latent.cat(den_timepos.reshape((1,1,*den_timepos.shape[-2:])).expand((*den_latent.shape[:2],*den_timepos.shape[-2:])), dim=-1)

      ctx_latent = ctx_latent.reshape((ctx_latent.shape[0],1,*ctx_latent.shape[1:])).expand((ctx_latent.shape[0],den_latent.shape[1],*ctx_latent.shape[1:])).reshape((-1,*ctx_latent.shape[1:]))
      orig_den_shape = den_latent.shape
      den_latent = den_latent.reshape((-1,*den_latent.shape[-2:]))

      for layer in self.layers:
         den_latent = layer(den_latent, ctx_latent, attn_mask)
      
      den_latent = den_latent.reshape(orig_den_shape)
      B,T,D,C = den_latent.shape
      assert C > self.latent_dim
      return den_latent.shrink( ((0,B), (0,T), (0,D), (0,self.latent_dim)) )

   def estimate(self, x:Tensor) -> Tensor:
      return self.class_head(x).log_softmax()


def make_alphas(show=False) -> np.ndarray:
   T = Config.den_model_params.timesteps
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
   with open(Config.dataset) as f:
      all_text = f.read()
   chars = list(set(all_text))
   chars = sorted(chars)
   c_to_t = { c:i for i,c in enumerate(chars) }
   t_to_c = { v:k for k,v in c_to_t.items() }
   global encode, decode
   encode = lambda chars: [c_to_t[c] for c in chars]
   decode = lambda toks:  [t_to_c[t] for t in toks ]

   tokens = encode(all_text)
   split_i = int(Config.split * len(tokens))
   return tokens[:split_i], tokens[split_i:]

def get_latest_folder(archive:bool=False) -> str:
   root = f"weights/{os.path.basename(os.path.dirname(__file__))}" if not archive else f"archive/{os.path.basename(os.path.dirname(os.path.dirname(__file__)))}"
   last_folders = [f"{root}/{f}" for f in os.listdir(root)]
   last_folder = max([f for f in last_folders if os.path.isdir(f)], key=os.path.getmtime)
   return last_folder

def load_latest_weight(model, model_type, archive=False, phase:Optional[int]=None):
   last_folder = get_latest_folder(archive)
   last_weights = [f"{last_folder}/{f}" for f in os.listdir(last_folder) if model_type in f and f.endswith("safetensors") and (phase is None or f.startswith(f"p{phase}"))]
   last_weight = max(last_weights, key=os.path.getmtime)
   print(f"Using {last_weight}")
   load_state_dict(model, safe_load(last_weight))

def train(phase:int, sigma:float=0.01):
   assert phase in (options:=[1,2,3]), f"phase was {phase}, must be in {options}"

   ctx_model = CtxTransformer(**Config.ctx_model_params.to_dict())
   den_model = DenTransformer(**Config.den_model_params.to_dict())

   type_name = os.path.basename(os.path.dirname(__file__))
   if phase == 1:
      weights_folder = f"weights/{type_name}/{datetime.datetime.now()}".replace(" ", "_").replace(":", "_").replace("-", "_").replace(".", "_")
   else:
      weights_folder = get_latest_folder()
      if len([f for f in os.listdir(weights_folder) if f.startswith(f"p{phase}_")]):
         while True:
            text = input(f"Files already detected for phase {phase}, continue? [y/n]: ")
            if text.lower().startswith("y"):
               break
            elif text.lower().startswith("n"):
               raise RuntimeError("Avoiding overwriting previous run")

   if phase == 2 or phase == 3:
      load_latest_weight(ctx_model, "ctx", phase=1)
   if phase == 3:
      load_latest_weight(den_model, "den", phase=2)

   params = []
   if phase == 1 or phase == 3:
      params += ctx_model.get_parameters(phase == 1)
   if phase == 2 or phase == 3:
      params += den_model.get_parameters()
   opt = Adam(params, Config.train[phase].learning_rate)

   all_alphas = make_alphas()
   X_train, X_test = load_train_test()

   BS = Config.train[phase].batch_size
   TS = Config.den_model_params.timesteps
   CS = Config.ctx_model_params.pos_size
   DS = Config.den_model_params.pos_size

   TD = Config.den_model_params.time_deltas

   s_time = time.time()
   step, test_index = 0, 0
   train_loss:List[float] = []
   test_loss: List[float] = []
   if phase == 1:
      train_acc:List[float] = []
      test_acc: List[float] = []
   else:
      train_accs:List[List[float]] = [[] for _ in range(DS)]
      test_accs: List[List[float]] = [[] for _ in range(DS)]
   while True:
      np.random.seed(step if test_index < 2 else 1337)
      data = X_train if test_index <= 1 else X_test
      
      index = np.random.randint(0, len(data)-CS-DS, size=BS)

      X_tok = np.array([data[i:i+CS] for i in index], dtype=np.float32)
      context = ctx_model.make_context_from(Tensor(X_tok, dtype=dtypes.float32, requires_grad=False))

      if phase == 1:
         Y_tok = np.array([data[i+1:i+1+CS] for i in index], dtype=np.float32)
         Y = Tensor(Y_tok, dtype=dtypes.float32)
         output = ctx_model.estimate(context)
         loss = output.sparse_categorical_crossentropy(Y)
      else:
         Y_tok = np.array([[data[i+j:i+j+DS] for j in range(1,CS+1)] for i in index], dtype=np.float32)
         Y = Tensor(Y_tok, dtype=dtypes.float32)

         diff_start_amount = np.random.randint(1, TD+1) if test_index==0 else TD // 2

         alphas = np.ones((DS,), dtype=np.float32)
         timesteps = np.zeros((DS,), dtype=np.float32)
         for i in range(DS):
            ts = min(diff_start_amount + i*TD, TS - 1)
            alphas[i] = all_alphas[int(ts)]
            timesteps[i] = ts
         alphas = (1 - ((1 - alphas) * (1+np.abs((sigma*np.random.randn(*alphas.shape)))))).clip(0, 1)
         alphas = Tensor(alphas, dtype=dtypes.float32, requires_grad=False).reshape(1,1,DS,1).expand(BS,CS,DS,Config.den_model_params.latent_dim)

         attn_mask = Tensor.ones(CS,CS).tril(0).cast(dtypes.bool).reshape(1,CS,1,CS).expand(BS,CS,DS,CS).reshape(-1,DS,CS)
         x_0 = den_model.make_x_0_from(Y.detach())
         x_t = x_0*alphas + ((1-alphas)*Tensor.randn(*x_0.shape)).detach()

         pred_x_0 = den_model(x_t, context, Tensor(timesteps, dtype=dtypes.float32, requires_grad=False), attn_mask)
         output = den_model.estimate(pred_x_0)

         loss_1 = (pred_x_0 - x_0).pow(2).sum() / prod(pred_x_0.shape)
         loss_2 = den_model.estimate(x_0).sparse_categorical_crossentropy(Y)
         loss_3 = output.sparse_categorical_crossentropy(Y)
         loss = loss_1 + loss_2 + loss_3

      if test_index == 0:
         loss.realize()
         opt.zero_grad()
         loss.backward()
         opt.step()
      else:
         loss_l = test_loss if test_index==2 else train_loss
         loss_l.append(loss.numpy().item())
         
         if phase == 1:
            acc_l = test_acc if test_index==2 else train_acc
            acc_l.append((output.argmax(axis=-1)==Y).mean().numpy().item())
         else:
            accs_l = test_accs if test_index==2 else train_accs
            for i in range(DS):
               accs_l[i].append((output[:,:,i:i+1].argmax(axis=-1)==Y[:,:,i:i+1]).mean().numpy().item())

      TE = Config.train[phase].test_every
      if (step+1) % TE == 0:
         if test_index == 2:
            step += 1
            if phase == 1:
               train_acc_str, test_acc_str = f"{100.0*train_acc[-1]:.2f}", f"{100.0*test_acc[-1]:.2f}"
            else:
               train_acc_str, test_acc_str = f"{100.0*sum(train_accs[i][-1] for i in range(DS))/DS:.2f}", f"{100.0*sum(test_accs[i][-1] for i in range(DS))/DS:.2f}"
            print(f"Step {str(step): >5} | Train Loss: {train_loss[-1]:.4f} | Train Accuracy: {train_acc_str}% | Test Loss: {test_loss[-1]:.4f} | Test Accuracy: {test_acc_str}% | {(time.time() - s_time) / float(TE):.2f} sec/iter")
            write_graph(train_loss, test_loss, f"{weights_folder}/p{phase}_graph_loss.png", delta=TE)
            if phase == 1:
               write_graph(train_acc,  test_acc,  f"{weights_folder}/p{phase}_graph_acc.png", ylim=(0,1), delta=TE)
            else:
               write_graph(train_accs, test_accs, f"{weights_folder}/p{phase}_graph_acc.png", ylim=(0,1), segmented=True, delta=TE)
            s_time = time.time()
            test_index = 0
         else:
            test_index += 1
      else:
         step += 1

      if step % Config.train[phase].save_every == 0:
         if not os.path.exists(weights_folder):
            os.makedirs(weights_folder)
         if phase == 1 or phase == 3:
            safe_save(get_state_dict(ctx_model), os.path.join(weights_folder, Config.ctx_save_name.format(phase, step)))
         if phase == 2 or phase == 3:
            safe_save(get_state_dict(den_model), os.path.join(weights_folder, Config.den_save_name.format(phase, step)))
         config_filepath = f"{weights_folder}/config.py"
         if not os.path.exists(config_filepath):
            shutil.copyfile(f"{os.path.dirname(__file__)}/config.py", config_filepath)
         main_filepath = f"{weights_folder}/{os.path.basename(__file__)}"
         if not os.path.exists(main_filepath):
            shutil.copyfile(__file__, main_filepath)

      if step % Config.train[phase].gen_every == 0:
         g_time = time.time()
         if phase == 1:
            text = generate_ctx(Config.train[phase].gen_count, ctx_model=ctx_model)
         else:
            text = generate_den(Config.train[phase].gen_count, ctx_model=ctx_model, den_model=den_model)
         gen_folder = f"{weights_folder}/p{phase}_gens"
         if not os.path.exists(gen_folder):
            os.makedirs(gen_folder)
         with open(f"{gen_folder}/text_{step}.txt", "w") as f:
            f.write(text)
         s_time += (time.time() - g_time)

text = """SEBASTIAN:
Bate, I beseech you, widow Dido.

ANTONIO:
"""

def generate_ctx(count=20, ctx_model=None, start=text, archive=False):
   load_train_test()
   if ctx_model is None:
      ctx_model = CtxTransformer(**Config.ctx_model_params.to_dict())
      load_latest_weight(ctx_model, "ctx", archive)

   CS = Config.ctx_model_params.pos_size
   all_output = start

   X = Tensor(encode(all_output), dtype=dtypes.float32, requires_grad=False).reshape(1,-1).pad( ((0,0), (0,CS-len(all_output))) )
   for i in trange(count):
      assert X.shape == (1,CS,)
      pull_i = min(i, CS-1)
      context = ctx_model.make_context_from(X.realize())
      pred = ctx_model.estimate(context)[:,pull_i:pull_i+1].argmax(axis=-1)
      char = decode([pred.numpy().item()])[0]
      all_output += char

      X_np = np.zeros((1,CS+1))
      X_np[:,:-1] = X.numpy()
      if i + 1 < CS:
         X_np[:,i+1:i+2] = pred.numpy()
         X = Tensor(X_np[:,:-1], dtype=dtypes.float32, requires_grad=False)
      else:
         X_np[:,-1:] = pred.numpy()
         X = Tensor(X_np[:,1:], dtype=dtypes.float32, requires_grad=False)
      del context, pred
   
   del X
   return all_output

def generate_den(count=20, timestep_reduce=25, ctx_model=None, den_model=None, start=text, archive=False):
   global encode, decode
   load_train_test()
   all_alphas = make_alphas()
   if ctx_model is None:
      ctx_model = CtxTransformer(**Config.ctx_model_params.to_dict())
      load_latest_weight(ctx_model, "ctx", archive)
   if den_model is None:
      den_model = DenTransformer(**Config.den_model_params.to_dict())
      load_latest_weight(den_model, "den", archive)

   BS = 1
   TS = Config.den_model_params.timesteps
   CS = Config.ctx_model_params.pos_size
   DS = Config.den_model_params.pos_size
   TD = Config.den_model_params.time_deltas
   all_output = start

   def make_context(toks, start_i):
      if len(toks) > CS:
         toks = toks[-CS:]
      data = np.zeros((CS,))-1
      data[:start_i] = np.array(encode(toks[:start_i]))
      data_i = start_i
      context = ctx_model.make_context_from(Tensor(data, dtype=dtypes.float32, requires_grad=False).reshape(1,-1))
      return context, data_i
   
   def make_x_0(toks, start_i: int):
      assert start_i < len(toks)
      start_data = np.array(encode(toks[start_i:])) # type: ignore
      return den_model.make_x_0_from(Tensor(start_data, dtype=dtypes.float32, requires_grad=False).reshape(BS,1,-1))

   start_i = len(all_output) - DS
   assert start_i > 0, f"input size {len(all_output)} must be atleast 1 greater than decoder head size {DS}"
   x_0 = make_x_0(all_output, start_i)
   z_0 = x_0
   den_start_amount = timestep_reduce
   context, data_i = make_context(all_output, start_i)

   for i in trange(count):

      alphas = np.ones((DS,), dtype=np.float32)
      timesteps = np.zeros((DS,), dtype=np.float32)
      for i in range(DS):
         ts = min(den_start_amount + i*TD, TS-1)
         alphas[i] = all_alphas[int(ts)]
         timesteps[i] = ts
      alphas = Tensor(alphas, dtype=dtypes.float32, requires_grad=False).reshape(1,1,DS,1)
      attn_mask = Tensor.ones(CS,CS).tril(0).cast(dtypes.bool).reshape(1,CS,1,CS).expand(BS,CS,DS,CS)[:,data_i-1:data_i,:,:].reshape(-1,DS,CS)

      x_t = x_0*alphas + Tensor.randn(BS,1,Config.den_model_params.latent_dim)*(1-alphas)
      e_t = den_model(x_t, context, Tensor(timesteps, dtype=dtypes.float32, requires_grad=False), attn_mask)
      pred_x_0 = x_t - e_t

      while den_start_amount <= timestep_reduce:
         if start_i >= len(all_output):
            pred = den_model.estimate(pred_x_0)[0,0,0,:]
            all_output += decode([pred.argmax(axis=-1).numpy().item()])[0]
         start_i += 1
         context, data_i = make_context(all_output, start_i)
         pred_x_0 = pred_x_0[:,:,1:].cat(Tensor.zeros(*pred_x_0.shape[:2],1,pred_x_0.shape[-1]), dim=-2)
         den_start_amount += TD
      den_start_amount -= timestep_reduce

      if start_i < len(all_output):
         pred_x_0[:,:,:len(all_output)-start_i] = make_x_0(all_output, start_i).detach()

      x_0 = pred_x_0.realize().detach()

   return all_output

if __name__ == "__main__":
   # train(phase=1)
   # print(generate_ctx(count=512))

   # train(phase=2)
   train(phase=3)
   # print(generate_den(count=128))
