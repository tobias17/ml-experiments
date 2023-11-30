from tinygrad.tensor import Tensor # type: ignore
from tinygrad.nn import Embedding, Linear, LayerNorm # type: ignore
from tinygrad.nn.optim import Adam # type: ignore
from tinygrad.nn.state import get_parameters, get_state_dict, load_state_dict, safe_save, safe_load # type: ignore
from tinygrad.helpers import dtypes, all_int, prod # type: ignore
import numpy as np
from config import Config
from util import write_graph, Schedules
from typing import Dict, Optional, List, Tuple, Union
import time, datetime, os, shutil, math
from tqdm import tqdm, trange # type: ignore

# copied from tensor.py
def scaled_dot_product_attention(self:Tensor, key:Tensor, value:Tensor, attn_mask:Optional[Tensor]=None, dropout_p:float=0.0, is_causal:bool=False) -> Tensor:
   assert all_int(self.shape), f"does not support symbolic shape {self.shape}"
   if is_causal: attn_mask = Tensor.ones(self.shape[-2], key.shape[-2], requires_grad=False, device=self.device).tril(0).cast(dtypes.bool)
   if attn_mask is not None and attn_mask.dtype == dtypes.bool: attn_mask = (attn_mask == 0).where(-float("inf"), attn_mask)
   return (self @ key.transpose(-2,-1) / math.sqrt(self.shape[-1]) + attn_mask).softmax(-1).dropout(dropout_p) @ value

class SelfAttention:
   def __init__(self, query_dim, n_heads, d_head, dropout=Config.dropout, is_causal=False):
      self.to_q = Linear(query_dim, n_heads*d_head, bias=False)
      self.to_k = Linear(query_dim, n_heads*d_head, bias=False)
      self.to_v = Linear(query_dim, n_heads*d_head, bias=False)
      self.num_heads = n_heads
      self.head_size = d_head
      self.to_out = [Linear(n_heads*d_head, query_dim)]
      self.dropout = dropout
      self.is_causal = is_causal
   def __call__(self, x:Tensor) -> Tuple[Tensor, Tensor, Tensor]:
      q,k,v = self.to_q(x), self.to_k(x), self.to_v(x)
      q,k,v = [y.reshape(x.shape[0], -1, self.num_heads, self.head_size).transpose(-3,-2) for y in (q,k,v)]
      attention = scaled_dot_product_attention(q, k, v, is_causal=self.is_causal).dropout(self.dropout).transpose(-3,-2)
      h_ = attention.reshape(shape=(*x.shape[0:-2], -1, self.num_heads * self.head_size))
      return h_.sequential(self.to_out),k,v
   def get_parameters(self, is_last:bool):
      return [
         *([] if is_last else get_parameters(self.to_q)),
         *get_parameters(self.to_k),
         *get_parameters(self.to_v),
         *([] if is_last else get_parameters(self.to_out)),
      ]

class CrossAttention:
   def __init__(self, query_dim, n_heads, d_head, dropout=Config.dropout):
      self.to_q = Linear(query_dim, n_heads*d_head, bias=False)
      self.num_heads = n_heads
      self.head_size = d_head
      self.to_out = [Linear(n_heads*d_head, query_dim)]
      self.dropout = dropout
   def __call__(self, x:Tensor, attn_mask:Tensor, k:Tensor, v:Tensor) -> Tuple[Tensor, Tensor, Tensor]:
      fnx = lambda y: y.reshape(x.shape[0], -1, self.num_heads, self.head_size).transpose(-3,-2)
      q = fnx(self.to_q(x))
      attn_mask = attn_mask.reshape(attn_mask.shape[0], 1, *attn_mask.shape[1:]).expand((attn_mask.shape[0], self.num_heads, *attn_mask.shape[1:]))
      attention = scaled_dot_product_attention(q, k, v, attn_mask=attn_mask).dropout(self.dropout).transpose(-3,-2)
      h_ = attention.reshape(shape=(*x.shape[0:-2], -1, self.num_heads * self.head_size))
      return h_.sequential(self.to_out), k, v

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

class AttentionBlock:
   def __init__(self, dim, n_heads, d_head, ff_mult, dropout=Config.dropout, is_causal=False, cross_attn=True):
      self.norm1 = LayerNorm(dim)
      self.attn1 = SelfAttention(dim, n_heads, d_head, is_causal=is_causal)
      self.cross_attn = cross_attn
      if self.cross_attn:
         self.norm2 = LayerNorm(dim)
         self.attn2 = CrossAttention(dim, n_heads, d_head)
      self.norm3 = LayerNorm(dim)
      self.ff    = FeedForward(dim, mult=ff_mult)
      self.dropout = dropout
   def __call__(self, x, attn_mask:Optional[Tensor]=None, k:Optional[Tensor]=None, v:Optional[Tensor]=None) -> Tuple[Tensor, Tensor, Tensor]:
      h,k_,v_ = self.attn1(self.norm1(x))
      x = x + h
      if self.cross_attn:
         h,_,_ = self.attn2(self.norm2(x), attn_mask=attn_mask, k=k, v=v)
         x = x + h
      x = self.ff(self.norm3(x)).dropout(self.dropout) + x
      return x, k_, v_
   def get_parameters(self, is_last:bool):
      return [
         *(get_parameters(self.norm1)),
         *(self.attn1.get_parameters(is_last)),
         *(get_parameters(self.norm2) + get_parameters(self.attn2) if self.cross_attn else []),
         *([] if is_last else get_parameters(self.norm3) + get_parameters(self.ff)),
      ]

class TimeFusion:
   def __init__(self, channels, time_channels, emb_channels):
      self.in_layers = [
         Tensor.silu,
         Linear(channels, emb_channels),
      ]
      self.emb_layers = [
         Tensor.silu,
         Linear(time_channels, emb_channels),
      ]
      self.out_layers = [
         Tensor.silu,
         Linear(emb_channels, channels),
      ]
   def __call__(self, x:Tensor, time_emb:Tensor):
      return x + (x.sequential(self.in_layers) + time_emb.sequential(self.emb_layers)).sequential(self.out_layers)

def timestep_embedding(timesteps, dim, max_period=10000) -> Tensor:
  half = dim // 2
  freqs = (-math.log(max_period) * Tensor.arange(half) / half).exp()
  args = timesteps * freqs
  return Tensor.cat(args.cos(), args.sin(), dim=-1)

class FusedTransformer:
   def __init__(self, vocab_size:int, timesteps:int, time_deltas:int, ctx_pos_size:int, den_pos_size:int, n_layers:int, ctx_dim:int, den_dim:int, time_dim:int, fusion_mult:int, ctx_heads:int, den_heads:int, ctx_ff_mult:int, den_ff_mult:int):
      self.ctx_tok_embed = Embedding(vocab_size, ctx_dim)
      self.ctx_pos_embed = Embedding(ctx_pos_size, ctx_dim)

      self.den_tok_embed  = Embedding(den_pos_size, den_dim)
      self.den_time_embed = [
         Linear(time_dim, den_dim*2),
         Tensor.silu,
         Linear(den_dim*2, den_dim*2),
      ]
      self.time_dim = time_dim

      self.layers: List[Tuple[AttentionBlock,TimeFusion,AttentionBlock]] = [
         (
            AttentionBlock(ctx_dim, ctx_heads, ctx_dim//ctx_heads, ctx_ff_mult, is_causal=True, cross_attn=False),
            TimeFusion(den_dim, den_dim*2, den_dim*fusion_mult),
            AttentionBlock(den_dim, den_heads, den_dim//den_heads, den_ff_mult),
         ) for _ in range(n_layers)
      ]
      self.n_layers = n_layers

      self.ctx_class_head = Linear(ctx_dim, vocab_size)
      self.den_class_head = Linear(den_dim, vocab_size)
   
   def get_parameters(self, phase:int) -> List[Tensor]:
      params = []

      if phase == 1 or phase == 3:
         params += get_parameters(self.ctx_tok_embed)
         params += get_parameters(self.ctx_pos_embed)
      if phase == 2 or phase == 3:
         params += get_parameters(self.den_tok_embed)
         params += get_parameters(self.den_time_embed)

      for i, v in enumerate(self.layers):
         ctx_layer, time_layer, den_layer = v
         if phase == 1 or phase == 3: params += ctx_layer.get_parameters(i+1 == self.n_layers)
         if phase == 2 or phase == 3:
            params += get_parameters(time_layer)
            params += get_parameters(den_layer)

      if phase == 1:
         params += get_parameters(self.ctx_class_head)
      else:
         params += get_parameters(self.den_class_head)

      return params

   def forward_ctx_only(self, ctx_toks:Tensor) -> Tensor:
      x = self.ctx_tok_embed(ctx_toks) + self.ctx_pos_embed(Tensor.arange(0, ctx_toks.shape[-1], requires_grad=False).reshape((1,-1)))
      for ctx_layer, _, _ in self.layers:
         x,_,_ = ctx_layer(x)
      return self.ctx_class_head(x).log_softmax()

   def make_x_0_from(self, toks:Tensor) -> Tensor:
      return self.den_tok_embed(toks)

   def __call__(self, den_latent:Tensor, ctx_toks:Tensor, timesteps:Tensor, attn_mask:Tensor, detach_ctx:bool=False) -> Tensor:
      ctx_latent  = self.ctx_tok_embed(ctx_toks) + self.ctx_pos_embed(Tensor.arange(0, ctx_toks.shape[-1], requires_grad=False).reshape((1,-1)))
      time_latent = timestep_embedding(timesteps.reshape((-1,1)), self.time_dim).sequential(self.den_time_embed)

      B,T,DS,DD = attn_mask.shape
      den_latent = den_latent.reshape(-1,DS,DD)
      attn_mask  = attn_mask .reshape(-1,DS,DD)

      for ctx_layer, time_layer, den_layer in self.layers:
         ctx_latent,k,v = ctx_layer(ctx_latent)
         k,v = [y.reshape((B,1,*y.shape[1:])).expand((B,T,*y.shape[1:])).reshape((-1,*y.shape[1:])) for y in (k,v,)]
         if detach_ctx:
            k,v = k.detach(),v.detach()
         den_latent = time_layer(den_latent, time_latent)
         den_latent,_,_ = den_layer(den_latent, attn_mask, k, v) # type: ignore
      
      return den_latent.reshape((B,T,DS,DD))

   def estimate(self, den_latent:Tensor) -> Tensor:
      return self.den_class_head(den_latent).log_softmax()




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






def train(phase:int, scale:float=0.5):
   assert phase in (options:=[1,2,3]), f"phase was {phase}, must be in {options}"

   model = FusedTransformer(**Config.model_params.to_dict())

   type_name = os.path.basename(os.path.dirname(__file__))
   if phase == 1 or True:
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

   # if phase == 2 or phase == 3:
   #    load_latest_weight(model, "model", phase=(phase-1))

   all_params = get_parameters(model)
   train_params = model.get_parameters(phase)
   for label, params in (("Model Params:", all_params), ("Train Params:", train_params)):
      sz = 0
      for p in params:
         sz += prod(p.shape)
      print(f"{label} {sz/1e6:.2f}m")
   opt = Adam(train_params, Config.train[phase].learning_rate)

   all_alphas = make_alphas()
   X_train, X_test = load_train_test()

   TS = Config.model_params.timesteps
   BS = Config.train[phase].batch_size
   CS = Config.model_params.ctx_pos_size
   DS = Config.model_params.den_pos_size
   TD = Config.model_params.time_deltas

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
      X = Tensor(X_tok, dtype=dtypes.float32, requires_grad=False)

      if phase == 1:
         Y_tok = np.array([data[i+1:i+1+CS] for i in index], dtype=np.float32)
         Y = Tensor(Y_tok, dtype=dtypes.float32)
         
         output = model.forward_ctx_only(X)
         output.realize()
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
         alphas_np = alphas
         alphas = Tensor(alphas, dtype=dtypes.float32, requires_grad=False).reshape(1,1,DS,1).expand(BS,CS,DS,Config.model_params.den_dim)

         attn_mask = Tensor.ones(CS,CS).tril(0).cast(dtypes.bool).reshape(1,CS,1,CS).expand(BS,CS,DS,CS)
         x_0 = model.make_x_0_from(Y.detach())
         x_t = x_0*alphas + ((1-alphas)*Tensor.randn(*x_0.shape)).detach()

         e_t = model(x_t, X, Tensor(timesteps, dtype=dtypes.float32, requires_grad=False), attn_mask)
         pred_x_0 = x_t - e_t

         loss_1 = (pred_x_0 - x_0).pow(2).sum() / prod(pred_x_0.shape)
         loss_2 = model.estimate(x_0).sparse_categorical_crossentropy(Y)
         loss_3 = (output:=model.estimate(pred_x_0)).sparse_categorical_crossentropy(Y)
         loss = loss_1 + loss_2 + loss_3

         del attn_mask, loss_1, loss_2, loss_3, x_0, pred_x_0, x_t, e_t
      
      loss = output.sparse_categorical_crossentropy(Y)

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

      del X, Y, output, loss

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
         safe_save(get_state_dict(model), os.path.join(weights_folder, Config.save_name.format(phase, step)))
         config_filepath = f"{weights_folder}/config.py"
         if not os.path.exists(config_filepath):
            shutil.copyfile(f"{os.path.dirname(__file__)}/config.py", config_filepath)
         main_filepath = f"{weights_folder}/{os.path.basename(__file__)}"
         if not os.path.exists(main_filepath):
            shutil.copyfile(__file__, main_filepath)

      if step % Config.train[phase].gen_every == 0:
         g_time = time.time()
         if phase == 1:
            text = generate_ctx(Config.train[phase].gen_count, model=model)
         else:
            text = generate_den(Config.train[phase].gen_count, model=model)
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

def generate_ctx(count=20, model=None, start=text, archive=False):
   load_train_test()
   if model is None:
      model = FusedTransformer(**Config.model_params.to_dict())
      load_latest_weight(model, "model", archive)

   CS = Config.model_params.ctx_pos_size
   all_output = start
   start_i = len(all_output) - 1

   X = Tensor(encode(all_output), dtype=dtypes.float32, requires_grad=False).reshape(1,-1).pad( ((0,0), (0,CS-len(all_output))) )
   for i in trange(start_i, count+start_i):
      assert X.shape == (1,CS,)

      pull_i = min(i, CS-1)
      pred = model.forward_ctx_only(X.realize())[:,pull_i:pull_i+1].argmax(axis=-1)
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
      del pred
   
   del X
   return all_output

def generate_den(count=20, timestep_reduce=10, model:Optional[FusedTransformer]=None, start=text, archive=False):
   global encode, decode
   load_train_test()
   all_alphas = make_alphas()
   if model is None:
      model = FusedTransformer(**Config.model_params.to_dict())
      load_latest_weight(model, "model", archive)

   BS = 1
   TS = Config.model_params.timesteps
   CS = Config.model_params.ctx_pos_size
   DS = Config.model_params.den_pos_size
   TD = Config.model_params.time_deltas
   all_output = start

   def make_X(toks, start_i):
      if len(toks) > CS:
         toks = toks[-CS:]
      data = np.zeros((CS,))-1
      data[:start_i] = np.array(encode(toks[:start_i]))
      data_i = start_i
      return Tensor(data, dtype=dtypes.float32, requires_grad=False).reshape(1,-1), data_i
   
   def make_x_0(toks, start_i: int) -> Tensor:
      assert start_i < len(toks)
      start_data = np.array(encode(toks[start_i:])) # type: ignore
      return model.make_x_0_from(Tensor(start_data, dtype=dtypes.float32, requires_grad=False).reshape(1,-1))

   start_i = len(all_output) - DS
   assert start_i > 0, f"input size {len(all_output)} must be atleast 1 greater than decoder head size {DS}"
   x_0 = make_x_0(all_output, start_i)
   z_0 = x_0
   den_start_amount = timestep_reduce
   X, data_i = make_X(all_output, start_i)

   for i in trange(count):

      alphas = np.ones((DS,), dtype=np.float32)
      timesteps = np.zeros((DS,), dtype=np.float32)
      for i in range(DS):
         ts = min(den_start_amount + i*TD, TS-1)
         alphas[i] = all_alphas[int(ts)]
         timesteps[i] = ts
      alphas = Tensor(alphas, dtype=dtypes.float32, requires_grad=False).reshape(1,1,DS,1)
      attn_mask = Tensor.ones(CS,CS).tril(0).cast(dtypes.bool).reshape(1,CS,1,CS).expand(BS,CS,DS,CS)[:,data_i-1:data_i,:,:]

      x_t = x_0*alphas + Tensor.randn(BS,1,Config.model_params.den_dim)*(1-alphas)
      e_t = model(x_t, X, Tensor(timesteps, dtype=dtypes.float32, requires_grad=False), attn_mask)
      pred_x_0 = x_t - e_t

      while den_start_amount <= timestep_reduce:
         if start_i >= len(all_output):
            pred = model.estimate(pred_x_0)[0,0,0,:]
            all_output += decode([pred.argmax(axis=-1).numpy().item()])[0] # type: ignore
         start_i += 1
         X, data_i = make_X(all_output, start_i)
         pred_x_0 = pred_x_0[:,:,1:].cat(Tensor.zeros(*pred_x_0.shape[:2],1,pred_x_0.shape[-1]), dim=-2)
         den_start_amount += TD
      den_start_amount -= timestep_reduce

      if start_i < len(all_output):
         new_x_0 = make_x_0(all_output, start_i)
         pred_x_0[:,:,:len(all_output)-start_i] = new_x_0.reshape((1,*new_x_0.shape)).detach()

      x_0 = pred_x_0.realize().detach()

   return all_output

if __name__ == "__main__":
   # train(phase=1)
   # print(generate_ctx(count=512))

   # train(phase=2)
   train(phase=3)
   # print(generate_den(count=128, model=FusedTransformer(**Config.model_params.to_dict())))
