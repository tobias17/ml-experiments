import torch
from torch import Tensor
from torch.nn import Linear, LayerNorm, Embedding, Dropout, Sequential, functional, SiLU, GELU, CrossEntropyLoss, Module, LogSoftmax, Softmax
from torch.optim import Adam
import tiktoken
import numpy as np
from config import Config
from util import write_graph, write_probs, Schedules
from typing import Dict, Optional, List, Tuple, Union
import time, datetime, os, shutil, math, json
from tqdm import tqdm, trange # type: ignore
from functools import reduce
from safetensors.torch import load_model, save_model

device = "cuda"
torch.set_default_device(device)
TO: Dict = { "device": device, "dtype": torch.bfloat16 }

def prod(collection):
   return reduce(lambda a,b: a*b, collection, 1)

class SelfAttention(Module):
   def __init__(self, query_dim, n_heads, d_head, dropout=Config.dropout, is_causal=False):
      super().__init__()
      self.to_q = Linear(query_dim, n_heads*d_head, bias=False)
      self.to_k = Linear(query_dim, n_heads*d_head, bias=False)
      self.to_v = Linear(query_dim, n_heads*d_head, bias=False)
      self.num_heads = n_heads
      self.head_size = d_head
      self.to_out = Linear(n_heads*d_head, query_dim)
      self.dropout = Dropout(dropout)
      self.is_causal = is_causal
   def forward(self, x:Tensor) -> Tuple[Tensor, Tensor, Tensor]:
      q,k,v = self.to_q(x), self.to_k(x), self.to_v(x)
      q,k,v = [y.reshape(x.shape[0], -1, self.num_heads, self.head_size).transpose(-3,-2) for y in (q,k,v)]
      attention = self.dropout(functional.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal)).transpose(-3,-2)
      h_ = attention.reshape(shape=(*x.shape[0:-2], -1, self.num_heads * self.head_size))
      return self.to_out(h_),k,v
   def freeze_parameters(self, is_last:bool):
      if is_last:
         for param in list(self.to_q.parameters()) + list(self.to_out.parameters()):
            param.requires_grad = False

class CrossAttention(Module):
   def __init__(self, query_dim, n_heads, d_head, dropout=Config.dropout):
      super().__init__()
      self.to_q = Linear(query_dim, n_heads*d_head, bias=False)
      self.num_heads = n_heads
      self.head_size = d_head
      self.to_out = Linear(n_heads*d_head, query_dim)
      self.dropout = Dropout(dropout)
   def forward(self, x:Tensor, attn_mask:Tensor, k:Tensor, v:Tensor) -> Tuple[Tensor, Tensor, Tensor]:
      fnx = lambda y: y.reshape(x.shape[0], -1, self.num_heads, self.head_size).transpose(-3,-2)
      q = fnx(self.to_q(x))
      attn_mask = attn_mask.reshape(attn_mask.shape[0], 1, *attn_mask.shape[1:]).expand((attn_mask.shape[0], self.num_heads, *attn_mask.shape[1:]))
      attention = self.dropout(functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)).transpose(-3,-2)
      h_ = attention.reshape(shape=(*x.shape[0:-2], -1, self.num_heads * self.head_size))
      return self.to_out(h_), k, v

class GEGLU(Module):
   def __init__(self, dim_in, dim_out):
      super().__init__()
      self.proj = Linear(dim_in, dim_out * 2)
      self.dim_out = dim_out
      self.act = GELU()
   def forward(self, x:Tensor) -> Tensor:
      x, gate = self.proj(x).chunk(2, dim=-1)
      return x * self.act(gate)

class FeedForward(Module):
   def __init__(self, dim, mult=4):
      super().__init__()
      self.net = Sequential(
         GEGLU(dim, dim*mult),
         Linear(dim*mult, dim),
      )
   def forward(self, x:Tensor) -> Tensor:
      return self.net(x)

class AttentionBlock(Module):
   def __init__(self, dim, n_heads, d_head, ff_mult, dropout=Config.dropout, is_causal=False, cross_attn=True):
      super().__init__()
      self.norm1 = LayerNorm(dim)
      self.attn1 = SelfAttention(dim, n_heads, d_head, is_causal=is_causal)
      self.cross_attn = cross_attn
      if self.cross_attn:
         self.norm2 = LayerNorm(dim)
         self.attn2 = CrossAttention(dim, n_heads, d_head)
      self.norm3 = LayerNorm(dim)
      self.ff    = FeedForward(dim, mult=ff_mult)
      self.dropout = Dropout(dropout)
   def forward(self, x, attn_mask:Optional[Tensor]=None, k:Optional[Tensor]=None, v:Optional[Tensor]=None) -> Tuple[Tensor, Tensor, Tensor]:
      h,k_,v_ = self.attn1(self.norm1(x))
      x = x + h
      if self.cross_attn:
         h,_,_ = self.attn2(self.norm2(x), attn_mask=attn_mask, k=k, v=v)
         x = x + h
      x = self.dropout(self.ff(self.norm3(x))) + x
      return x, k_, v_
   def freeze_parameters(self, is_last:bool):
      self.attn1.freeze_parameters(is_last)
      if is_last:
         for param in list(self.norm3.parameters()) + list(self.ff.parameters()):
            param.requires_grad = False

class TimeFusion(Module):
   def __init__(self, channels, time_channels, emb_channels):
      super().__init__()
      self.in_layers = Sequential(
         SiLU(),
         Linear(channels, emb_channels),
      )
      self.emb_layers = Sequential(
         SiLU(),
         Linear(time_channels, emb_channels),
      )
      self.out_layers = Sequential(
         SiLU(),
         Linear(emb_channels, channels),
      )
   def forward(self, x:Tensor, time_emb:Tensor):
      return x + self.out_layers(self.in_layers(x) + self.emb_layers(time_emb))

def timestep_embedding(timesteps, dim, max_period=10000) -> Tensor:
  half = dim // 2
  freqs = (-math.log(max_period) * torch.arange(half) / half).exp().to(**TO)
  args = timesteps * freqs
  return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

class FusedTransformer(Module):
   def __init__(self, vocab_size:int, timesteps:int, time_deltas:int, ctx_pos_size:int, den_pos_size:int, n_layers:int, ctx_dim:int, den_dim:int, time_dim:int, fusion_mult:int, ctx_heads:int, den_heads:int, ctx_ff_mult:int, den_ff_mult:int):
      super().__init__()
      self.ctx_tok_embed = Embedding(vocab_size, ctx_dim)
      self.ctx_pos_embed = Embedding(ctx_pos_size, ctx_dim)

      self.den_tok_embed  = Embedding(vocab_size, den_dim)
      self.den_time_embed = Sequential(
         Linear(time_dim, den_dim*2),
         SiLU(),
         Linear(den_dim*2, den_dim*2),
      )
      self.den_dim, self.time_dim = den_dim, time_dim

      self.layers: List[Tuple[AttentionBlock,TimeFusion,AttentionBlock]] = []
      for i in range(n_layers):
         a = AttentionBlock(ctx_dim, ctx_heads, ctx_dim//ctx_heads, ctx_ff_mult, is_causal=True, cross_attn=False)
         b = TimeFusion(den_dim, den_dim*2, den_dim*fusion_mult)
         c = AttentionBlock(den_dim, den_heads, den_dim//den_heads, den_ff_mult)
         self.layers.append((a,b,c))
         setattr(self, f"layer{i}_a", a)
         setattr(self, f"layer{i}_b", b)
         setattr(self, f"layer{i}_c", c)
      self.n_layers = n_layers

      self.ctx_class_head = Linear(ctx_dim, vocab_size)
      self.den_class_head = Linear(den_dim, vocab_size)

      self.log_softmax = LogSoftmax(dim=-1)
      self.softmax = Softmax(dim=-1)
   
   def freeze_parameters(self, phase:int) -> None:
      tc = Config.train[phase]
      freeze: List[Module | Sequential] = []

      if not tc.grad_ctx:
         freeze.append(self.ctx_tok_embed)
         freeze.append(self.ctx_pos_embed)
      if not tc.grad_den:
         freeze.append(self.den_tok_embed)
         freeze.append(self.den_time_embed)

      for i, v in enumerate(self.layers):
         ctx_layer, time_layer, den_layer = v
         if tc.grad_ctx:
            ctx_layer.freeze_parameters(i+1 == self.n_layers and (not tc.ctx_tok_loss))
         else:
            freeze.append(ctx_layer)
         if not tc.grad_den:
            freeze.append(time_layer)
            freeze.append(den_layer)

      if not tc.ctx_tok_loss:
         freeze.append(self.ctx_class_head)
      if not tc.den_tok_loss_orig and not tc.den_tok_loss_pred:
         freeze.append(self.den_class_head)
      
      for module in freeze:
         for param in module.parameters():
            param.requires_grad = False

   def forward_ctx_only(self, ctx_toks:Tensor, temperature:float=1.0, use_log=True) -> Tensor:
      x = self.ctx_tok_embed(ctx_toks) + self.ctx_pos_embed(torch.arange(0, ctx_toks.shape[-1], requires_grad=False).reshape((1,-1)))
      for ctx_layer, _, _ in self.layers:
         x,_,_ = ctx_layer(x)
      return self.ctx_predict(x, temperature, use_log)

   def ctx_predict(self, ctx_latent:Tensor, temperature:float=1.0, use_log=True) -> Tensor:
      fnx = self.log_softmax if use_log else self.softmax
      return fnx(self.ctx_class_head(ctx_latent).float() / (temperature+1e-10))

   def make_x_0_from(self, toks:Tensor) -> Tensor:
      return self.den_tok_embed(toks)

   def forward(self, den_latent:Tensor, ctx_toks:Tensor, timesteps:Tensor, attn_mask:Tensor, detach_ctx:bool=False) -> Tuple[Tensor, Tensor]:
      ctx_latent  = self.ctx_tok_embed(ctx_toks) + self.ctx_pos_embed(torch.arange(0, ctx_toks.shape[-1], requires_grad=False).reshape((1,-1)))
      time_latent = self.den_time_embed(timestep_embedding(timesteps.reshape((-1,1)), self.time_dim))

      B,T,DS,_ = attn_mask.shape
      den_latent = den_latent.reshape(-1,DS,self.den_dim)
      attn_mask  = attn_mask .reshape(-1,DS,attn_mask.shape[-1])

      for ctx_layer, time_layer, den_layer in self.layers:
         ctx_latent,k,v = ctx_layer(ctx_latent)
         k,v = [y.reshape((B,1,*y.shape[1:])).expand((B,T,*y.shape[1:])).reshape((-1,*y.shape[1:])) for y in (k,v,)]
         if detach_ctx:
            k,v = k.detach(),v.detach()
         den_latent = time_layer(den_latent, time_latent)
         den_latent,_,_ = den_layer(den_latent, attn_mask, k, v) # type: ignore
      
      return den_latent.reshape((B,T,DS,self.den_dim)), ctx_latent

   def estimate(self, den_latent:Tensor, temperature=1.0, use_log=True) -> Tensor:
      fnx = self.log_softmax if use_log else self.softmax
      return fnx(self.den_class_head(den_latent).float() / (temperature+1e-10))



import matplotlib.pyplot as plt # type: ignore

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
      plt.plot(np.arange(T), a)
      plt.show()
   return a

def load_train_test():
   global encode, decode
   enc = tiktoken.get_encoding("gpt2")
   encode = enc.encode_ordinary
   decode = enc.decode_bytes
   return [np.memmap(Config.dataset.format(split), dtype=np.uint16, mode='r') for split in ('train', 'val')]

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
   load_model(model, last_weight)






def train(phase:int, token_ptr=0, recover=False):
   tc = Config.train[phase]
   phase_offset = (phase-1) * 800_000_000

   model = FusedTransformer(**Config.model_params.to_dict()).to(**TO)

   TS = Config.model_params.timesteps
   BS = tc.batch_size
   CS = Config.model_params.ctx_pos_size
   DS = Config.model_params.den_pos_size
   TD = Config.model_params.time_deltas

   step, test_index = 0, 0
   train_loss:List[float] = []
   test_loss: List[float] = []
   if phase == 1:
      train_acc:List[float] = []
      test_acc: List[float] = []
   else:
      train_accs:List[List[float]] = [[] for _ in range(DS)]
      test_accs: List[List[float]] = [[] for _ in range(DS)]
      deep_acc:  List[float] = []
      deep_probs:List[Tuple[float,float,float]] = []
      
   if recover == True:
      weights_folder = get_latest_folder()
      data_filepath = f"{weights_folder}/p{phase}_data.json"
      if not os.path.exists(data_filepath):
         raise ValueError(f"Could not find data file {data_filepath} with recover=True")
      with open(data_filepath) as f:
         data = json.load(f)
         step = data["step"]
         token_ptr  = data["token_ptr"]
         train_loss = data["train_loss"]
         test_loss  = data["test_loss"]
         if phase == 1:
            train_acc = data["train_acc"]
            test_acc = data["test_acc"]
         else:
            train_accs = data["train_acc"]
            test_accs  = data["test_acc"]
            deep_acc   = data.get("deep_acc", [])
            deep_probs = data.get("deep_probs", [])
      weights_filepath = f"{weights_folder}/" + Config.save_name.format(phase, f"{step//1000}k")
      if not os.path.exists(weights_filepath):
         raise ValueError(f"Could not find weights file {weights_filepath} with recover=True")
      load_model(model, weights_filepath)
      print(f"Recovering to Step {step} in Phase {phase} from {weights_folder}")
   else:
      if phase == Config.start_phase:
         type_name = os.path.basename(os.path.dirname(__file__))
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
         load_latest_weight(model, "model", phase=(phase-1))

   model.freeze_parameters(phase)
   all_params = list(model.parameters())
   train_params = [p for p in all_params if p.requires_grad]
   print(f"\nPhase {phase}")
   for label, params in (("Train Params:", train_params), ("Model Params:", all_params)):
      sz = 0
      for p in params:
         sz += prod(p.shape)
      print(f"{label} {sz/1e6:.2f}m")
   print("="*25)
   opt = Adam(train_params, tc.learning_rate)

   all_alphas = make_alphas()
   X_train, X_test = load_train_test()
   cross_entroy = CrossEntropyLoss(reduce="none")
   def loss_fnx(x, y):
      x = x.reshape(-1,*x.shape[-2:]).permute((0,2,1))
      y = y.reshape(-1,*y.shape[-1:])
      return cross_entroy(x, y).sum()

   s_time = time.time()
   while True:
      np.random.seed(step if test_index < 2 else 1337)
      if test_index <= 1:
         data = X_train
         index = [phase_offset+token_ptr+CS*i for i in range(BS)]
         if test_index == 0:
            token_ptr += CS*BS
      else:
         data = X_test
         index = [CS*i for i in range(BS)]

      X_tok = np.array([data[i:i+CS] for i in index], dtype=np.int32)
      X = Tensor(X_tok).int().detach().to(device)

      loss = torch.zeros((1)).sum()
      if tc.den_tok_loss_orig or tc.den_tok_loss_pred or tc.den_tok_noise_loss:
         Y_tok = np.array([[data[i+j:i+j+DS] for j in range(1,CS+1)] for i in index], dtype=np.float32)
         Y = Tensor(Y_tok).long().detach().to(device)

         diff_start_amount = np.random.randint(1, TD+1) if test_index==0 else TD // 2

         alphas = np.ones((DS,), dtype=np.float32)
         timesteps = np.zeros((DS,), dtype=np.float32)
         for i in range(DS):
            ts = min(diff_start_amount + i*TD, TS - 1)
            alphas[i] = all_alphas[int(ts)]
            timesteps[i] = ts
         alphas = Tensor(alphas).detach().reshape(1,1,DS,1).expand(BS,CS,DS,Config.model_params.den_dim).to(**TO) # type: ignore

         attn_mask = torch.ones(CS,CS).tril(0).bool().reshape(1,CS,1,CS).expand(BS,CS,DS,CS).to(device)
         x_0 = model.make_x_0_from(Y)
         x_t = x_0*alphas + ((1-alphas)*torch.randn(*x_0.shape).to(**TO)).detach() # type: ignore

         e_t, ctx_latent = model(x_t, X, Tensor(timesteps).detach().to(**TO), attn_mask, detach_ctx=tc.detach_ctx)
         pred_x_0 = x_t - e_t
         
         output = model.estimate(pred_x_0)

         if tc.ctx_tok_loss:
            ctx_Y_tok = np.array([data[i+1:i+1+CS] for i in index], dtype=np.float32)
            loss = loss + loss_fnx(model.ctx_predict(ctx_latent), Tensor(ctx_Y_tok).to(**TO))
         if tc.den_tok_loss_orig:
            loss = loss + loss_fnx(model.estimate(x_0), Y)
         if tc.den_tok_loss_pred:
            loss = loss + loss_fnx(output, (Y))
         if tc.den_tok_noise_loss:
            loss = loss + ((pred_x_0 - x_0).pow(2).sum() / prod(pred_x_0.shape))

         del attn_mask, x_0, pred_x_0, x_t, e_t
      elif tc.ctx_tok_loss:
         Y_tok = np.array([data[i+1:i+1+CS] for i in index], dtype=np.int64)
         Y = Tensor(Y_tok).long().to(device)
         
         output = model.forward_ctx_only(X)
         loss = loss + loss_fnx(output, Y)

      if test_index == 0:
         opt.zero_grad()
         loss.backward()
         opt.step()
      else:
         loss_l = test_loss if test_index==2 else train_loss
         loss_l.append(loss.detach().cpu().numpy().item())
         
         if phase == 1:
            acc_l = test_acc if test_index==2 else train_acc
            acc_l.append((torch.argmax(output, dim=-1)==Y).float().mean().detach().cpu().numpy().item())
         else:
            accs_l = test_accs if test_index==2 else train_accs
            for i in range(DS):
               accs_l[i].append((torch.argmax(output[:,:,i:i+1], dim=-1)==Y[:,:,i:i+1]).float().mean().detach().cpu().numpy().item())

      del X, Y, output, loss

      if (step+1) % Config.train[phase].deep_every == 0 and phase > 1 and test_index == 0:
         g_time = time.time()
         acc, probs = deep_test_den(X_test, model)
         deep_acc.append(acc)
         deep_probs.append(probs)
         s_time += (time.time() - g_time)

      TE = Config.train[phase].test_every
      if (step+1) % TE == 0:
         if test_index == 2:
            step += 1
            if phase == 1:
               train_acc_str, test_acc_str = f"{100.0*train_acc[-1]:.2f}", f"{100.0*test_acc[-1]:.2f}"
            else:
               train_acc_str, test_acc_str = f"{100.0*sum(train_accs[i][-1] for i in range(DS))/DS:.2f}", f"{100.0*sum(test_accs[i][-1] for i in range(DS))/DS:.2f}"
            deep_text = f" | Deep Acc: {100.0*deep_acc[-1]:.2f}%" if phase > 1 and len(deep_acc) > 0 else ""
            print(f"Step {str(step): >5} | Train Loss: {train_loss[-1]:.4f} | Train Acc: {train_acc_str}% | Test Loss: {test_loss[-1]:.4f} | Test Acc: {test_acc_str}%{deep_text} | {(time.time() - s_time) / float(TE):.2f} sec/iter")
            div = 1_000_000
            write_graph(train_loss, test_loss, f"{weights_folder}/p{phase}_graph_loss.png", delta=token_ptr//(len(train_loss))/div, x_label="tokens (million)", y_label="loss", title=f"Phase {phase} Loss")
            if phase == 1:
               write_graph(train_acc,  test_acc,  f"{weights_folder}/p{phase}_graph_acc.png", ylim=(0,1), delta=token_ptr//(len(train_acc))/div, x_label="tokens (million)", y_label="acc", title=f"Phase {phase} Accuracy")
            else:
               write_graph(train_accs, test_accs, f"{weights_folder}/p{phase}_graph_shallow_acc.png", ylim=(0,1), segmented=True, delta=token_ptr//(len(train_accs[0]))/div, x_label="tokens (million)", y_label="acc", title=f"Phase {phase} Shallow Accuracy")
               if len(deep_acc) > 0:
                  write_graph(deep_acc, deep_acc, f"{weights_folder}/p{phase}_graph_deep_acc.png", ylim=(0,1), delta=token_ptr//(len(deep_acc))/div, x_label="tokens (million)", y_label="acc", title=f"Phase {phase} Deep Accuracy")
                  write_probs(deep_probs, f"{weights_folder}/p{phase}_graph_deep_probs", delta=token_ptr//(len(deep_probs))/div, x_label="tokens (million)", y_label="acc", title=f"Phase {phase} Deep Accuracy")
            s_time = time.time()
            test_index = 0
         else:
            test_index += 1
      else:
         step += 1

      if step % Config.train[phase].save_every == 0:
         if not os.path.exists(weights_folder):
            os.makedirs(weights_folder)
         save_model(model, os.path.join(weights_folder, Config.save_name.format(phase, f"{step//1000}k")))
         config_filepath = f"{weights_folder}/config.py"
         if not os.path.exists(config_filepath):
            shutil.copyfile(f"{os.path.dirname(__file__)}/config.py", config_filepath)
         main_filepath = f"{weights_folder}/{os.path.basename(__file__)}"
         if not os.path.exists(main_filepath):
            shutil.copyfile(__file__, main_filepath)
         with open(f"{weights_folder}/p{phase}_data.json", "w") as f:
            data = {
               "step": step,
               "token_ptr": token_ptr,
               "train_loss": train_loss,
               "test_loss": test_loss,
               "train_acc": (train_acc if phase==1 else train_accs),
               "test_acc": (test_acc if phase==1 else test_accs),
               "deep_acc": [] if phase == 1 else deep_acc,
               "deep_probs": [] if phase == 1 else deep_probs,
            }
            json.dump(data, f)

      if step % Config.train[phase].gen_every == 0:
         g_time = time.time()
         if phase == 1:
            text = generate_ctx(Config.train[phase].gen_count, model=model)
         else:
            text = generate_den(Config.train[phase].gen_count, model=model)
         gen_folder = f"{weights_folder}/p{phase}_gens"
         if not os.path.exists(gen_folder):
            os.makedirs(gen_folder)
         with open(f"{gen_folder}/text_{step//1000}k.txt", "w") as f:
            try:
               f.write(text)
            except Exception:
               print(f"Failed to write text to file:\n{text}")
         s_time += (time.time() - g_time)






text = """SEBASTIAN:
Bate, I beseech you, widow Dido.

ANTONIO:
"""

def generate_ctx(count=8, model=None, start=text, archive=False, temperature=0.4):
   with torch.no_grad():
      np.random.seed(1337)
      load_train_test()
      if model is None:
         model = FusedTransformer(**Config.model_params.to_dict())
         load_latest_weight(model, "model", archive)

      CS = Config.model_params.ctx_pos_size
      all_output = start

      X = Tensor(encode(all_output)).float().reshape(1,-1).to(**TO)
      start_i = X.shape[-1]-1
      X = torch.cat([X, torch.zeros(1,CS-X.shape[-1])], dim=-1).int()
      for i in trange(start_i, count+start_i):
         assert X.shape == (1,CS,)

         pull_i = min(i, CS-1)
         probs_np = model.forward_ctx_only(X, temperature=temperature, use_log=False)[0,pull_i,:].cpu().numpy()
         tok = int(np.random.choice(len(probs_np), p=probs_np))
         byte = decode([tok])
         try:
            char = byte.decode()
         except Exception:
            char = "<?>"
         all_output += char

         X_np = np.zeros((1,CS+1), dtype=np.float32)
         X_np[:,:-1] = X.cpu().numpy()
         if i + 1 < CS:
            X_np[:,i+1:i+2] = tok
            X = Tensor(X_np[:,:-1]).int().to(device)
         else:
            X_np[:,-1:] = tok
            X = Tensor(X_np[:,1:]).int().to(device)

      return all_output

def generate_den(count=20, timestep_reduce=8, model:Optional[FusedTransformer]=None, start=text, archive=False, temperature=0.4):
   with torch.no_grad():

      global encode, decode
      load_train_test()
      all_alphas = make_alphas()
      if model is None:
         model = FusedTransformer(**Config.model_params.to_dict())
         load_latest_weight(model, "model", archive)
      
      count = int(count * (Config.model_params.time_deltas / timestep_reduce))

      BS = 1
      TS = Config.model_params.timesteps
      CS = Config.model_params.ctx_pos_size
      DS = Config.model_params.den_pos_size
      TD = Config.model_params.time_deltas
      all_output = start

      def make_X(toks, start_i):
         if len(toks) > CS:
            toks = toks[-CS:]
            start_i = CS
         data = np.zeros((CS,))
         data[:start_i] = np.array(toks[:start_i])
         data_i = start_i
         return Tensor(data).reshape(1,-1).int().to(device), data_i
      
      def make_x_0(toks, start_i: int) -> Tensor:
         assert start_i < len(toks)
         return model.make_x_0_from(Tensor(toks[start_i:]).int().reshape(1,-1).to(device))

      toks = encode(all_output) # type: ignore
      start_i = len(toks) - DS
      assert start_i > 0, f"input size {len(toks)} must be atleast 1 greater than decoder head size {DS}"
      x_0 = make_x_0(toks, start_i).to(**TO)
      den_start_amount = timestep_reduce
      X, data_i = make_X(toks, start_i)

      for i in trange(count):

         alphas = np.ones((DS,), dtype=np.float32)
         timesteps = np.zeros((DS,), dtype=np.float32)
         for i in range(DS):
            ts = min(den_start_amount + i*TD, TS-1)
            alphas[i] = all_alphas[int(ts)]
            timesteps[i] = ts
         alphas = Tensor(alphas).reshape(1,1,DS,1).to(**TO) # type: ignore
         attn_mask = torch.ones(CS,CS).tril(0).bool().reshape(1,CS,1,CS).expand(BS,CS,DS,CS)[:,data_i-1:data_i,:,:]
         attn_mask = Tensor(attn_mask.cpu().numpy()).to(**TO)

         x_t = (x_0*alphas + torch.randn(BS,1,Config.model_params.den_dim, dtype=torch.float32)*(1-alphas)).to(**TO)

         e_t = model(x_t, X, Tensor(timesteps).int().to(device), attn_mask)[0]
         pred_x_0 = x_t - e_t

         while den_start_amount <= timestep_reduce:
            if start_i >= len(toks):
               probs_np = model.estimate(pred_x_0, temperature=temperature, use_log=False)[0,0,0,:].cpu().numpy()
               tok = int(np.random.choice(len(probs_np), p=probs_np))
               toks.append(tok)
            start_i += 1
            X, data_i = make_X(toks, start_i)
            pred_x_0 = torch.cat([pred_x_0[:,:,1:], torch.zeros(*pred_x_0.shape[:2],1,pred_x_0.shape[-1]).to(**TO)], dim=-2)
            den_start_amount += TD
         den_start_amount -= timestep_reduce

         if start_i < len(toks):
            new_x_0 = make_x_0(toks, start_i)
            pred_x_0[:,:,:len(toks)-start_i] = new_x_0.reshape((1,*new_x_0.shape)).detach()

         x_0 = pred_x_0.detach()

      output = ""
      for tok in toks:
         try:
            if tok >= 50257:
               tok = 10
            output += decode([tok]).decode() # type: ignore
         except Exception:
            output += "<?>"
      return output

def deep_test_den(data, model:FusedTransformer, iterations:int=16, timestep_reduce:int=8, start_index:int=Config.model_params.ctx_pos_size//2) -> Tuple[float,Tuple[float,float,float]]:
   acc = 0
   probs = []
   all_alphas = make_alphas()
   BS = 1
   TS = Config.model_params.timesteps
   CS = Config.model_params.ctx_pos_size
   DS = Config.model_params.den_pos_size
   TD = Config.model_params.time_deltas

   with torch.no_grad():
      for iteration in trange(iterations):
         np.random.seed(iteration)
         torch.manual_seed(iteration)
         offset = np.random.randint(0, data.shape[0]-CS-DS, dtype=np.int32)
         X = Tensor(data[offset:offset+CS].astype(int)).int().to(device)
         x_0 = model.make_x_0_from(Tensor(np.array([data[offset+i:offset+i+DS] for i in range(1,CS+1)]).astype(int)).int().to(device))
         x_0 = x_0.reshape(1,*x_0.shape)

         den_index = 0
         den_start_amount = timestep_reduce
         while den_index < DS:
            overwrite = (DS-den_index-1)
            Y = Tensor(np.array([data[offset+den_index+i:offset+den_index+i+overwrite] for i in range(1,CS+1)], dtype=np.int32)).int().to(device)
            x_0[:,:,:overwrite,:] = model.make_x_0_from(Y)

            alphas = np.ones((DS,), dtype=np.float32)
            timesteps = np.zeros((DS,), dtype=np.float32)
            for i in range(DS):
               ts = min(den_start_amount + i*TD, TS-1)
               alphas[i] = all_alphas[int(ts)]
               timesteps[i] = ts
            alphas = Tensor(alphas).reshape(1,1,DS,1).to(**TO) # type: ignore
            attn_mask = torch.ones(CS,CS).tril(0).bool().reshape(1,CS,1,CS).expand(BS,CS,DS,CS)
            attn_mask = Tensor(attn_mask.cpu().numpy()).to(**TO)

            x_t = (x_0*alphas + torch.randn(BS,1,Config.model_params.den_dim, dtype=torch.float32)*(1-alphas)).to(**TO)

            e_t = model(x_t, X, Tensor(timesteps).int().to(device), attn_mask)[0]
            pred_x_0 = x_t - e_t

            while den_start_amount <= timestep_reduce:
               den_index += 1
               first_x_0 = pred_x_0[:,:,0]
               pred_x_0 = torch.cat([pred_x_0[:,:,1:], torch.zeros(*pred_x_0.shape[:2],1,pred_x_0.shape[-1])], dim=-2).to(**TO)
               den_start_amount += TD
               X = Tensor(data[offset+den_index:offset+den_index+CS].astype(int)).int().to(device)
            den_start_amount -= timestep_reduce

            x_0 = pred_x_0

         Y = Tensor(data[offset+DS:offset+DS+CS].astype(int)).int()[start_index:].to(device)
         probs_y = model.estimate(first_x_0, 1.0, False)[:,start_index:]
         probs.append(np.array([probs_y[0,i,Y[i]].cpu().numpy().item() for i in range(Y.shape[0])]))

         if False:
            text = ""
            for i in range(X.shape[0]):
               byte = decode([X[i].cpu().numpy().item()]) # type: ignore
               try:
                  char = byte.decode()
               except Exception:
                  char = "<?>"
               text += char if (i != X.shape[0]-1) else f"<|{char}|>"
            print(text)
            
            probs_np = probs_y.cpu().numpy()[0,-1,:]
            sort_idx = np.argsort(probs_np)

            top_n = 16
            top_idx = sort_idx[:top_n:-1]
            for i in range(top_n):
               idx = top_idx[i]
               prob = probs_np[idx]
               byte = decode([idx]) # type: ignore
               try:
                  char = byte.decode()
               except Exception:
                  char = "<?>"
               print(f"{idx: >5d}: {prob:.8f} |{char}|")

         pred_y = probs_y.argmax(dim=-1)
         acc += (Y == pred_y).float().mean().cpu().numpy().item()
   
   probs_np = np.array(probs).flatten()
   return acc / iterations, [np.percentile(probs_np, p) for p in [75, 50, 25]] # type: ignore

if __name__ == "__main__":
   train(phase=1, recover=False)
   # print(generate_ctx(count=16))

   # train(phase=2, recover=False)
   # train(phase=3, recover=False)
   # print(generate_den(count=64, temperature=0.4))
