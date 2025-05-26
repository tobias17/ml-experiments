from tinygrad import Tensor, nn, dtypes, TinyJit, Context, Device
from tinygrad.helpers import prod
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
import os, json, datetime, time
import matplotlib.pyplot as plt # type: ignore
import numpy as np

from util import compress
from model import ModelConfig, Model
from common import make_filename, ENTRIES_PER_FILE


BASELINE_SMALL = ModelConfig(cross_attn=False, n_layers=20)
CHEAT_SHEET    = ModelConfig(cross_attn=True,  n_layers=20)
BASELINE_LARGE = ModelConfig(cross_attn=False, n_layers=32)


@dataclass
class TrainingData:
   train_losses: Dict[str,List[float]]
   train_accs: Dict[str,List[float]]
   last_weight_files: Dict[str,str]
   dataset_i: int = 0
   step_i: int = 0

   @staticmethod
   def from_json(data:Dict) -> 'TrainingData': return TrainingData(**data)
   def to_json(self) -> Dict: return asdict(self)

def get_models(print_params:bool=True) -> Dict[str,Model]:
   models = {
      "baseline_small": Model(BASELINE_SMALL),
      "cheat_sheet":    Model(CHEAT_SHEET),
      "baseline_large": Model(BASELINE_LARGE),
   }

   if print_params:
      print()
      for name, model in models.items():
         params = nn.state.get_state_dict(model)
         pad = " "*(max(map(len, models.keys())) - len(name))
         print(f"{name}:{pad} {compress(sum(prod(p.shape) for p in params.values()), ['k','m','b'])}") # type: ignore
   print()

   return models

DATASET_ROOT = "/raid/datasets/fineweb/with-wiki-top1"
class DataLoader:
   def __init__(self):
      self.entries: List[Dict[str,Tensor]] = []
      for i in range(1024):
         filepath = os.path.join(DATASET_ROOT, make_filename(i))
         if not os.path.exists(filepath):
            break
         self.entries.append(nn.state.safe_load(filepath))
      assert len(self.entries) > 0
      self.max_index = len(self.entries) * ENTRIES_PER_FILE
   def get(self, index:int, amount:int) -> Tuple[Tensor,Tensor]:
      page_index  = index // ENTRIES_PER_FILE
      assert page_index <= len(self.entries), f"Requested {index=} which would index page {page_index} but only {len(self.entries)} pages were found"
      entry_index = index %  ENTRIES_PER_FILE
      assert entry_index + amount <= ENTRIES_PER_FILE, f"Requested {index=} and {amount=} which overlaps pages, {entry_index} + {amount} > {ENTRIES_PER_FILE}"
      page = self.entries[page_index]
      def ret(key:str) -> Tensor:
         return page[key][entry_index:entry_index+amount].contiguous().to(Device.DEFAULT).realize()
      return ret("tok"), ret("wiki")


BS = 8
LR = 2**-16

TRAIN_BREAM   = 20
AVERAGE_EVERY = 500
GRAPH_EVERY   = 500
SAVE_EVERY    = 5000

MAX_DATASET_ENTRIES = 5_900_000


def train(restore:str|None=None, keep_all_weights:bool=False):
   Tensor.no_grad = False

   data_loader = DataLoader()
   print(f"\nMax dataset index: {data_loader.max_index}")
   models = get_models(print_params=True)

   if restore is not None:
      weights_folder = os.path.realpath(restore)
      data_path = os.path.realpath(os.path.join(weights_folder, "data.json"))
      assert os.path.isfile(data_path), f"failed to find data json at the restore path, searched for {data_path}"
      with open(data_path, "r") as f:
         data = TrainingData.from_json(json.load(f))
      for name, filename in data.last_weight_files.items():
         weight_path = os.path.join(weights_folder, filename)
         assert os.path.exists(weight_path), f"failed to find weights path for restore model, searched for {weight_path}"
         nn.state.load_state_dict(models[name], nn.state.safe_load(weight_path))
   else:
      weights_folder = f"weights/{os.path.basename(os.path.dirname(__file__))}/{datetime.datetime.now()}".replace(" ", "_").replace(":", "_").replace("-", "_").replace(".", "_")
      data = TrainingData({k:[] for k in models.keys()}, {k:[] for k in models.keys()}, {})

   optims: Dict[str,nn.optim.LAMB] = {}
   for i, (name, model) in enumerate(models.items()):
      params = nn.state.get_parameters(model)
      for p in params:
         p.shard_((f"{Device.DEFAULT}:{i*2}", f"{Device.DEFAULT}:{i*2+1}"))
      optims[name] = nn.optim.AdamW(params, lr=LR, eps=1e-7)

   @TinyJit
   def train_step(tok:Tensor, ctx:Tensor) -> Tuple[Dict[str,Tensor],Dict[str,Tensor]]:
      losses: Dict[str,Tensor] = {}
      accs:   Dict[str,Tensor] = {}
      for k, model in models.items():
         dev_tok = tok.to(model.device) if isinstance(model.device, str) else tok.shard(model.device, axis=0)
         dev_ctx = ctx.to(model.device) if isinstance(model.device, str) else ctx.shard(model.device, axis=0)
         logits = model(dev_tok[:,:-1], dev_ctx if model.cross_attn else None)
         y = dev_tok[:,1:]
         optims[k].zero_grad()
         losses[k] = logits.sparse_categorical_crossentropy(y).backward()
         optims[k].step()
         accs[k] = Tensor.cast(logits.argmax(-1) == y, dtypes.float32).mean().realize()
      return losses, accs

   loss_vs: List[Dict[str,float]] = []
   acc_vs:  List[Dict[str,float]] = []
   with Tensor.train():
      while True:
         # Configure Step
         start_time = time.perf_counter()
         Tensor.manual_seed(data.step_i)

         # Overrun check
         if data.dataset_i >= MAX_DATASET_ENTRIES:
            print(f"Trained on {data.dataset_i} tokens")

         # Perform Training Step
         tok, ctx = data_loader.get(data.dataset_i, BS)
         with Context(BEAM=TRAIN_BREAM):
            loss, acc = train_step(tok.realize(), ctx.realize())
            loss_vs.append({ k:l.item() for k,l in loss.items()})
            acc_vs .append({ k:a.item() for k,a in acc .items()})

         # Increment Counters
         data.step_i += 1
         data.dataset_i += BS
         print(
            f"{data.step_i: 10d} | Dataset {100.0*min(1.0,data.dataset_i/MAX_DATASET_ENTRIES):.2f}% | {1000.0*(time.perf_counter()-start_time):.0f} ms" +
            " | Train Loss " + " - ".join(f"{l:.4f}" for l in loss_vs[-1].values()) +
            " | Train Acc "  + " - ".join(f"{100.0*a:.2f}%" for a in acc_vs[-1].values())
         )

         # Average the Loss Data
         if data.step_i > 0 and data.step_i % AVERAGE_EVERY == 0:
            for k in models.keys():
               data.train_losses[k].append(sum(l[k] for l in loss_vs)/len(loss_vs))
               data.train_accs  [k].append(sum(a[k] for a in acc_vs )/len(acc_vs ))

         # Create the Plots
         if data.step_i > 0 and data.step_i % GRAPH_EVERY == 0:
            def plot(items, title:str, ymax:float|None=None):
               xs, ys = {}, {}
               for name in models.keys():
                  xs[name] = (np.arange(len(items[name]))+1)*AVERAGE_EVERY*BS
                  ys[name] = np.array(items[name])
               plt.clf()
               max_95th = 0.0
               for label in xs.keys():
                  if ymax is None:
                     max_95th = max(max_95th, np.percentile(ys[label], 95))
                  plt.plot(xs[label], ys[label], label=label)
               plt.ylim((0, max_95th*1.2 if ymax is None else ymax))
               plt.xlim((0,None))
               plt.title(title)
               plt.legend()
               figure = plt.gcf()
               figure.set_size_inches(18, 10)
               if not os.path.exists(weights_folder): os.makedirs(weights_folder)
               plt.savefig(os.path.join(weights_folder, f"graph_{title.lower()}.png"), dpi=100)
            plot(data.train_losses, "Loss")
            plot(data.train_accs,   "Acc", ymax=1.0)

         if data.step_i > 0 and data.step_i % SAVE_EVERY == 0:
            # Save All of the Models Weights
            new_weight_files = {}
            if not os.path.exists(weights_folder):
               os.makedirs(weights_folder)
            for key in models.keys():
               new_weight_files[key] = os.path.join(weights_folder, f"model_{data.step_i:04d}k_{key}.st")
               nn.state.safe_save(nn.state.get_state_dict(models[key]), new_weight_files[key])

            # Potentially Purge the Last Weights Saved
            if not keep_all_weights:
               for path in data.last_weight_files.values():
                  if path in new_weight_files.values():
                     print("WARNING: weights overwrote themselves, skipping purge step")
                  else:
                     print(f"Saving {path}")
                     try: os.remove(path)
                     except Exception as ex: print(f"WARNING: ran into error deleting old weights file, {ex}")
            data.last_weight_files = new_weight_files

            # Save a Data Json
            with open(os.path.join(weights_folder, f"data.json"), "w") as f:
               json.dump(data.to_json(), f)



if __name__ == "__main__":
   train()
