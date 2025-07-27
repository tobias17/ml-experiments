from tinygrad import Tensor, nn, dtypes, TinyJit, Device
from tinygrad.helpers import prod
from dataclasses import dataclass, asdict, field
import matplotlib.pyplot as plt # type: ignore
from typing import List, Dict, Tuple
import os, json, datetime, time, math
import numpy as np

from model import ModelConfig, Model
from common import (
   make_filename, loglerp, ENTRIES_PER_FILE, BLOCK_SIZE,
   dataset_root, SPLICED_DIRNAME, compress, fmt_percent, fmt_digits, fmt_time,
)


@dataclass
class TrainingData:
   train_loss: List[float] = field(default_factory=lambda: [])
   train_acc:  List[float] = field(default_factory=lambda: [])
   weight_files: List[str] = field(default_factory=lambda: [])
   dataset_i: int = 0
   step_i: int = 0
   run_time: float = 0.0

   @staticmethod
   def from_json(data:Dict) -> 'TrainingData': return TrainingData(**data)
   def to_json(self) -> Dict: return asdict(self)

class DataLoader:
   def __init__(self):
      root = dataset_root() / f"{SPLICED_DIRNAME}-best1"
      assert root.exists(), f"Failed to find training data, searched for {root}"
      self.entries: List[Dict[str,Tensor]] = []
      for i in range(1024):
         filepath = root / make_filename(i)
         if not filepath.exists():
            break
         self.entries.append(nn.state.safe_load(filepath))
      assert len(self.entries) > 0, f"Found 0 entries in {root}"
      self.max_index = len(self.entries) * ENTRIES_PER_FILE
   def get(self, index:int, amount:int) -> Tuple[Tensor,Tensor]:
      page_index  = index // ENTRIES_PER_FILE
      assert page_index <= len(self.entries), f"Requested {index=} which would index page {page_index} but only {len(self.entries)} pages were found"
      entry_index = index %  ENTRIES_PER_FILE
      assert entry_index + amount <= ENTRIES_PER_FILE, f"Requested {index=} and {amount=} which overlaps pages, {entry_index} + {amount} > {ENTRIES_PER_FILE}"
      page = self.entries[page_index]
      def process(key:str) -> Tensor:
         return page[key][entry_index:entry_index+amount].contiguous().to(Device.DEFAULT).realize()
      return process("tok"), process("wiki")

TRAIN_DTYPE = dtypes.bfloat16
GPUS = tuple(f"{Device.DEFAULT}:{i}" for i in range(6))

DEVICE_BS = 2
GLOBAL_BS = DEVICE_BS * len(GPUS)
LR_A = 2**-16
LR_B = 2**-18

AVERAGE_EVERY = 500
GRAPH_EVERY   = 500
EVAL_EVERY    = 10000
SAVE_EVERY    = 10000

MAX_DATASET_ENTRIES = 18_000_000
MAX_KEEP_WEIGHTS = 2

CONFIGS = {
   "cheat_sheet":     ModelConfig(cross_attn=True,  n_layers=24),
   "baseline_ctx":    ModelConfig(cross_attn=False, n_layers=32, ff_mult=6.0, ctx_length=1024),
   "baseline_no_ctx": ModelConfig(cross_attn=False, n_layers=40, ff_mult=8.0),
}

def get_model(cfg_name:str, print_params:bool=True) -> Model:
   cfg = CONFIGS[cfg_name]
   model = Model(cfg)

   if print_params:
      params = nn.state.get_state_dict(model)
      print(f"{cfg_name}: {compress(sum(prod(p.shape) for p in params.values()), ['k','m','b'])}") # type: ignore
   print()

   return model


def train(cfg_name:str, restore:str|None=None, keep_all_weights:bool=False, only_bake:bool=False) -> None:
   dtypes.default_float = TRAIN_DTYPE

   data_loader = DataLoader()
   print()
   if only_bake: print("Only baking...")
   print(f"Max dataset size : {data_loader.max_index}\nMax dataset usage: {MAX_DATASET_ENTRIES}")

   model = get_model(cfg_name)

   if restore is not None:
      weights_folder = os.path.realpath(restore)
      data_path = os.path.realpath(os.path.join(weights_folder, "data.json"))
      assert os.path.isfile(data_path), f"failed to find data json at the restore path, searched for {data_path}"
      with open(data_path, "r") as f:
         data = TrainingData.from_json(json.load(f))
      filename = data.weight_files[-1]
      weight_path = os.path.join(weights_folder, filename)
      assert os.path.exists(weight_path), f"failed to find weights path for restore model, searched for {weight_path}"
      nn.state.load_state_dict(model, nn.state.safe_load(weight_path))
   else:
      weights_folder = f"weights/{os.path.basename(os.path.dirname(__file__))}/{cfg_name}_{datetime.datetime.now()}".replace(" ", "_").replace(":", "_").replace("-", "_").replace(".", "_")
      data = TrainingData()

   params = nn.state.get_parameters(model)
   for p in params:
      p.shard_(GPUS)
   optim = nn.optim.AdamW(params, lr=LR_A, eps=1e-5)

   def spread(x:Tensor, device:str|Tuple[str,...]) -> Tensor:
      return (x.to(device)) if isinstance(device, str) else (x.shard(device, axis=0))

   @TinyJit
   def train_step(tok:Tensor, ctx:Tensor|None) -> Tuple[Tensor,Tensor]:
      combined = cfg_name.startswith("baseline_") and ctx is not None
      if combined:
         assert tok.ndim == 2 and ctx is not None
         dev_tok = spread(Tensor.cat(ctx, tok, dim=-1), model.device)
         dev_ctx = None
      else:
         dev_tok = spread(tok, model.device)
         dev_ctx = spread(ctx, model.device) if ctx is not None else None
      logits = model(dev_tok[:,:-1], dev_ctx)
      if combined:
         assert ctx is not None
         y = dev_tok[:,1+ctx.shape[-1]:]
         logits = logits[:,int(ctx.shape[-1]):]
      else:
         y = dev_tok[:,1:]
      optim.zero_grad()
      loss = logits.sparse_categorical_crossentropy(y).backward()
      optim.step()
      acc = Tensor.cast(logits.argmax(-1) == y, TRAIN_DTYPE).mean().realize()
      return loss, acc

   end_time = None
   loss_vs = np.zeros(AVERAGE_EVERY)
   acc_vs  = np.zeros(AVERAGE_EVERY)
   while True:
      # Configure
      Tensor.manual_seed(data.step_i)
      total_time = time.perf_counter() if end_time is None else end_time
      start_time = time.perf_counter()

      # Overrun check
      if data.dataset_i >= MAX_DATASET_ENTRIES or (only_bake and data.step_i >= 10):
         print(f"\nTrained on {data.dataset_i*BLOCK_SIZE} tokens\n")
         break

      # Set the optime LR for decaying items
      new_lr = loglerp(LR_A, LR_B, data.dataset_i / MAX_DATASET_ENTRIES)
      optim.lr.assign(Tensor([new_lr], device=optim.lr.device, dtype=optim.lr.dtype))

      # Perform training set
      vs_idx = data.step_i % AVERAGE_EVERY
      with Tensor.train():
         tok, ctx = data_loader.get(data.dataset_i, GLOBAL_BS)
         ctx = None if cfg_name.endswith("no_ctx") else ctx.realize() # type: ignore
         loss, acc = train_step(tok.realize(), ctx)
         loss_vs[vs_idx] = loss.cast(dtypes.float32).item()
         acc_vs [vs_idx] = acc .cast(dtypes.float32).item()
      step_time = time.perf_counter() - start_time

      # Average data
      if data.step_i > 0 and data.step_i % AVERAGE_EVERY == 0:
         data.train_loss.append(loss_vs.mean())
         data.train_acc .append(acc_vs .mean())

      # Create the Plots
      if data.step_i > 0 and data.step_i % GRAPH_EVERY == 0:
         def plot(items:List[float], title:str, ymax:float|None=None):
            ylabel = title
            xs = (np.arange(len(items))+1)*AVERAGE_EVERY*GLOBAL_BS*BLOCK_SIZE / 1_000_000_000.0
            ys = np.array(items)
            plt.clf()
            max_95th = 0.0
            if ymax is None:
               max_95th = max(max_95th, np.percentile(ys, 95))
            plt.plot(xs, ys, label=cfg_name)
            plt.ylim((0, max_95th*1.2 if ymax is None else ymax))
            plt.xlim((0,None))
            plt.xlabel("Tokens (Billions)")
            plt.ylabel(ylabel)
            plt.title(title)
            plt.legend()
            figure = plt.gcf()
            figure.set_size_inches(18, 10)
            if not os.path.exists(weights_folder): os.makedirs(weights_folder)
            plt.savefig(os.path.join(weights_folder, f"graph_{title.lower().replace(' ','_')}.png"), dpi=100)
         plot(data.train_loss, "Loss")
         plot(data.train_acc,  "Acc", ymax=1.0)

      # Print the step line
      rem = max(0, MAX_DATASET_ENTRIES - (data.dataset_i + GLOBAL_BS))
      print(" | ".join([
         f"{data.step_i: 9d}",
         "Dataset: " + fmt_percent(data.dataset_i/MAX_DATASET_ENTRIES),
         f"{step_time*1000.0:.0f} ms",
         f"Loss: {fmt_digits(loss_vs[vs_idx], 5)}",
         f"Acc: {fmt_percent(acc_vs[vs_idx])}",
         f"LR: {math.log2(new_lr):.3f}",
         f"Run: {fmt_time(data.run_time)}",
         f"Eta: {fmt_time(0 if data.dataset_i == 0 else (rem * data.run_time / data.dataset_i))}",
      ]))

      # Increment Counters
      data.step_i += 1
      data.dataset_i += GLOBAL_BS
      end_time = time.perf_counter()
      data.run_time += (end_time - total_time)

      if data.step_i - 1 > 0 and (data.step_i - 1) % SAVE_EVERY == 0:
         # Save All of the Models Weights
         if not os.path.exists(weights_folder):
            os.makedirs(weights_folder)
         new_weight_file = f"model_{data.step_i//1000:04d}k.st"
         nn.state.safe_save(nn.state.get_state_dict(model), os.path.join(weights_folder, new_weight_file))
         data.weight_files.append(new_weight_file)

         # Potentially Purge the Last Weights Saved
         if not keep_all_weights and len(data.weight_files) > MAX_KEEP_WEIGHTS:
            filename = data.weight_files.pop(0)
            if filename is not None:
               path = os.path.join(weights_folder, filename)
               if path == new_weight_file:
                  print("WARNING: weights would have overwrote themselves, skipping purge step")
               else:
                  print(f"Saving {path}")
                  try: os.remove(path)
                  except Exception as ex: print(f"WARNING: ran into error deleting old weights file, {ex}")

         # Save a Data Json
         with open(os.path.join(weights_folder, f"data.json"), "w") as f:
            json.dump(data.to_json(), f)


if __name__ == "__main__":
   import argparse
   parser = argparse.ArgumentParser()
   parser.add_argument('cfg_name', choices=list(CONFIGS.keys()))
   parser.add_argument('--restore', type=str)
   parser.add_argument('--only-bake', action='store_true')
   args = parser.parse_args()

   train(args.cfg_name, args.restore, only_bake=args.only_bake)
