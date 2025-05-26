from tinygrad import Tensor, nn, dtypes, TinyJit, Device
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
import os, json, datetime, time

from model import ModelConfig, Model
from common import get_latest_folder


BASELINE_SMALL = ModelConfig(cross_attn=False)
CHEAT_SHEET    = ModelConfig(cross_attn=True)
BASELINE_LARGE = ModelConfig(cross_attn=False, n_layers=32)


@dataclass
class TrainingData:
   train_losses: List[List[float]]
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
         print(f"{name}:{pad} {compress(sum(prod(p.shape) for p in params), ['k','m','b'])}") # type: ignore
   print()

   return models


BS = 1
LR = 2**-18


def train(restore:str|None=None):
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
      data = TrainingData([[] for _ in range(len(models))], {})

   optims = { k: nn.optim.AdamW(nn.state.get_parameters(model), lr=LR) for k, model in models.items() }

   @TinyJit
   def train_step(tok:Tensor, ctx:Tensor) -> Tuple[Dict[str,Tensor],Dict[str,Tensor]]:
      losses, accs = {}, {}
      for k, model in models.items():
         optims[k].zero_grad()
         dev_tok, dev_ctx = tok.to(model.device), ctx.to(model.device)
         y = model(dev_tok[:,:-1], dev_ctx)
         losses[k] = y.sparse_categorical_crossentropy(target := dev_tok[:,1:]).backward()
         optims[k].step()
         accs[k] = Tensor.cast(y.argmax(-1) == target, dtypes.float32).mean().realize()
      return losses, accs

   with Tensor.train():
      while True:
         # Configure Step
         start_time = time.perf_counter()
         Tensor.manual_seed(data.step_i)

         loss_vs, acc_vs = [], []


if __name__ == "__main__":
   train()
