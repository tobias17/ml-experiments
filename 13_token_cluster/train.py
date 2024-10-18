from tinygrad import Tensor, nn, dtypes, TinyJit, Device # type: ignore
from tinygrad.helpers import prod, BEAM, Context # type: ignore

from typing import List, Dict, Tuple, Optional
import datetime, os, time, json
import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore
from dataclasses import dataclass, asdict

import sys
sys.path.append(os.path.dirname(__file__))
from modelfile import create_models, MAX_CLUSTER_CONTEXT, CLUSTER_SIZE


@dataclass
class TrainingData:
   train_losses: List[Dict[str,List[float]]]
   last_weight_files: List[str]

   @staticmethod
   def from_json(data:Dict) -> 'TrainingData': return TrainingData(**data)
   def to_json(self) -> Dict: return asdict(self)


def train_model(restore:Optional[str], cluster_loss:bool, decoded_loss:bool, predict_loss:bool, keep_all_weights:bool):
   Tensor.manual_seed(42)
   Tensor.training = True
   BEAM_VALUE = BEAM.value
   BEAM.value = 0

   models = create_models()
   MODEL_CONFIGS = len(models)
   GPUS_PER_MODEL = 1

   # Potentially Pick Up Old Weights
   if restore is not None:
      path = os.path.realpath(os.path.join(restore, "data.json"))
      assert os.path.isfile(path), f"failed to find data json at the restore path, searched for {path}"
      with open(path, "r") as f:
         data = TrainingData.from_json(json.load(f))
      for i, path in enumerate(data.last_weight_files):
         assert os.path.exists(path), f"failed to find weights path for restore model, searched for {path}"
         nn.state.load_state_dict(models[i], nn.state.safe_load(path))
   else:
      weights_folder = f"weights/{os.path.basename(os.path.dirname(__file__))}/{datetime.datetime.now()}".replace(" ", "_").replace(":", "_").replace("-", "_").replace(".", "_")
      data = TrainingData([dict() for _ in range(MODEL_CONFIGS)], [])

   # Load Dataset
   X_train, X_val = [np.memmap(f"/raid/datasets/fineweb/tokenized/fineweb_{split}.bin", dtype=np.uint16, mode='r') for split in ('train', 'val')]

   GPUS = [f"{Device.DEFAULT}:{i}" for i in range(6)]
   assert MODEL_CONFIGS * GPUS_PER_MODEL <= len(GPUS), f"{MODEL_CONFIGS} * {GPUS_PER_MODEL} > {len(GPUS)}"

   item_names = [
      "enc",
      "gen",
      "dec",
   ]

   MULT = 1.0 / 1024 / 1024 / 1024
   params = []
   print("\nModel Parameters (B):")
   for i in range(MODEL_CONFIGS):
      model_params = models[i].get_trainable_parameters(cluster_loss, decoded_loss, predict_loss)
      for w in model_params:
         w.replace(w.to(GPUS[i]) if GPUS_PER_MODEL == 1 else w.shard(GPUS[i*GPUS_PER_MODEL:(i+1)*GPUS_PER_MODEL])).realize()
      params.append(model_params)

      text = f"{i}: "
      total = 0
      for name in item_names:
         item_param_count = sum(prod(p.shape) for p in nn.state.get_parameters(getattr(models[i], name))) * MULT
         text += f"{name}={item_param_count:.3f}, "
         total += item_param_count
      print(f"{text}total={total:.3f}")
   print("")

   # Define the Optimizer
   LEARNING_RATES = [
      2e-5,
      2e-8,
      2e-11,
      2e-14,
   ]
   optims = [nn.optim.AdamW(params[i], LEARNING_RATES[i]) for i in range(MODEL_CONFIGS)]

   # Define some Globals
   DEVICE_BS = 12
   GLOBAL_BS = DEVICE_BS * GPUS_PER_MODEL
   TOKENS_CONTEXT_SIZE = MAX_CLUSTER_CONTEXT * CLUSTER_SIZE

   GRAPH_EVERY = 200
   SAVE_EVERY  = 10000

   @TinyJit
   def train_step(orig_tokens:Tensor) -> List[Tuple[Tensor,Dict[str,Tensor],Tensor]]:
      ret_val = []
      for i in range(MODEL_CONFIGS):
         dev_tokens = orig_tokens.to(GPUS[i]) if GPUS_PER_MODEL == 1 else orig_tokens.shard(GPUS[i*GPUS_PER_MODEL:(i+1)*GPUS_PER_MODEL], axis=0)
         losses, acc = models[i].compute_loss(dev_tokens)
         loss = sum(losses.values()).realize()
         optims[i].zero_grad()
         loss.backward()
         optims[i].step()

         ret_val.append((loss, losses, acc))
      return ret_val

   step_i = 0
   dataset_i = 0
   with Tensor.train():
      while True:
         # Configure Step
         start_time = time.time()
         Tensor.manual_seed(step_i)

         # Perform Training Step
         loss_vs, acc_vs = [], []
         with Context(BEAM=BEAM_VALUE):
            orig_batches = [Tensor(np.asarray(X_train[dataset_i + batch_i*TOKENS_CONTEXT_SIZE :dataset_i + (batch_i+1)*TOKENS_CONTEXT_SIZE]), dtype=dtypes.int32) for batch_i in range(GLOBAL_BS)]
            orig_tokens = Tensor.stack(*orig_batches)
            ret_vals = train_step(orig_tokens.realize())

            for i, (loss, losses, acc) in enumerate(ret_vals):
               loss_vs.append(loss.item())
               acc_vs.append(acc.item())
               for k,v in losses.items():
                  if k not in data.train_losses[i]:
                     data.train_losses[i][k] = []
                  data.train_losses[i][k].append(v.item())

         # Increment Counters
         step_i += 1
         dataset_i += TOKENS_CONTEXT_SIZE * GLOBAL_BS

         # Create the Plots
         if step_i > 0 and step_i % GRAPH_EVERY == 0:
            for i in range(MODEL_CONFIGS):
               plt.clf()
               x = np.arange(step_i)
               for label, y in data.train_losses[i].items():
                  plt.plot(x, y, label=label)
               plt.ylim((0,None))
               plt.title("Loss")
               plt.legend()
               figure = plt.gcf()
               figure.set_size_inches(18/1.5, 10/1.5)
               if not os.path.exists(weights_folder): os.makedirs(weights_folder)
               plt.savefig(os.path.join(weights_folder, f"graph_loss_c{i}.png"), dpi=100)
         
         if step_i > 0 and step_i % SAVE_EVERY == 0:
            # Save All of the Models Weights
            new_weight_files = []
            if not os.path.exists(weights_folder): os.makedirs(weights_folder)
            for i in range(MODEL_CONFIGS):
               new_weight_files.append(save_path := os.path.join(weights_folder, f"model_{step_i//1000:04d}k_c{i}.st"))
               nn.state.safe_save(nn.state.get_state_dict(models[i]), save_path)
            
            # Potentially Purge the Last Weights Saved
            if not keep_all_weights:
               for path in data.last_weight_files:
                  if path in new_weight_files:
                     print("WARNING: weights overwrote themselves, skipping purge step")
                  else:
                     try: os.remove(path)
                     except Exception as ex: print(f"WARNING: ran into error deleting old weights file, {ex}")
               data.last_weight_files = new_weight_files
            
            with open(os.path.join(weights_folder, f"data.json"), "w") as f:
               json.dump(data.to_json(), f)

         print(f"| {step_i-1:05d} | {1000.0*(time.time()-start_time):.0f} ms | Train Loss " + " - ".join(f"{l:.4f}" for l in loss_vs) + f" | Train Acc " + " - ".join(f"{100.0*a:.2f}%" for a in acc_vs) + " |")

if __name__ == "__main__":
   import argparse
   parser = argparse.ArgumentParser()
   parser.add_argument('-r', '--restore-folder', type=str)
   parser.add_argument('--cluster-loss', action='store_true')
   parser.add_argument('--decoded-loss', action='store_true')
   parser.add_argument('--predict-loss', action='store_true')
   parser.add_argument('--keep-all-weights', action='store_true')
   args = parser.parse_args()
   train_model(args.restore, args.cluster_loss, args.decoded_loss, args.predict_loss, args.keep_all_weights)
