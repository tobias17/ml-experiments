from tinygrad import Tensor, nn, Variable, dtypes, TinyJit, Device # type: ignore
from tinygrad.nn.state import get_parameters # type: ignore
from tinygrad.helpers import prod, BEAM, Context # type: ignore
from extra.models.llama import TransformerBlock, Attention, precompute_freqs_cis, apply_rotary_emb, repeat_kv # type: ignore

from sentencepiece import SentencePieceProcessor # type: ignore
from typing import List, Dict, Union, Optional, Tuple
import datetime, os, time
import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore

import sys
sys.path.append(os.path.dirname(__file__))
from modelfile import create_models, MAX_CLUSTER_CONTEXT, CLUSTER_SIZE


def main():
   Tensor.manual_seed(42)
   Tensor.training = True
   BEAM_VALUE = BEAM.value
   BEAM.value = 0

   models = create_models()
   MODEL_CONFIGS = len(models)

   # Load Dataset
   X_train, X_val = [np.memmap(f"/raid/datasets/fineweb/tokenized/fineweb_{split}.bin", dtype=np.uint16, mode='r') for split in ('train', 'val')]

   GPUS = [f"{Device.DEFAULT}:{i}" for i in range(MODEL_CONFIGS)]

   item_names = [
      "enc",
      "gen",
      "dec",
   ]

   MULT = 1.0 / 1024 / 1024 / 1024
   params = []
   print("\nModel Parameters (B):")
   for i in range(MODEL_CONFIGS):
      model_params = get_parameters(models[i])
      for w in model_params:
         w.replace(w.to(GPUS[i])).realize()
      params.append(model_params)

      text = f"{i}: "
      total = 0
      for name in item_names:
         item_param_count = sum(prod(p.shape) for p in get_parameters(getattr(models[i], name))) * MULT
         text += f"{name}={item_param_count:.3f}, "
         total += item_param_count
      print(f"{text}total={total:.3f}")
   print("")

   # Define the Optimizer
   LEARNING_RATES = [
      2e-6,
      2e-7,
      2e-8,
      2e-9,
   ]
   optims = [nn.optim.AdamW(params[i], LEARNING_RATES[i]) for i in range(MODEL_CONFIGS)]

   # Define some Globals
   DEVICE_BS = 16
   GLOBAL_BS = DEVICE_BS
   TOKENS_CONTEXT_SIZE = MAX_CLUSTER_CONTEXT * CLUSTER_SIZE

   GRAPH_EVERY = 200
   SAVE_EVERY  = 10000

   # Define some Tracking Variables
   weights_folder = f"weights/{os.path.basename(os.path.dirname(__file__))}/{datetime.datetime.now()}".replace(" ", "_").replace(":", "_").replace("-", "_").replace(".", "_")
   train_losses = [dict() for _ in range(MODEL_CONFIGS)]

   @TinyJit
   def train_step(orig_tokens:Tensor) -> List[Tuple[Tensor,Dict[str,Tensor],Tensor]]:
      ret_val = []
      for i in range(MODEL_CONFIGS):
         losses, acc = models[i].training_loss(orig_tokens.to(GPUS[i]))
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
         start_time = time.time()
         Tensor.manual_seed(step_i)

         loss_vs, acc_vs = [], []
         with Context(BEAM=BEAM_VALUE):
            orig_batches = [Tensor(np.asarray(X_train[dataset_i + batch_i*TOKENS_CONTEXT_SIZE :dataset_i + (batch_i+1)*TOKENS_CONTEXT_SIZE]), dtype=dtypes.int32) for batch_i in range(GLOBAL_BS)]
            orig_tokens = Tensor.stack(*orig_batches)
            ret_vals = train_step(orig_tokens.realize())

            for i, (loss, losses, acc) in enumerate(ret_vals):
               loss_vs.append(loss.item())
               acc_vs.append(acc.item())
               for k,v in losses.items():
                  if k not in train_losses[i]:
                     train_losses[i][k] = []
                  train_losses[i][k].append(v.item())

         step_i += 1
         dataset_i += TOKENS_CONTEXT_SIZE * GLOBAL_BS

         if step_i > 0 and step_i % GRAPH_EVERY == 0:
            for i in range(MODEL_CONFIGS):
               plt.clf()
               x = np.arange(step_i)
               for label, y in train_losses[i].items():
                  plt.plot(x, y, label=label)
               plt.ylim((0,None))
               plt.title("Loss")
               plt.legend()
               figure = plt.gcf()
               figure.set_size_inches(18/1.5, 10/1.5)
               if not os.path.exists(weights_folder): os.makedirs(weights_folder)
               plt.savefig(os.path.join(weights_folder, f"graph_loss_c{i}.png"), dpi=100)
         
         if step_i > 0 and step_i % SAVE_EVERY == 0:
            if not os.path.exists(weights_folder): os.makedirs(weights_folder)
            for i in range(MODEL_CONFIGS):
               nn.state.safe_save(nn.state.get_state_dict(models[i]), os.path.join(weights_folder, f"model_{step_i//1000:04d}k_c{i}.st"))

         print(f"| {step_i-1:05d} | {1000.0*(time.time()-start_time):.0f} ms | Train Loss " + " - ".join(f"{l:.4f}" for l in loss_vs) + f" | Train Acc " + " - ".join(f"{100.0*a:.2f}%" for a in acc_vs) + " |")

if __name__ == "__main__":
   main()
