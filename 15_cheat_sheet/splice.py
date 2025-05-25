from tinygrad import Tensor, dtypes, nn, Device, TinyJit
from typing import List, Tuple
import numpy as np
import os

from common import (
   load_tokenizer, make_filename, cosine_similarity,
   ENTRIES_PER_FILE,
)

LOAD_SLICES: int = 1024 * 4
LOAD_SIZE:   int = ENTRIES_PER_FILE // LOAD_SLICES
assert LOAD_SLICES * LOAD_SIZE == ENTRIES_PER_FILE

WIKI_ROOT     = "/raid/datasets/wikipedia/sentence-embedding"
FINE_WEB_ROOT = "/raid/datasets/fineweb/sentence-embedding"

EMB_GPUS = tuple(f"{Device.DEFAULT}:{i}" for i in range(1,5))
# TOK_GPUS = tuple(f"{Device.DEFAULT}:{i}" for i in range(4,6))

def main():
   tokenizer = load_tokenizer()

   ######################
   #   Load Wiki Data   #
   ######################
   wiki_list = {} # type: ignore
   dtype_dev_map = {
      "emb": (dtypes.float16, EMB_GPUS),
      # "tok": (dtypes.int32,   TOK_GPUS),
   }
   tok_disktensors = []
   for i in range(1024):
      filepath = os.path.join(WIKI_ROOT, make_filename(i))
      if not os.path.exists(filepath):
         break
      sd = nn.state.safe_load(filepath)
      for k, v in sd.items():
         if k == "tok":
            tok_disktensors.append(v)
            continue
         if k not in dtype_dev_map:
            continue
         target_dtype, gpus = dtype_dev_map[k]
         if k not in wiki_list:
            wiki_list[k] = []
         new_value = v.reshape(-1, len(gpus), v.shape[-1]).shard(gpus, axis=1).cast(target_dtype).realize()
         wiki_list[k].append(new_value)
   wiki = {}
   for k, v in wiki_list.items():
      wiki[k] = Tensor.cat(*v).realize() # type: ignore
      print(f"{k}: {wiki[k].shape}")

   wiki_emb: Tensor = wiki["emb"]
   # wiki_tok: Tensor = wiki["tok"]

   @TinyJit
   def _get_closest_wiki_entry_index(emb:Tensor) -> Tuple[Tensor,Tensor]:
      sim = cosine_similarity(emb.unsqueeze(1).unsqueeze(1).shard(EMB_GPUS), wiki_emb.unsqueeze(0)).to(Device.DEFAULT).realize()
      sim = sim.to_(Device.DEFAULT).realize()
      sim = sim.reshape(emb.shape[0], -1)
      print(sim.shape)
      argmax = sim.argmax(axis=-1)
      print(argmax.shape)
      return argmax.realize(), sim[:,argmax].mul(Tensor.eye(argmax.shape[0])).sum(axis=-1).realize()

   def get_closest_wiki_entry(emb:Tensor) -> Tensor:
      index, sims = _get_closest_wiki_entry_index(emb.realize())
      index = index.numpy()
      print(index)
      print(sims.numpy())
      toks = []
      for i in index:
         i = int(i)
         toks.append(tok_disktensors[i // ENTRIES_PER_FILE][i % ENTRIES_PER_FILE].to(Device.DEFAULT))
      return Tensor.stack(*toks).realize()

   #####################
   #   Load Fine Web   #
   #####################
   for i in range(1,1024):
      filepath = os.path.join(FINE_WEB_ROOT, make_filename(i))
      if not os.path.exists(filepath):
         break
      sd = nn.state.safe_load(filepath)
      fine_web_emb = sd["emb"].to(Device.DEFAULT).realize()
      fine_web_tok = sd["tok"].to(Device.DEFAULT).numpy()
      for k in range(LOAD_SLICES):
         wiki_tokens = get_closest_wiki_entry(fine_web_emb[k*LOAD_SIZE:(k+1)*LOAD_SIZE].contiguous().realize())
         fine_web_tokens = fine_web_tok[k*LOAD_SIZE:(k+1)*LOAD_SIZE]
         wiki_text     = tokenizer.Decode(wiki_tokens.numpy().tolist())
         fine_web_text = tokenizer.Decode(fine_web_tokens.tolist())
         for wiki_row, fine_web_row in zip(wiki_text, fine_web_text):
            print("\n"+">"*40+" fine web\n"+fine_web_row+"\n"+"="*40+" wiki\n"+wiki_row+"\n"+"<"*40+"\n")
         return


if __name__ == "__main__":
   main()
