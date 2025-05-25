from tinygrad import Tensor, dtypes, nn, Device, TinyJit
from typing import List
import numpy as np
import os

from common import (
   load_tokenizer, make_filename, cosine_similarity,
   ENTRIES_PER_FILE,
)

LOAD_SLICES: int = 1024 * 16
LOAD_SIZE:   int = ENTRIES_PER_FILE // LOAD_SLICES
assert LOAD_SLICES * LOAD_SIZE == ENTRIES_PER_FILE

WIKI_ROOT     = "/raid/datasets/wikipedia/sentence-embedding"
FINE_WEB_ROOT = "/raid/datasets/fineweb/sentence-embedding"

GPUS = tuple(f"{Device.DEFAULT}:{i}" for i in range(1,5))

def main():
   tokenizer = load_tokenizer()

   ######################
   #   Load Wiki Data   #
   ######################
   wiki_list = {} # type: ignore
   for i in range(1024):
      filepath = os.path.join(WIKI_ROOT, make_filename(i))
      if not os.path.exists(filepath):
         break
      sd = nn.state.safe_load(filepath)
      for k, v in sd.items():
         if k not in wiki_list:
            wiki_list[k] = []
         new_value = v.to(Device.DEFAULT)
         if k == "emb":
            new_value = new_value.reshape(-1, len(GPUS), new_value.shape[-1]).shard(GPUS, axis=1).realize()
         wiki_list[k].append(new_value)
   wiki = {}
   for k, v in wiki_list.items():
      wiki[k] = Tensor.cat(*v).realize() # type: ignore
      print(f"{k}: {wiki[k].shape}")

   wiki_emb: Tensor = wiki["emb"]
   wiki_tok: Tensor = wiki["tok"]

   @TinyJit
   def get_closest_wiki_entry(emb:Tensor) -> Tensor:
      sim = cosine_similarity(emb.shard(GPUS), wiki_emb).to(wiki_tok.device).realize()
      sim.to_(wiki_tok.device).realize()
      return wiki_tok[sim.reshape(emb.shape[0], -1).argmax(axis=-1)].realize()

   #####################
   #   Load Fine Web   #
   #####################
   for i in range(1024):
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
