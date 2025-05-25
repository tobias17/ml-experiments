

from datasets import load_dataset # type: ignore
from tqdm import tqdm
import os, json

from common import (
   split_list_with_overlap, load_tokenizer, load_sentence_model, cosine_similarity, make_filename,
   BLOCK_SIZE, TARGET_OVERLAP, BLOCKS_PER_BATCH, ENTRIES_PER_FILE,
)

DEVICE_ID = 1

OUT_ROOT = "/raid/datasets/wikipedia/sentence-embedding"
if not os.path.exists(OUT_ROOT):
   os.makedirs(OUT_ROOT)

SPLITTERS = ["See also", "References", "Citations", "Notes and references", "Further reading", "External links", "Notes"]
def clean_text(text:str) -> str|None:
   mintext = text
   for base in SPLITTERS:
      for suffix in ["", " ", "  "]:
         splitter = f"\n\n{base}{suffix}\n\n"
         if splitter in text:
            newtext = text.split(splitter, 1)[0]
            if len(newtext) < len(mintext):
               mintext = newtext
   return None if len(mintext) == len(text) else mintext


def create():
   from safetensors.torch import save_file
   import torch

   model = load_sentence_model(DEVICE_ID)
   tokenizer = load_tokenizer()

   ds = load_dataset("wikimedia/wikipedia", "20231101.en")
   train_ds = ds["train"]

   info_file = os.path.join(OUT_ROOT, "index.json")
   if os.path.exists(info_file):
      with open(info_file) as f:
         info = json.load(f)
      assert info["size"] == ENTRIES_PER_FILE, f'size {info["size"]} != {ENTRIES_PER_FILE} in cached files'
   else:
      info = { "i":-1, "size":ENTRIES_PER_FILE, "blobs":[] }

   # print(train_ds.select([20])["title"])
   # return

   idx = []
   tok = []
   blk = []
   emb = [] # type: ignore

   i = -1
   for row in tqdm(train_ds, disable=False):
      i += 1
      if info["i"] >= i:
         continue
      # if i >= 10:
      #    break

      title: str = row["title"]
      orig = row["text"]
      text = clean_text(orig)
      if text is None:
         # print(f"WARNING: entry '{title}' did not contain splitter, skipping")
         continue
      
      tokens = tokenizer.Encode(text)
      chunks = split_list_with_overlap(tokens, BLOCK_SIZE, TARGET_OVERLAP)
      if chunks is None:
         continue

      # print(f"Row {i}: '{title}', text length: {len(orig)} -> {len(text)}, token count: {len(tokens)}, k: {len(chunks)}")

      for chunk in chunks:
         idx.append(i)
         tok.append(chunk)
         text = tokenizer.Decode(chunk)
         assert text is not None
         blk.append(text)
      
         if len(blk) >= BLOCKS_PER_BATCH:
            emb += model.encode(blk, convert_to_tensor=True)
            blk = []

            if len(emb) >= ENTRIES_PER_FILE:
               filename = make_filename(len(info['blobs']))
               info["i"] = i
               info["blobs"].append(filename)
               data = {
                  "idx": torch.Tensor(idx).int(),
                  "tok": torch.Tensor(tok).int(),
                  "emb": torch.stack(emb).half(),
               }
               save_file(data, os.path.join(OUT_ROOT, filename))
               with open(info_file, "w") as f:
                  json.dump(info, f)
               print(f"Data contains idx[{data['idx'].shape}] tok[{data['tok'].shape}] emb[{data['emb'].shape}] rows")
               print(f"Saved embeddings to {filename}")

               idx = []
               tok = []
               emb = []


def use():
   from tinygrad import Tensor, nn, dtypes, Device
   Device.DEFAULT = "CUDA"
   GPUS = tuple(f"{Device.DEFAULT}:{i}" for i in range(1,5))

   model = load_sentence_model(DEVICE_ID)
   tokenizer = load_tokenizer()

   data_list = {} # type: ignore
   for i in range(128):
      filepath = os.path.join(OUT_ROOT, make_filename(i))
      if not os.path.exists(filepath):
         break
      sd = nn.state.safe_load(filepath)
      for k, v in sd.items():
         if k not in data_list:
            data_list[k] = []
         new_value = v.to(Device.DEFAULT)
         if k == "emb":
            new_value = new_value.cast(dtypes.float16).reshape(-1, len(GPUS), new_value.shape[-1]).shard(GPUS, axis=1).realize()
         data_list[k].append(new_value)
   data_cat = {}
   for k, v in data_list.items():
      data_cat[k] = Tensor.cat(*v).realize() # type: ignore
      print(f"{k}: {data_cat[k].shape}")

   ref = data_cat["emb"]
   tok = data_cat["tok"].numpy()
   print(f"{ref.shape=}")

   text = "This is the tale, of Captain Jack Sparrow!"
   emb = Tensor(model.encode(text), dtype=dtypes.float16).rearrange('d -> 1 d').shard(GPUS).realize()
   print(f"{emb.shape=}")

   sim = cosine_similarity(emb, ref).to(Device.DEFAULT).flatten().realize()
   print(f"{sim.shape=}")  

   TOP_K = 10
   top_val, top_ind = [v.numpy() for v in sim.topk(TOP_K)]
   for i in range(TOP_K):
      ref_text = tokenizer.Decode(tok[int(top_ind[i])].tolist())
      print(f"\nRank: {i+1}\nSim:  {top_val[i]:.4f}\n" + ">"*40 + f"\n{ref_text}\n" + "<"*40 + "\n")


if __name__ == "__main__":
   fnx_map = {
      "create": create,
      "use": use,
   }
   import argparse
   parser = argparse.ArgumentParser()
   parser.add_argument("--run", type=str, choices=list(fnx_map.keys()), default="create")
   args = parser.parse_args()

   fnx_map[args.run]()
