from sentence_transformers import SentenceTransformer
from sentencepiece import SentencePieceProcessor # type: ignore
from datasets import load_dataset # type: ignore
from safetensors.torch import save_file
from tqdm import tqdm
import os, json, torch

from common import split_list_with_overlap

OUT_ROOT = "/raid/datasets/fineweb/sentence-embedding"
if not os.path.exists(OUT_ROOT):
   os.makedirs(OUT_ROOT)

BLOCK_SIZE     = 512
TARGET_OVERLAP = 0

BLOCKS_PER_BATCH     =  4 * 1024
MAX_ENTRIES_PER_FILE = 64 * 1024

def make_filename(i:int) -> str:
   return  f"blob_{i:04d}.st"

def main():
   model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device="cuda:2")
   tokenizer = SentencePieceProcessor(model_file="/raid/downloads/LLaMA-2/7B/tokenizer.model")
   dataset = load_dataset(path="/raid/datasets/fineweb/slim")
   split_dataset = dataset["train"].train_test_split(test_size=0.0001, seed=1337, shuffle=True)

   info_file = os.path.join(OUT_ROOT, "index.json")
   if os.path.exists(info_file):
      with open(info_file) as f:
         info = json.load(f)
      assert info["size"] == MAX_ENTRIES_PER_FILE, f'size {info["size"]} != {MAX_ENTRIES_PER_FILE} in cached files'
   else:
      info = { "i":-1, "size":MAX_ENTRIES_PER_FILE, "blobs":[] }

   toks = []
   blks = []
   embs = [] # type: ignore

   i = -1
   for row in tqdm(split_dataset["train"], disable=False):
      i += 1
      # if i >= 10:
      #    break
      if info["i"] >= i:
         continue

      text: str = row["text"]
      tokens = tokenizer.Encode(text)

      chunks = split_list_with_overlap(tokens, block_size=BLOCK_SIZE, target_overlap=TARGET_OVERLAP)
      if chunks is None:
         continue

      # print(f"Row {i}, text length: {len(text)}, token count: {len(tokens)}, k: {len(chunks)}")

      for chunk in chunks:
         toks.append(chunk)
         text = tokenizer.Decode(chunk)
         assert text is not None
         blks.append(text)
   
         if len(blks) >= BLOCKS_PER_BATCH:
            embs += model.encode(blks, convert_to_tensor=True)
            blks = []

            if len(embs) >= MAX_ENTRIES_PER_FILE:
               filename = make_filename(len(info['blobs']))
               info["i"] = i
               info["blobs"].append(filename)
               data = {
                  "tok": torch.Tensor(toks).int(),
                  "emb": torch.stack(embs).half(),
               }
               save_file(data, os.path.join(OUT_ROOT, filename))
               with open(info_file, "w") as f:
                  json.dump(info, f)
               print(f"Data contains tok[{data['tok'].shape}] emb[{data['emb'].shape}] rows")
               print(f"Saved embeddings to {filename}")

               toks = []
               embs = []


if __name__ == '__main__':
   main()
