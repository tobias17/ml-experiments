from safetensors.torch import save_file
from datasets import load_dataset # type: ignore
from tqdm import tqdm
import torch, json

from common import (
   split_list_with_overlap, load_tokenizer, load_sentence_model, make_filename,
   dataset_root, RAW_DATA_DIRNAME, DATA_EMBED_DIRNAME, BLOCKS_PER_BATCH, ENTRIES_PER_FILE, DATASET_BLOCK,
)


TARGET_OVERLAP = 0
DEVICE_ID = 1


def main():
   model = load_sentence_model(DEVICE_ID)
   tokenizer = load_tokenizer()
   in_root = dataset_root() / RAW_DATA_DIRNAME
   dataset = load_dataset(str(in_root))
   split_dataset = dataset["train"].train_test_split(test_size=0.0001, seed=1337, shuffle=True)

   out_root = dataset_root() / DATA_EMBED_DIRNAME
   if not out_root.exists():
      out_root.mkdir()
   info_file = out_root / "index.json"
   if info_file.exists():
      with open(info_file) as f:
         info = json.load(f)
      assert info["size"] == DATASET_BLOCK, f'size {info["size"]} != {ENTRIES_PER_FILE} in cached files'
      assert info["amount"] == ENTRIES_PER_FILE, f'amount {info["amount"]} != {ENTRIES_PER_FILE} in cached files'
   else:
      info = { "i":-1, "size":DATASET_BLOCK, "amount":ENTRIES_PER_FILE, "blobs":[] }

   toks = []
   blks = []
   embs = [] # type: ignore

   i = -1
   for row in tqdm(split_dataset["train"], disable=False):
      i += 1
      if info["i"] >= i:
         continue

      text: str = row["text"]
      tokens = tokenizer.Encode(text)

      chunks = split_list_with_overlap(tokens, block_size=DATASET_BLOCK, target_overlap=TARGET_OVERLAP)
      if chunks is None:
         continue

      for chunk in chunks:
         toks.append(chunk)
         text = tokenizer.Decode(chunk)
         assert text is not None
         blks.append(text)
   
         if len(blks) >= BLOCKS_PER_BATCH:
            embs += model.encode(blks, convert_to_tensor=True)
            blks = []

            if len(embs) >= ENTRIES_PER_FILE:
               filename = make_filename(len(info['blobs']))
               info["i"] = i
               info["blobs"].append(filename)
               data = {
                  "tok": torch.Tensor(toks).int(),
                  "emb": torch.stack(embs).half(),
               }
               save_file(data, str(out_root / filename))
               with open(info_file, "w") as f:
                  json.dump(info, f)
               print(f"Data contains tok[{data['tok'].shape}] emb[{data['emb'].shape}] rows")
               print(f"Saved embeddings to {filename}")

               toks = []
               embs = []

if __name__ == '__main__':
   main()
