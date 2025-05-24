from sentence_transformers import SentenceTransformer
from datasets import load_dataset # type: ignore
from tqdm import tqdm
import os, json

MAX_BLOCK_SIZE = 1024
BLOCK_OVERLAP  = 64
SHIFT_AMOUNT   = MAX_BLOCK_SIZE - BLOCK_OVERLAP

BLOCKS_PER_BATCH     =  4 * 1024
MAX_ENTRIES_PER_FILE = 64 * 1024

OUT_ROOT = "/raid/datasets/wikipedia/sentence-embedding"
if not os.path.exists(OUT_ROOT):
   os.makedirs(OUT_ROOT)

SPLITTERS = ["\n\nSee also\n\n", "\n\nReferences\n\n", "\n\nCitations\n\n", "\n\nNotes and references\n\n", "\n\nFurther reading\n\n", "\n\nExternal links\n\n", "\n\nNotes\n\n"]
def clean_text(text:str) -> str|None:
   for splitter in SPLITTERS:
      if splitter in text:
         return text.split(splitter, 1)[0]
   return None

def make_filename(i:int) -> str:
   return  f"blob_{i:04d}.st"

def load_sentence_model() -> SentenceTransformer:
   return SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def create():
   from safetensors.torch import save_file
   import torch

   model = load_sentence_model()

   ds = load_dataset("wikimedia/wikipedia", "20231101.en")
   train_ds = ds["train"]

   info_file = os.path.join(OUT_ROOT, "index.json")
   if os.path.exists(info_file):
      with open(info_file) as f:
         info = json.load(f)
      assert info["size"] == MAX_ENTRIES_PER_FILE, f'size {info["size"]} != {MAX_ENTRIES_PER_FILE} in cached files'
   else:
      info = { "i":0, "size":MAX_ENTRIES_PER_FILE, "blobs":[] }

   # print(train_ds.select([20])["title"])
   # return

   document_ids = []
   text_start = []
   blocks = []
   embeddings = [] # type: ignore

   i = -1
   for row in tqdm(train_ds):
      i += 1
      if info["i"] >= i:
         continue
      # if i >= 30:
      #    break

      # title: str = row["title"]
      orig = row["text"]
      text = clean_text(orig)
      if text is None:
         # print(f"WARNING: entry '{title}' did not contain splitter, skipping")
         continue
      # print(f"Row {i}: '{title}', text length: {len(orig)} -> {len(text)}")

      ptr = 0
      while True:
         document_ids.append(i)
         text_start.append(ptr)
         blocks.append(text[ptr:(ptr+MAX_BLOCK_SIZE)])

         if len(blocks) >= BLOCKS_PER_BATCH:
            # print("Encoding block")
            embeddings += model.encode(blocks, convert_to_tensor=True)
            blocks = []

            if len(embeddings) >= MAX_ENTRIES_PER_FILE:
               filename = make_filename(len(info['blobs']))
               info["i"] = i
               info["blobs"].append(filename)
               data = {
                  "document_id": torch.Tensor(document_ids).int(),
                  "text_start": torch.Tensor(text_start).int(),
                  "embeddings": torch.stack(embeddings),
               }
               save_file(data, os.path.join(OUT_ROOT, filename))
               with open(info_file, "w") as f:
                  json.dump(info, f)
               print(f"Data contains [{data['document_id'].shape}] [{data['text_start'].shape}] [{data['embeddings'].shape}] rows")
               print(f"Saved embeddings to {filename}")

               document_ids = []
               text_start = []
               embeddings = []

         ptr += SHIFT_AMOUNT
         if ptr >= len(text):
            break


def use():
   from tinygrad import Tensor, nn, dtypes, Device
   Device.DEFAULT = "CUDA"
   GPUS = tuple(f"{Device.DEFAULT}:{i}" for i in range(1,5))

   ds = load_dataset("wikimedia/wikipedia", "20231101.en")
   train_ds = ds["train"]

   def to_dev(x:Tensor, axis=None) -> Tensor:
      return x

   model = load_sentence_model()
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
         if k == "embeddings":
            new_value = new_value.cast(dtypes.float16).reshape(-1, len(GPUS), new_value.shape[-1]).shard(GPUS, axis=1).realize()
         data_list[k].append(new_value)
   data_cat = {}
   for k, v in data_list.items():
      data_cat[k] = Tensor.cat(*v).realize()
      print(f"{k}: {data_cat[k].shape}")

   ref = data_cat["embeddings" ]
   ids = data_cat["document_id"]
   ptr = data_cat["text_start" ]
   print(f"{ref.shape=}")

   def norm(x:Tensor, dim:int=-1, eps:float=1e-8) -> Tensor:
      return ((x * x).sum(axis=dim, keepdim=True) + eps).sqrt()

   def cosine_similarity(a:Tensor, b:Tensor, dim:int=-1, eps:float=1e-8) -> Tensor:
      a_norm = a / (norm(a, dim=dim) + eps)
      b_norm = b / (norm(b, dim=dim) + eps)
      return Tensor.mul(a_norm, b_norm).sum(axis=dim)

   text = "This is the tale, of Captain Jack Sparrow!"
   emb = Tensor(model.encode(text), dtype=dtypes.float16).rearrange('d -> 1 d').shard(GPUS).realize()
   print(f"{emb.shape=}")

   sim = cosine_similarity(emb, ref).to(Device.DEFAULT).flatten().realize()
   print(f"{sim.shape=}")  

   TOP_K = 10
   top_val, top_ind = [v.numpy() for v in sim.topk(TOP_K)]
   for i in range(TOP_K):
      amax = int(top_ind[i])
      aids = int(ids[amax].item())
      aptr = int(ptr[amax].item())
      for row in train_ds.select([aids]):
         block = row['text'][aptr:aptr+MAX_BLOCK_SIZE]
         print(f"\nRank: {i+1}\nSim:  {top_val[i]:.4f}\nName: {row['title']}\n" + ">"*40 + f"\n{block}\n" + "<"*40 + "\n")

if __name__ == "__main__":
   use()
