from typing import List
from pathlib import Path
import json, os, math

BLOCK_SIZE    = 512
DATASET_BLOCK = BLOCK_SIZE + 1

BLOCKS_PER_BATCH =    4*1024
ENTRIES_PER_FILE = 6*32*1024


FILES_ROOT = Path(os.path.dirname(__file__))

RAW_DATA_DIRNAME = "raw"
WIKI_EMBED_DIRNAME = "wiki-embed"

def dataset_root() -> Path:
   env_json = FILES_ROOT / "env.json"
   assert env_json.exists(), f"Failed to find env json file, searched for {env_json}"
   with open(env_json) as f:
      data = json.load(f)
   root = data.get(k := "dataset_root")
   assert root is not None, f"Could not find '{k}' entry in env json, make sure it is populated, only found {list(data.keys())}"
   root_path = Path(root)
   if not root_path.exists():
      root_path.mkdir()
   return root_path

def load_wiki_dataset():
   from datasets import load_dataset # type: ignore
   return load_dataset("wikimedia/wikipedia", "20231101.en")


from sentencepiece import SentencePieceProcessor # type: ignore
def load_tokenizer() -> SentencePieceProcessor:
   return SentencePieceProcessor(model_file="/raid/downloads/LLaMA-2/7B/tokenizer.model")

from sentence_transformers import SentenceTransformer
def load_sentence_model(i:int=0) -> SentenceTransformer:
   return SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=f"cuda:{i}")


from tinygrad import Tensor
def norm(x:Tensor, dim:int=-1, eps:float=1e-8) -> Tensor:
   return ((x * x).sum(axis=dim, keepdim=True) + eps).sqrt()
def cosine_similarity(a:Tensor, b:Tensor, dim:int=-1, eps:float=1e-8) -> Tensor:
   a_norm = a / (norm(a, dim=dim) + eps)
   b_norm = b / (norm(b, dim=dim) + eps)
   return Tensor.mul(a_norm, b_norm).sum(axis=dim)


def make_filename(i:int) -> str:
   return  f"blob_{i:04d}.st"

def split_list_with_overlap(input_list:List, block_size:int, target_overlap:int) -> List|None:
   if block_size <= 0:
      raise ValueError("block_size must be positive")
   if block_size > len(input_list):
      return None

   n = len(input_list)

   if n < block_size:
      return None
   if n == block_size:
      return [input_list]
   if n < 2*block_size - target_overlap:
      return [input_list[:block_size], input_list[-block_size:]]

   k_approx = (n - block_size) / (block_size - target_overlap) + 1
   k = max(1, round(k_approx))  # Round to nearest integer

   result = []
   delta = (n - block_size) / k
   for i in range(k):
      s = math.floor(i*delta) if i < k-1 else n-block_size
      result.append(input_list[s:s+block_size])

   return result
