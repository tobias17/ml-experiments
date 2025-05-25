

BLOCK_SIZE     = 512
TARGET_OVERLAP = 0

BLOCKS_PER_BATCH =  4 * 1024
ENTRIES_PER_FILE = 64 * 1024


from tinygrad import Tensor
from typing import List
import math

from sentencepiece import SentencePieceProcessor # type: ignore
def load_tokenizer() -> SentencePieceProcessor:
   return SentencePieceProcessor(model_file="/raid/downloads/LLaMA-2/7B/tokenizer.model")

from sentence_transformers import SentenceTransformer
def load_sentence_model(i:int=0) -> SentenceTransformer:
   return SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=f"cuda:{i}")

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

if __name__ == "__main__":
   res = split_list_with_overlap(list(range(1,1001)), block_size=64, target_overlap=8)
   assert res is not None
   for r in res:
      print(r, "\n")
