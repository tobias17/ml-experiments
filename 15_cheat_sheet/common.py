from typing import List, Tuple
import math

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
