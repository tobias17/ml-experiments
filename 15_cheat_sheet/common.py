from typing import List, Tuple
import math

def split_list_with_overlap(input_list:List, block_size:int, target_overlap:int) -> Tuple[List|None,List]:
   if block_size <= 0:
      raise ValueError("block_size must be positive")
   if block_size > len(input_list):
      return None, []

   n = len(input_list)

   if n < block_size:
      return None, []
   if n == block_size:
      return [input_list], [0]
   if n < 2*block_size - target_overlap:
      return [input_list[:block_size], input_list[-block_size:]], [0,n-block_size]

   # Compute effective overlap
   # Approximate number of sublists: k = (n - block_size) / (block_size - overlap) + 1
   # Adjust overlap to make k close to an integer
   k_approx = (n - block_size) / (block_size - target_overlap) + 1
   k = max(1, round(k_approx))  # Round to nearest integer

   result = []
   starts = []

   delta = (n - block_size) / k
   for i in range(k):
      s = math.floor(i*delta) if i < k-1 else n-block_size
      result.append(input_list[s:s+block_size])
      starts.append(s)
   
   return result, starts

if __name__ == "__main__":
   res, starts = split_list_with_overlap(list(range(1,1001)), block_size=64, target_overlap=8)
   assert res is not None
   for r, s in zip(res, starts):
      print(s, r, "\n")
