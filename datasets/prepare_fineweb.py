from sentencepiece import SentencePieceProcessor # type: ignore
from datasets import load_dataset # type: ignore
from tqdm import tqdm # type: ignore
import numpy as np
import os

OUT_ROOT = "/raid/datasets/fineweb/tokenized"
if not os.path.exists(OUT_ROOT):
   os.makedirs(OUT_ROOT)
num_proc = 24

if __name__ == '__main__':
   tokenizer = SentencePieceProcessor(model_file="/raid/downloads/LLaMA-2/7B/tokenizer.model")

   dataset = load_dataset(path="/raid/datasets/fineweb/data")
   split_dataset = dataset["train"].train_test_split(test_size=0.0001, seed=1337, shuffle=True)
   split_dataset['val'] = split_dataset.pop('test') # rename the test split to val

   def process(example):
      ids = tokenizer.Encode(example['text'])
      ids.append(tokenizer.eos_id())
      out = {'ids': ids, 'len': len(ids)}
      return out
   
   tokenized = split_dataset.map(
      process,
      remove_columns=['text'],
      desc="tokenizing the splits",
      num_proc=num_proc,
   )

   for split, dset in tokenized.items():
      arr_len = np.sum(dset['len'], dtype=np.uint64)
      filename = os.path.join(OUT_ROOT, f'fineweb_{split}.bin')
      dtype = np.uint16
      arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
      total_batches = 256

      idx = 0
      for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
         # Batch together samples for faster write
         batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
         arr_batch = np.concatenate(batch['ids'])
         # Write into mmap
         arr[idx : idx + len(arr_batch)] = arr_batch
         idx += len(arr_batch)
      arr.flush()
