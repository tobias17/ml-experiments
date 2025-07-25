from tinygrad import Tensor, dtypes, nn, Device, TinyJit, Context
from tqdm import tqdm

from common import (
   make_filename, cosine_similarity, dataset_root,
   WIKI_EMBED_DIRNAME, DATA_EMBED_DIRNAME, SPLICED_DIRNAME, ENTRIES_PER_FILE,
)

LOAD_SIZE:   int = 64
LOAD_SLICES: int = ENTRIES_PER_FILE // LOAD_SIZE
assert LOAD_SLICES * LOAD_SIZE == ENTRIES_PER_FILE

GPUS = tuple(f"{Device.DEFAULT}:{i}" for i in range(6))
EMB_DTYPE = dtypes.float16

def main():
   wiki_root = dataset_root() / WIKI_EMBED_DIRNAME
   data_root = dataset_root() / DATA_EMBED_DIRNAME
   assert wiki_root.exists() and data_root.exists()

   out_root = dataset_root() / f"{SPLICED_DIRNAME}-best1"
   if not out_root.exists():
      out_root.mkdir()

   ######################
   #   Load Wiki Data   #
   ######################
   wiki_list = {} # type: ignore
   tok_disktensors = []
   for i in range(1024):
      filepath = wiki_root / make_filename(i)
      if not filepath.exists():
         break
      sd = nn.state.safe_load(str(filepath))
      for k, v in sd.items():
         if k == "tok":
            tok_disktensors.append(v)
         if k != "emb":
            continue
         if k not in wiki_list:
            wiki_list[k] = []
         new_value = v.reshape(-1, len(GPUS), v.shape[-1]).shard(GPUS, axis=1).cast(EMB_DTYPE).realize()
         wiki_list[k].append(new_value)
   wiki = {}
   for k, v in wiki_list.items():
      wiki[k] = Tensor.cat(*v).realize() # type: ignore
      print(f"{k}: {wiki[k].shape}")

   wiki_emb: Tensor = wiki["emb"]

   @TinyJit
   def _get_closest_wiki_entry_index(emb:Tensor) -> Tensor:
      sim = cosine_similarity(emb.cast(EMB_DTYPE).unsqueeze(1).unsqueeze(1).shard(GPUS), wiki_emb.unsqueeze(0)).to(Device.DEFAULT).realize()
      sim = sim.to_(Device.DEFAULT).realize()
      return sim.reshape(emb.shape[0], -1).argmax(axis=-1).realize()

   def get_closest_wiki_entry(emb:Tensor) -> Tensor:
      with Context(BEAM=1):
         index = _get_closest_wiki_entry_index(emb.realize()).numpy()
      index = index
      toks = []
      for i in index:
         i = int(i)
         toks.append(tok_disktensors[i // ENTRIES_PER_FILE][i % ENTRIES_PER_FILE].to(Device.DEFAULT))
      return Tensor.stack(*toks).realize()

   #####################
   #   Load Fine Web   #
   #####################
   for i in range(1024):
      filename = make_filename(i)
      print(filename)
      output_filepath = out_root / filename
      if output_filepath.exists():
         continue
      input_filepath = data_root / filename
      if not input_filepath.exists():
         break
      sd = nn.state.safe_load(str(input_filepath))
      fine_web_emb = sd["emb"].to(Device.DEFAULT).realize()

      wiki_tokens = []
      wiki_blocks = []
      for k in tqdm(range(LOAD_SLICES), disable=False):
         # print(f"{fine_web_emb.shape=}")
         # print(f"{k*LOAD_SIZE=}")
         wiki_blocks.append(get_closest_wiki_entry(fine_web_emb[k*LOAD_SIZE:(k+1)*LOAD_SIZE].contiguous().realize())) # type: ignore
         if len(wiki_blocks) >= 64:
            wiki_tokens.append(Tensor.cat(*wiki_blocks).realize())
            wiki_blocks = []
      if len(wiki_blocks) > 0:
         wiki_tokens.append(Tensor.cat(*wiki_blocks).realize())
         wiki_blocks = []
      data = {
         "wiki": Tensor.cat(*wiki_tokens).realize(),
         "tok":  sd["tok"],
      }
      nn.state.safe_save(data, str(output_filepath))
      print(f"saved data wiki[{data['wiki'].shape}] tok[{data['tok'].shape}]")


if __name__ == "__main__":
   main()
