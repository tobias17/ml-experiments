from tinygrad import Tensor, dtypes, nn, Device, TinyJit, Context
from tqdm import tqdm
import os, json

from common import (
   make_filename, cosine_similarity, load_tokenizer, load_sentence_model,
   ENTRIES_PER_FILE,
)

LOAD_SIZE:   int = 64
LOAD_SLICES: int = ENTRIES_PER_FILE // LOAD_SIZE
assert LOAD_SLICES * LOAD_SIZE == ENTRIES_PER_FILE

WIKI_ROOT     = "/raid/datasets/wikipedia/sentence-embedding"
FINE_WEB_ROOT = "/raid/datasets/fineweb/sentence-embedding"
OUT_ROOT      = "/raid/datasets/fineweb/with-wiki-top1"
if not os.path.exists(OUT_ROOT):
   os.makedirs(OUT_ROOT)

GPUS = tuple(f"{Device.DEFAULT}:{i}" for i in range(1,5))
EMB_DTYPE = dtypes.float16

def main():
   ######################
   #   Load Wiki Data   #
   ######################
   wiki_list = {} # type: ignore
   tok_disktensors = []
   for i in range(1024):
      filepath = os.path.join(WIKI_ROOT, make_filename(i))
      if not os.path.exists(filepath):
         break
      sd = nn.state.safe_load(filepath)
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

   RUN_EVAL_SET = True

   if RUN_EVAL_SET:
      #####################
      #   Load Eval Set   #
      #####################
      model = load_sentence_model(1)
      tokenizer = load_tokenizer()
      filepath = os.path.join(os.path.dirname(__file__), "eval_questions.txt")
      with open(filepath) as f:
         eval_questions = f.read().split("\n")
         eval_questions = [q.strip() for q in eval_questions if q]

      text_data = []
      tok_data  = []
      wiki_data = []
      size_data = []

      for q in eval_questions:
         emb = Tensor(model.encode(q)).reshape(1, 1, 1, -1).contiguous().realize()
         entry = get_closest_wiki_entry(emb)
         text_data.append({
            "question": q,
            "wiki": tokenizer.Decode(entry.tolist()[0]), # type: ignore
         })

         full_q = f"Q: {q}\nA:"
         tokens = Tensor(tokenizer.Encode(full_q)).flatten()
         size   = tokens.shape[0]
         tok_data .append(tokens.pad((0,512-size)).realize())
         wiki_data.append(entry.flatten().realize())
         size_data.append(Tensor([size]))

      data = {
         "tok":  Tensor.stack(*tok_data ).realize(),
         "wiki": Tensor.stack(*wiki_data).realize(),
         "size": Tensor.stack(*size_data).realize(),
      }
      print()
      for k,v in data.items(): print(f"{k}: {v.shape}")
      print()
      nn.state.safe_save(data, os.path.join(OUT_ROOT, "eval.st"))
      with open(os.path.join(OUT_ROOT, "eval.json"), "w") as f:
         json.dump(text_data, f, indent="\t")

   else:
      #####################
      #   Load Fine Web   #
      #####################
      for i in range(1024):
         filename = make_filename(i)
         print(filename)
         output_filepath = os.path.join(OUT_ROOT, filename)
         if os.path.exists(output_filepath):
            continue
         input_filepath = os.path.join(FINE_WEB_ROOT, filename)
         if not os.path.exists(input_filepath):
            break
         sd = nn.state.safe_load(input_filepath)
         fine_web_emb = sd["emb"].to(Device.DEFAULT).realize()

         wiki_tokens = []
         wiki_blocks = []
         for k in tqdm(range(LOAD_SLICES), disable=False):
            wiki_blocks.append(get_closest_wiki_entry(fine_web_emb[k*LOAD_SIZE:(k+1)*LOAD_SIZE].contiguous().realize()))
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
         nn.state.safe_save(data, output_filepath)
         print(f"saved data wiki[{data['wiki'].shape}] tok[{data['tok'].shape}]")


if __name__ == "__main__":
   main()
