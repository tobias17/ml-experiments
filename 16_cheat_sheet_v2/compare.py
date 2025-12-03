import matplotlib.pyplot as plt
import json, os, argparse
from pathlib import Path
import numpy as np

from train import AVERAGE_EVERY, GLOBAL_BS, BLOCK_SIZE

def main(cheat_sheet_json_path:Path, baseline_json_path:Path):
   assert cheat_sheet_json_path.exists() and baseline_json_path.exists()
   with open(cheat_sheet_json_path) as f: cheat_sheet_data: dict = json.load(f)
   with open(baseline_json_path)    as f: baseline_data:    dict = json.load(f)

   for name in ["loss", "acc"]:
      key = f"train_{name}"
      assert key in cheat_sheet_data, f"Required key {key} not in cheat sheet json, options were {list(cheat_sheet_data.keys())}"
      assert key in baseline_data,    f"Required key {key} not in baseline json, options were {list(baseline_data.keys())}"

      size = min(len(cheat_sheet_data[key]), len(baseline_data[key]))
      print(f"Key {key} using size {size} (cheat_sheet={len(cheat_sheet_data[key])}, baseline={len(baseline_data[key])})")
      
      ys = np.array(cheat_sheet_data[key][:size], dtype=np.float32) - np.array(baseline_data[key][:size], dtype=np.float32)
      xs = (np.arange(len(ys))+1)*AVERAGE_EVERY*GLOBAL_BS*BLOCK_SIZE / 1_000_000_000.0
      plt.clf()
      ymax = 1.2 * np.percentile(ys, 95)
      ymin = 1.2 * np.percentile(ys, 5)
      diff = ymax - ymin
      ymax += diff * 0.2
      ymin -= diff * 0.2
      plt.plot(xs, ys)
      plt.plot([xs[0], xs[-1]], [0.0, 0.0])
      plt.ylim((ymin, ymax))
      plt.xlim((0, None))
      plt.xlabel("Tokens (Billions)")
      plt.ylabel(f"{name} (cheat_sheet - baseline)")
      plt.title(f"Train {name[0].upper()}{name[1:]}")
      figure = plt.gcf()
      figure.set_size_inches(18, 10)
      save_path = os.path.join(os.path.dirname(__file__), f"diff_{name}.png")
      plt.savefig(save_path, dpi=100)
      print(f"Saving to {save_path}...")

      BUCKETS = 128
      amnt_per_bucket = size // BUCKETS
      size = BUCKETS * amnt_per_bucket
      plt.clf()
      for label, data in {"cheat_sheet": cheat_sheet_data, "baseline": baseline_data}.items():
         ys = np.array(data[key][:size], dtype=np.float32).reshape(BUCKETS, amnt_per_bucket).mean(axis=1, keepdims=False)
         xs = (np.arange(len(ys))+1)*AVERAGE_EVERY*GLOBAL_BS*BLOCK_SIZE*amnt_per_bucket / 1_000_000_000.0
         plt.plot(xs, ys, label=label)
      plt.xlim((0, None))
      plt.xlabel("Tokens (Billions)")
      plt.ylabel(name)
      plt.title(f"Train {name[0].upper()}{name[1:]}")
      plt.legend()
      figure = plt.gcf()
      figure.set_size_inches(18, 10)
      save_path = os.path.join(os.path.dirname(__file__), f"joined_{name}.png")
      plt.savefig(save_path, dpi=100)
      print(f"Saving to {save_path}...")


if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument("--cheat-sheet", type=Path, required=True)
   parser.add_argument("--baseline",    type=Path, required=True)
   args = parser.parse_args()

   main(args.cheat_sheet, args.baseline)
