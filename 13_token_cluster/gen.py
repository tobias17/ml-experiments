from tinygrad import Tensor # type: ignore
from tinygrad.nn.state import load_state_dict, safe_load # type: ignore

import os, sys

def load_model(index:int, root_dir:str, folder_name:str, weight_name:str, config_pat:str):
   repo_root, experiment = os.path.split(os.path.dirname(os.path.realpath(__file__)))

   weights_dir = os.path.join(repo_root, root_dir, experiment)
   assert os.path.isdir(weights_dir), f"expected to find a weights root dir at {weights_dir}"

   if folder_name.lower() == "latest":
      existing_folders = os.listdir(weights_dir)
      assert len(existing_folders) > 0, f"found 0 run folders in weights root dir {weights_dir}"
      folder_name = sorted(existing_folders)[-1]
   
   run_dir = os.path.join(weights_dir, folder_name)
   assert os.path.isdir(run_dir), f"expected to find a run root dir at {run_dir}"

   if weight_name.lower() == "latest":
      config_pat = config_pat.format(index)
      weight_files = [f for f in os.listdir(run_dir) if os.path.isfile(os.path.join(run_dir, f)) and config_pat in f]
      assert len(weight_files) > 0, f"found 0 weight files in run root dir {run_dir}"
      weight_name = sorted(weight_files)[-1]

   weight_path = os.path.join(run_dir, weight_name)
   assert os.path.isfile(weight_path), f"expected to find a weights file at {weight_path}"
   print(f"Loading weights from {weight_path}")

   sys.path.append(run_dir)
   from modelfile import create_models
   model = create_models()[0]

   load_state_dict(model, safe_load(weight_path))

   return model

text_init = """SEBASTIAN:
Bate, I beseech you, widow Dido.

ANTONIO:
"""

if __name__ == "__main__":
   import argparse
   parser = argparse.ArgumentParser()
   parser.add_argument('index', type=int)
   parser.add_argument('-r', '--root-dir',    type=str, default='archive')
   parser.add_argument('-f', '--folder-name', type=str, default='latest')
   parser.add_argument('-w', '--weight-name', type=str, default='latest')
   parser.add_argument('-c', '--config-pat',  type=str, default='_c{0}.st')
   parser.add_argument('-g', '--gen-amount',  type=int, default=32)
   args = parser.parse_args()

   model = load_model(args.index, args.root_dir, args.folder_name, args.weight_name, args.config_pat)
   
   text_out = model.generate(text_init, args.gen_amount)
   print(text_out)
