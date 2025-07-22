import json
import os
from pathlib import Path

FILES_ROOT = Path(os.path.dirname(__file__))

RAW_DATA_DIRNAME = "raw"

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
