import subprocess

from common import dataset_root, load_wiki_dataset, RAW_DATA_DIRNAME

BASE_URL = "https://huggingface.co/datasets/HuggingFaceFW/fineweb/resolve/main/data/CC-MAIN-2025-26"
AMOUNT_TO_GRAB = 12

def main():
   print("Downloading fineweb subset")
   root = dataset_root() / RAW_DATA_DIRNAME
   if not root.exists():
      root.mkdir()
   for i in range(AMOUNT_TO_GRAB):
      filename = f"000_{i:05d}.parquet"
      outpath = root / filename
      if outpath.exists():
         print(f"{filename} Already found, skipping")
      else:
         subprocess.run(f"wget {BASE_URL}/{filename} --directory-prefix={root} -q --show-progress", shell=True, check=True)
   
   print("\nDownloading en wiki")
   load_wiki_dataset()

if __name__ == "__main__":
   main()
