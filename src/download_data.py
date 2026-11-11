"""
Step 0: download KazSAnDRA (Yeshpanov & Varol, LREC-COLING 2024, CC BY 4.0).

The authors publish the data openly on GitHub (no login needed):
  https://github.com/IS2AI/KazSAnDRA/tree/main/dataset
We use the "score classification" files (they keep the original 1-5 star rating):
  06_sc_train_ib.zip, 09_sc_valid.zip, 10_sc_test.zip  (~9 MB together)
Together they contain all 175,158 KazSAnDRA reviews.  The authors' own
train/valid/test split is NOT used: we make our own split in prepare_data.py.

Usage:  python src/download_data.py
"""
import os
import urllib.request
import zipfile

from text_utils import load_config

BASE_URL = "https://raw.githubusercontent.com/IS2AI/KazSAnDRA/main/dataset/"
ZIPS = ["06_sc_train_ib.zip", "09_sc_valid.zip", "10_sc_test.zip"]


def main():
    cfg = load_config()
    out_dir = cfg["data"]["raw_dir"]
    os.makedirs(out_dir, exist_ok=True)
    for name in ZIPS:
        zip_path = os.path.join(out_dir, name)
        if not os.path.exists(zip_path):
            print("downloading", BASE_URL + name)
            urllib.request.urlretrieve(BASE_URL + name, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(out_dir)
    print("done ->", out_dir, sorted(os.listdir(out_dir)))


if __name__ == "__main__":
    main()
