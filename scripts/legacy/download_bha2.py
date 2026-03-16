#!/usr/bin/env python3
"""
Download BHA2 (Brain Hierarchical Atlas 2) data.zip from Zenodo 8158914 and unzip to data/bha2/.
Requires ~12.2 GB disk and network. Run from project root or set BHA2_DATA_DIR.
"""
import os
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve, urlopen

ZENODO_RECORD = "8158914"
ZENODO_FILE = "data.zip"
DOWNLOAD_URL = f"https://zenodo.org/records/{ZENODO_RECORD}/files/{ZENODO_FILE}?download=1"

def main():
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data" / "bha2"
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / ZENODO_FILE

    if zip_path.exists():
        print(f"Found existing {zip_path}; skipping download. Delete to re-download.")
    else:
        print(f"Downloading BHA2 {ZENODO_FILE} (~12.2 GB) to {zip_path} ...")
        try:
            urlretrieve(DOWNLOAD_URL, zip_path)
        except Exception as e:
            print(f"Download failed: {e}", file=sys.stderr)
            print("Manual: get", DOWNLOAD_URL, "and save as", zip_path, file=sys.stderr)
            sys.exit(1)
        print("Download done.")

    extract_to = data_dir
    if (extract_to / "data").exists():
        print("data/ already extracted; skipping unzip.")
    else:
        print("Unzipping (this may take a while) ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_to)
        print("Unzip done.")

    print("BHA2 data ready under", data_dir)

if __name__ == "__main__":
    main()
