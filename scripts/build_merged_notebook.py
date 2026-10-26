"""Builds colab/faculty_q7_merged_gcb.ipynb from the tested scripts."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "colab", "faculty_q7_merged_gcb.ipynb")
SCRIPTS = ["devmap.py", "train_lib.py", "14_merged_finetune.py"]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def writefile(name):
    with open(os.path.join(HERE, name)) as fh:
        return code(f"%%writefile {name}\n" + fh.read())


cells = [
    md("""
# Faculty point 7: GraphCodeBERT on AMD + NVIDIA merged into one dataset

**Question:** if we merge the two machines into one dataset (1,360 rows), does one GraphCodeBERT model do better than two separate ones?

**How:** the model is told which GPU each row is for (the GPU name goes at the start of its input). Folds are dealt by application, so both copies of a kernel are always on the same side. On exactly these folds we train:
- **merged:** one GraphCodeBERT for both machines
- **separate:** one GraphCodeBERT per machine (the usual way), for a fair comparison

The Random Forest version, on the same folds, gave: separate 80.9 / 67.4, merged 80.2 / 66.7 (AMD / NVIDIA).

**How to run:**
1. **Runtime → Change runtime type → T4 GPU**, save. (Your Gemini notebook can keep running on CPU in another tab.)
2. **Runtime → Run all.** Allow Google Drive (tick **Select all**).
3. Seed 1 comes first: merged (about 30 min), then separate (about 30 min). Seeds 2 and 3 are repeats (another 2 hours) and can run another day.
4. **If you must stop** (or Colab disconnects): every finished pile is saved in `devmap_results` on your Drive. Run all again later and it carries on.
5. Send Claude a screenshot of the last cell.
"""),
    code("""
import os, shutil, subprocess, torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE - use Runtime > Change runtime type > T4 GPU")
if not os.path.exists("GM_SOURCE_CODE"):
    subprocess.run(["git", "clone", "--depth", "1", "-q", "https://github.com/marco-siino/GM_SOURCE_CODE.git"], check=True)
os.makedirs("data", exist_ok=True)
shutil.copy("GM_SOURCE_CODE/AMD_DS/dataset-devmap-amd.csv", "data")
shutil.copy("GM_SOURCE_CODE/NVIDIA_DS/dataset-devmap-nvidia.csv", "data")
print("Ready.")
"""),
    md("## Helper code (no need to read or change)"),
    *[writefile(n) for n in SCRIPTS],
    md("## Settings"),
    code("""
import os
SEEDS = [1, 2, 3]          # seed 1 first for both ways; 2 and 3 are repeats

RESULTS_DIR = "results"
try:
    from google.colab import drive
    drive.mount("/content/drive")                      # tick "Select all"
    RESULTS_DIR = "/content/drive/MyDrive/devmap_results"
except ImportError:
    pass
os.makedirs(RESULTS_DIR, exist_ok=True)
print("Saving to:", RESULTS_DIR)
"""),
    md("## Train and sit the exam (resumable)"),
    code("""
import importlib
M = importlib.import_module("14_merged_finetune")
M.run(seeds=SEEDS, data_dir="data", out_dir=RESULTS_DIR)
print("\\nAll requested runs are finished.")
"""),
    md("## Results so far (send Claude a screenshot)"),
    code("""
import importlib, pandas as pd
M = importlib.import_module("14_merged_finetune")
pd.set_option("display.width", 200)
print(M.summary(RESULTS_DIR).to_string(index=False))
print("\\nRandom Forest on the same folds: separate 80.9 / 67.4, merged 80.2 / 66.7 (AMD / NVIDIA)")
"""),
    md("## Download the results to your laptop too (a backup in case Drive fails)"),
    code("""
import glob, os, shutil
from google.colab import files
os.makedirs("backup", exist_ok=True)
for f in glob.glob(os.path.join(RESULTS_DIR, "preds__graphcodebert+*__both__*")):
    shutil.copy(f, "backup")
shutil.make_archive("results_backup", "zip", "backup")
print("Files in backup:", sorted(os.listdir("backup")))
files.download("results_backup.zip")
"""),
]

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as fh:
    json.dump(nb, fh, indent=1)
print("wrote", os.path.normpath(OUT), "with", len(cells), "cells")
