"""Builds colab/faculty_q9_end_to_end.ipynb from the tested scripts."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "colab", "faculty_q9_end_to_end.ipynb")
SCRIPTS = ["devmap.py", "features.py", "train_lib.py", "10_end_to_end.py"]


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
# Faculty question 9: let GraphCodeBERT make the decision itself

**Question:** why add a Random Forest? Why not tweak the Transformer's last layer and let it decide?

**Test:** give ONE neural network exactly what our hybrid gets (the code + the counts + the ratios) and train it end to end, so its own last layer decides CPU or GPU. Then compare on the same exam (new applications) and the same piles:

| | AMD | NVIDIA |
|---|---|---|
| GraphCodeBERT fine-tuned, numbers as text (done) | 72.5 | 64.5 |
| Hybrid: GraphCodeBERT + Random Forest (done) | 75.8 | 72.1 |
| **This notebook: GraphCodeBERT + numbers, end to end** | ? | ? |

**How to run:**
1. **Runtime → Change runtime type → T4 GPU**, save.
2. **Runtime → Run all.** Allow Google Drive (tick **Select all**).
3. Each run prints its progress. Seed 1 for AMD and NVIDIA comes first (that is the important part); seeds 2 and 3 are repeats.
4. **If you must stop** (or Colab disconnects): every finished pile is saved in `devmap_results` on your Drive. Run all again later and it carries on.
5. Send me a screenshot of the last cell when you stop.
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
MACHINES = ["amd", "nvidia"]
SEEDS = [1, 2, 3]          # seed 1 first for both machines; 2 and 3 are repeats

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
E = importlib.import_module("10_end_to_end")
E.run(machines=MACHINES, seeds=SEEDS, data_dir="data", out_dir=RESULTS_DIR)
print("\\nAll requested runs are finished.")
"""),
    md("## Results so far (send Claude a screenshot)"),
    code("""
import importlib, pandas as pd
E = importlib.import_module("10_end_to_end")
pd.set_option("display.width", 200)
print(E.summary(RESULTS_DIR).to_string(index=False))
print("\\nFor comparison, the hybrid (GraphCodeBERT + Random Forest): AMD 75.8, NVIDIA 72.1")
"""),
    md("## Download the results to your laptop too (a backup in case Drive fails)"),
    code("""
import glob, os, shutil
from google.colab import files
os.makedirs("backup", exist_ok=True)
for f in glob.glob(os.path.join(RESULTS_DIR, "preds__graphcodebert+e2e__*")):
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
