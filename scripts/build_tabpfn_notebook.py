"""Builds colab/phase2_tabpfn.ipynb from the tested scripts, so the notebook
always carries exactly the code that was tested here."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "colab", "phase2_tabpfn.ipynb")
SCRIPTS = ["devmap.py", "embed_lib.py", "features.py", "05_hybrid.py", "08_decision_makers.py"]


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
# Phase 2: TabPFN as the decision-maker

TabPFN is a pretrained Transformer made for small tables (Hollmann et al., *Nature*, 2025). Here it replaces the Random Forest as the part that makes the final CPU/GPU decision, on exactly the same inputs and exams as before. No code model is trained here.

We use **TabPFN version 2** (`Prior-Labs/TabPFN-v2-clf`), the model described in the *Nature* paper. It is openly downloadable: **no Hugging Face account or token is needed.**

**How to run:**
1. **Runtime → Change runtime type → T4 GPU** (about 15–20 minutes; on CPU it can take a few hours).
2. **Runtime → Run all.** The first cell installs TabPFN 2.0.9 and the library versions it needs, then Colab **restarts by itself** (a "session crashed"/"restarting" message is expected).
3. When it has reconnected, press **Runtime → Run all** again. Allow Google Drive access (tick **Select all**).

It reads the GraphCodeBERT embeddings already in your Drive (`devmap_results/embeddings`) and saves its results in `devmap_results/phase2`.
"""),
    code("""
import os, shutil, subprocess
from importlib.metadata import version, PackageNotFoundError

# TabPFN 2.0.9 downloads the openly available TabPFN-v2 model (no login needed).
# It needs slightly older versions of a few libraries, so after installing it the
# notebook restarts once; then press Runtime > Run all again.
try:
    installed = version("tabpfn")
except PackageNotFoundError:
    installed = None
if installed != "2.0.9":
    print("Installing TabPFN 2.0.9 ...")
    subprocess.run(["pip", "install", "-q", "tabpfn==2.0.9"], check=True)
    print(">>> Installed. Colab will now restart by itself. When it has reconnected, press Runtime > Run all again. <<<")
    os.kill(os.getpid(), 9)
print("TabPFN version:", installed)
if not os.path.exists("GM_SOURCE_CODE"):
    subprocess.run(["git", "clone", "--depth", "1", "-q",
                    "https://github.com/marco-siino/GM_SOURCE_CODE.git"], check=True)
os.makedirs("data", exist_ok=True)
shutil.copy("GM_SOURCE_CODE/AMD_DS/dataset-devmap-amd.csv", "data")
shutil.copy("GM_SOURCE_CODE/NVIDIA_DS/dataset-devmap-nvidia.csv", "data")
import torch
print("GPU available:", torch.cuda.is_available())
"""),
    md("## Helper code (no need to read or change)"),
    *[writefile(n) for n in SCRIPTS],
    md("## Settings"),
    code("""
import os
LEARNERS = ["tabpfn"]
INPUTS = ["hand", "ratios", "hybrid"]
SEEDS = [1, 2, 3, 4, 5]

EMB_DIR, RESULTS_DIR = "embeddings", "phase2"
try:
    from google.colab import drive
    drive.mount("/content/drive")               # tick "Select all"
    EMB_DIR = "/content/drive/MyDrive/devmap_results/embeddings"
    RESULTS_DIR = "/content/drive/MyDrive/devmap_results/phase2"
except ImportError:
    pass
os.makedirs(RESULTS_DIR, exist_ok=True)
print("Embeddings from:", EMB_DIR, "| results to:", RESULTS_DIR)
print("GraphCodeBERT embeddings found:", os.path.exists(os.path.join(EMB_DIR, "embeddings__graphcodebert.npz")))
"""),
    md("## Run"),
    code("""
import importlib
dm = importlib.import_module("08_decision_makers")
summary = dm.main(EMB_DIR, learners=LEARNERS, kinds=INPUTS, seeds=SEEDS, results=RESULTS_DIR,
                  tag="_tabpfn", data_dir="data")
print("\\nDone. Results saved in", RESULTS_DIR)
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
