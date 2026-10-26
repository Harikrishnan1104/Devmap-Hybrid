"""Builds colab/extract_embeddings.ipynb from devmap.py and embed_lib.py, so the
notebook always carries exactly the code that was tested here."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "colab", "extract_embeddings.ipynb")


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
# Code embeddings for the hybrid model (Phase 1: newer models)

This notebook does **no training**. It shows each of the 249 distinct DevMap kernels to several pretrained code models and saves the numbers each model uses to describe the code (its "embedding"). Only the code goes in, never the job numbers.

Models (each saved as its own file, so finished ones are skipped if you run it again):
- `graphcodebert`, `codeberta`: the models from the main study, reading only the first 512 tokens (already done if you ran this before)
- `graphcodebert-pieces`: GraphCodeBERT reading the **whole** kernel in pieces of 512 tokens
- `unixcoder`, `codet5p`, `codesage`, `jina-code`: newer models trained on C code; they also read whole kernels

**How to run:**
1. **Runtime → Change runtime type → T4 GPU** if you have GPU time left (about 5 minutes), otherwise **CPU** (about 30–45 minutes).
2. **Runtime → Run all.** The first cell installs an older version of the `transformers` library that the newer models need, then Colab **restarts by itself** (you may see a "session crashed" or "restarting" message; that is expected).
3. When it has reconnected, press **Runtime → Run all** again. This time it goes all the way through.
4. When Google asks for Drive access, choose your account and tick **Select all**.

If one model fails to load, the notebook prints the error and carries on with the others. Send Claude a screenshot of any error.

The files are saved in your Google Drive under `devmap_results/embeddings`.
"""),
    code("""
import os, shutil, subprocess

# The newer code models (codet5p, codesage, jina-code) ship their own code written
# for transformers version 4; version 5 removed functions they use. Install the last
# version 4 once, then Colab restarts by itself and you press Run all again.
import transformers
if int(transformers.__version__.split(".")[0]) >= 5:
    print("transformers", transformers.__version__, "found; installing version 4 for the newer models ...")
    subprocess.run(["pip", "install", "-q", "transformers>=4.44,<5"], check=True)
    print("Installed. Colab will now restart by itself.")
    print(">>> When it has reconnected, press Runtime > Run all again. <<<")
    os.kill(os.getpid(), 9)
print("transformers version:", transformers.__version__)

import torch
print("GPU available:", torch.cuda.is_available())
if not os.path.exists("GM_SOURCE_CODE"):
    subprocess.run(["git", "clone", "--depth", "1", "-q",
                    "https://github.com/marco-siino/GM_SOURCE_CODE.git"], check=True)
os.makedirs("data", exist_ok=True)
shutil.copy("GM_SOURCE_CODE/AMD_DS/dataset-devmap-amd.csv", "data")
shutil.copy("GM_SOURCE_CODE/NVIDIA_DS/dataset-devmap-nvidia.csv", "data")
print("Dataset files:", sorted(os.listdir("data")))
"""),
    md("## Helper code (no need to read or change)"),
    writefile("devmap.py"),
    writefile("embed_lib.py"),
    md("## Settings"),
    code("""
import os

MODELS = ["graphcodebert", "codeberta", "graphcodebert-pieces",
          "unixcoder", "codet5p", "codesage", "jina-code"]
SAVE_TO_GOOGLE_DRIVE = True

OUT_DIR = "embeddings"
try:
    from google.colab import drive
    if SAVE_TO_GOOGLE_DRIVE:
        drive.mount("/content/drive")            # a Google permission window opens: tick "Select all"
        OUT_DIR = "/content/drive/MyDrive/devmap_results/embeddings"
except ImportError:
    pass
os.makedirs(OUT_DIR, exist_ok=True)
print("Embeddings will be saved in:", OUT_DIR)
"""),
    md("## Extract"),
    code("""
import gc, traceback
import torch
import embed_lib

failed = []
for model in MODELS:
    print(f"\\n=== {model} ===")
    try:
        embed_lib.embed(model, "data", OUT_DIR)
    except Exception as err:
        failed.append(model)
        print(f"!!! {model} failed: {type(err).__name__}: {err}")
        traceback.print_exc(limit=2)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
print("\\nSaved files:", sorted(os.listdir(OUT_DIR)))
print("Failed models:", failed or "none")
"""),
    md("## Download\n\nThe files are already in your Google Drive. This also zips and downloads them; send the zip (or the whole `devmap_results` folder) to Claude."),
    code("""
import shutil
shutil.make_archive("embeddings", "zip", OUT_DIR)
try:
    from google.colab import files
    files.download("embeddings.zip")
except ImportError:
    print("Saved embeddings.zip in the current folder.")
"""),
]

nb = {
    "cells": cells,
    "metadata": {"colab": {"provenance": []},
                 "kernelspec": {"display_name": "Python 3", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 0,
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as fh:
    json.dump(nb, fh, indent=1)
print("wrote", os.path.normpath(OUT), "with", len(cells), "cells")
