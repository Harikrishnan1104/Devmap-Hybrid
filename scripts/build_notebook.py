"""Builds colab/recreate_base_paper.ipynb from devmap.py and train_lib.py, so the
notebook always carries exactly the code that was tested here."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "colab", "recreate_base_paper.ipynb")


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
# Re-creating the base paper's models on DevMap

This notebook trains the base paper's Transformer models the same way the authors did, and lets the same model sit our fairer exams.

**Before you run anything:** open the menu **Runtime → Change runtime type**, choose **T4 GPU**, and save. Then use **Runtime → Run all**.

**Words used here**
- *row*: one line of the dataset (code + two job numbers + the measured answer, CPU or GPU)
- *program*: one piece of code; several rows can share it
- *study rows*: what the model learns from. *exam rows*: what it is scored on
- *pass*: one full read-through of the study rows during training
"""),
    md("## 1. Setup: check the GPU and download the dataset"),
    code("""
import os, shutil, subprocess
import torch

print("GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("No GPU found. Training will be extremely slow. Use Runtime > Change runtime type > T4 GPU.")

# The dataset lives in the base paper's public code repository.
if not os.path.exists("GM_SOURCE_CODE"):
    subprocess.run(["git", "clone", "--depth", "1", "-q",
                    "https://github.com/marco-siino/GM_SOURCE_CODE.git"], check=True)
os.makedirs("data", exist_ok=True)
shutil.copy("GM_SOURCE_CODE/AMD_DS/dataset-devmap-amd.csv", "data")
shutil.copy("GM_SOURCE_CODE/NVIDIA_DS/dataset-devmap-nvidia.csv", "data")
print("Dataset files:", sorted(os.listdir("data")))
"""),
    md("""
## 2. Helper code (no need to read or change)

The next two cells save two helper files. `devmap.py` loads the dataset and builds the exams. `train_lib.py` is the training procedure copied from the authors' notebooks.
"""),
    writefile("devmap.py"),
    writefile("train_lib.py"),
    md("""
## 3. Settings — the only cell you change

**Run 1 (done):** CodeBERTa on the paper's own exam, the paper's exact way (`FAST_MODE = False`). We got 92.1% (AMD) and 88.8% (NVIDIA) against the paper's 92.8% and 89.1%.

**Run 2 (now):** leave everything as it is. CodeBERTa sits all four exams with the speed-ups on. The most important exam (`unseen_app`) runs first, for both machines. The old exam is included again so we can check the speed-ups give the same scores as Run 1.

**Run 3 (later):** set `MODELS_TO_RUN = ["graphcodebert"]` for our new model, keeping the same exams.

`FAST_MODE = True` pads each batch only to its longest input and uses half precision on the GPU. Results made this way are labelled `+fast` (for example `codeberta+fast`) so they never mix with Run 1.

This copy deals the `unseen_app` and `unseen_program` piles with our own code, so the piles are even and identical on every computer. Rounds saved by an older copy of this notebook with different piles are detected and redone automatically; everything else is kept.

Finished work is saved after every round, straight into a folder called `devmap_results` in your Google Drive. If Colab disconnects, press **Reconnect** and **Run all** again: finished rounds are skipped and it carries on where it stopped.

When this cell runs, Google asks for permission to use your Drive. Choose your account and press **Allow** (or **Continue**). The notebook only writes inside that one folder.
"""),
    code("""
import os

MODELS_TO_RUN = ["codeberta"]        # choose from: "codeberta", "distilbert", "codebert", "graphcodebert"
MACHINES = ["amd", "nvidia"]         # the two answer sheets
EXAMS = ["unseen_app", "old", "unseen_program", "unseen_suite"]   # run in this order; most important first
SHUFFLE_SEEDS = [1]                  # different shuffles for the two reshuffled fair exams, e.g. [1, 2, 3, 4, 5]
TRAINING_SEEDS = [1]                 # different training runs, e.g. [1, 2, 3]
FAST_MODE = True                     # False = exactly the paper's way (slower)

SAVE_TO_GOOGLE_DRIVE = True         # keeps finished rounds safe if Colab disconnects

DATA_DIR = "data"
RESULTS_DIR = "results"
try:
    from google.colab import drive
    if SAVE_TO_GOOGLE_DRIVE:
        drive.mount("/content/drive")            # a Google permission window opens: allow it
        RESULTS_DIR = "/content/drive/MyDrive/devmap_results"
except ImportError:
    pass                                         # not running on Colab
os.makedirs(RESULTS_DIR, exist_ok=True)
print("Results will be saved in:", RESULTS_DIR)
"""),
    md("## 4. Train and sit the exams"),
    code("""
import devmap, train_lib

# Exam by exam, both machines each, so the most important results arrive first.
for model in MODELS_TO_RUN:
    for exam in EXAMS:
        for machine in MACHINES:
            shuffle_seeds = SHUFFLE_SEEDS if exam in devmap.RESHUFFLED_EXAMS else [1]
            for shuffle_seed in shuffle_seeds:
                for training_seed in TRAINING_SEEDS:
                    train_lib.run_experiment(model, machine, exam, DATA_DIR, RESULTS_DIR,
                                             split_seed=shuffle_seed, train_seed=training_seed,
                                             settings={"fast": FAST_MODE})
print("\\nAll requested runs are finished.")
"""),
    md("""
## 5. Results

- **Main table:** accuracy in % for everything finished so far. `paper` is the base paper's score on the old exam, for comparison. `spread` is how much the score moved between repeated runs (0 if there was only one).
- **Old exam, pile by pile:** our score next to the paper's for each of the 5 piles.
- **Old exam, seen vs new code:** accuracy on exam rows whose code was also in the study rows, against rows with code the model never saw.
"""),
    code("""
import pandas as pd
pd.set_option("display.width", 200)

tables = train_lib.summarise(RESULTS_DIR)
titles = {
    "main": "MAIN TABLE - accuracy (%)",
    "old_exam_piles": "OLD EXAM - pile by pile, ours and the paper's (%)",
    "old_exam_seen_vs_new": "OLD EXAM - seen code vs new code (%)",
    "per_collection": "UNSEEN-COLLECTION EXAM - per held-out collection (%)",
}
for key, title in titles.items():
    table = tables.get(key)
    if table is not None and len(table):
        print("\\n" + title)
        print(table.to_string(index=(key != "main")))
"""),
    md("""
## 6. Save the results to your device

Your results are already safe in the `devmap_results` folder in your Google Drive. This cell also zips them and tries to download the zip. **Send me the zip (or a screenshot of the tables above)** and I will merge it with the Random Forest and memoriser results. If the download fails, you can download the folder from Google Drive instead.
"""),
    code("""
import shutil
shutil.make_archive("devmap_results", "zip", RESULTS_DIR)
try:
    from google.colab import files
    files.download("devmap_results.zip")
except ImportError:
    print("Saved devmap_results.zip in the current folder.")
"""),
]

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as fh:
    json.dump(nb, fh, indent=1)
print("wrote", os.path.normpath(OUT), "with", len(cells), "cells")
