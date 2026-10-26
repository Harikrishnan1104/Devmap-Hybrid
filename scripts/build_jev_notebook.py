"""Builds colab/faculty_q10_jev_alternatives.ipynb from the tested scripts."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "colab", "faculty_q10_jev_alternatives.ipynb")
SCRIPTS = ["devmap.py", "15_jev_alternatives.py"]


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
# Faculty point 10: open-source alternatives to Jev, fine-tuned on DevMap

**Jev** (TypeSafe AI, Sept 2026) is a new *closed* classifier: you give it a text and a question with fixed options, it returns a probability per option. It cannot be fine-tuned. Its open-source alternatives can:

- **Laya** (Convai): ModernBERT-large, 421M parameters, Apache-2.0
- **Kev-0.8B** (Jared Palmer): Qwen3.5-0.8B + LoRA, Apache-2.0

Each is fine-tuned on DevMap **with its own official training code**, on the same folds as all our other models (new applications, 5 piles). Every row becomes the question *"Which device runs this OpenCL kernel faster for this job: CPU or GPU?"*, with the GPU name, data size, work-group size and kernel code as the text.

**How to run:**
1. **Runtime → Change runtime type → T4 GPU**, save.
2. **Runtime → Run all.** Allow Google Drive (tick **Select all**).
3. Order: **Laya** on AMD and NVIDIA first (about 1 hour), then **Kev** (slower, a few hours). Each finished pile is saved to `devmap_results` on your Drive at once.
4. **If you must stop** (or Colab disconnects): run all again later; finished piles are skipped.
5. At the end (or when you stop), run the last two cells and send Claude the screenshot and the downloaded `results_backup.zip`.
"""),
    code("""
import os, shutil, subprocess, sys, torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE - use Runtime > Change runtime type > T4 GPU")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers>=5.17,<6", "peft>=0.21", "accelerate",
                "datasets", "pydantic", "laya==0.4.15"], check=True)
if not os.path.exists("kev"):
    subprocess.run(["git", "clone", "--depth", "1", "-q", "https://github.com/jaredpalmer/kev.git"], check=True)
if not os.path.exists("GM_SOURCE_CODE"):
    subprocess.run(["git", "clone", "--depth", "1", "-q", "https://github.com/marco-siino/GM_SOURCE_CODE.git"], check=True)
os.makedirs("data", exist_ok=True)
shutil.copy("GM_SOURCE_CODE/AMD_DS/dataset-devmap-amd.csv", "data")
shutil.copy("GM_SOURCE_CODE/NVIDIA_DS/dataset-devmap-nvidia.csv", "data")
import transformers; print("transformers", transformers.__version__, "| Ready.")
"""),
    md("## Helper code (no need to read or change)"),
    *[writefile(n) for n in SCRIPTS],
    md("## Settings"),
    code("""
import os
MODELS = ["laya", "kev"]        # Laya first (faster), then Kev
MACHINES = ["amd", "nvidia"]
SEEDS = [1]                     # one training run first; add 2, 3 later for repeats

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
    md("## Fine-tune and sit the exam (resumable)"),
    code("""
import importlib, os
os.environ["USE_TF"] = "0"          # Laya's tip: avoids a TensorFlow hang while loading
J = importlib.import_module("15_jev_alternatives")
make = {"laya": lambda: J.LayaRunner(), "kev": lambda: J.KevRunner(kev_dir="kev")}
for name in MODELS:                  # one model at a time, both machines
    J.run([make[name]()], machines=MACHINES, seeds=SEEDS, data_dir="data", out_dir=RESULTS_DIR)
print("\\nAll requested runs are finished.")
"""),
    md("## Results so far (send Claude a screenshot)"),
    code("""
import importlib, pandas as pd
J = importlib.import_module("15_jev_alternatives")
pd.set_option("display.width", 200)
print(J.summary(RESULTS_DIR).to_string(index=False))
print("\\nFor comparison: hybrid (GraphCodeBERT + Random Forest) 74.4 / 73.7 on these same piles (AMD / NVIDIA)")
"""),
    md("## Download the results to your laptop too (a backup in case Drive fails)"),
    code("""
import glob, os, shutil
from google.colab import files
os.makedirs("backup", exist_ok=True)
for pattern in ["preds__laya__*", "preds__kev__*"]:
    for f in glob.glob(os.path.join(RESULTS_DIR, pattern)):
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
