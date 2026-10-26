"""Builds colab/phase3_agent.ipynb from the tested scripts."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "colab", "phase3_agent.ipynb")
SCRIPTS = ["devmap.py", "embed_lib.py", "features.py", "agent_lib.py", "06_repeats.py", "09_agent_eval.py"]


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
# Phase 3: the junior + senior (AI agent) experiment

- **Junior:** a Random Forest answers every exam row and says how sure it is.
- **Senior:** Gemini looks only at the rows where the junior is less than 70% sure, about 470 rows over both GPUs.
- The senior is tested in 4 ways: code only; code + fact sheet; code + all three tools at once; and as an **agent** that chooses its own tools.

Exam: new applications (the fair exam), same piles as the fine-tuned models.

**Before you start:** your Gemini key must be saved in Colab **Secrets** as `GEMINI_API_KEY`, with **Notebook access** switched on.

**How to run:**
1. **No GPU needed:** **Runtime → Change runtime type → CPU** (the AI runs on Google's servers, so save your GPU time).
2. **Runtime → Run all.** Allow Google Drive access (tick **Select all**). If Colab asks whether this notebook may read the secret `GEMINI_API_KEY`, click **Grant access**.
3. The "Connection check" cell must print `Connection OK`. If it prints an error, send Claude a screenshot.
4. The full run is roughly 2,800 AI questions with a short pause between them (free-tier friendly), so expect **several hours**. You can leave it running.
5. **If it stops** (free daily limit, or Colab disconnects): nothing is lost. Every answer is saved to `devmap_results/phase3` in your Drive immediately. Run the notebook again later (limits reset daily) and it continues where it stopped.
"""),
    code("""
import os, shutil, subprocess
subprocess.run(["pip", "install", "-q", "google-genai"], check=True)
if not os.path.exists("GM_SOURCE_CODE"):
    subprocess.run(["git", "clone", "--depth", "1", "-q",
                    "https://github.com/marco-siino/GM_SOURCE_CODE.git"], check=True)
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
MODEL = "gemini-3.5-flash-lite"     # cheap, fast Gemini model; free tier
PAUSE_SECONDS = 4.0                  # wait between questions (keeps under free-tier per-minute limits)
VARIANTS = ["all_tools", "agent", "facts", "code"]   # most important first
LIMIT = None                         # e.g. 5 to try only 5 rows per variant first

KEY = None
EMB_PATH, OUT_DIR = "embeddings/embeddings__graphcodebert.npz", "phase3"
try:
    from google.colab import userdata, drive
    KEY = userdata.get("GEMINI_API_KEY")
    drive.mount("/content/drive")                     # tick "Select all"
    EMB_PATH = "/content/drive/MyDrive/devmap_results/embeddings/embeddings__graphcodebert.npz"
    OUT_DIR = "/content/drive/MyDrive/devmap_results/phase3"
except ImportError:
    KEY = os.environ.get("GEMINI_API_KEY")
os.makedirs(OUT_DIR, exist_ok=True)
print("Key found:", bool(KEY))
print("GraphCodeBERT embeddings found:", os.path.exists(EMB_PATH))
print("Answers will be saved in:", OUT_DIR)
"""),
    md("## Connection check (one tiny question)"),
    code("""
import agent_lib as A
llm = A.gemini_llm(KEY, MODEL, pause=PAUSE_SECONDS)
try:
    reply, usage = llm('Reply with exactly this JSON object: {"ok": true}')
    print("Connection OK. Reply:", reply.strip()[:80], "| tokens:", usage)
except Exception as err:
    print("Connection FAILED:", type(err).__name__, str(err)[:300])
    try:
        from google import genai
        names = [m.name for m in genai.Client(api_key=KEY).models.list() if "flash" in m.name]
        print("Models your key can use (set MODEL to one of these, without 'models/'):", names)
    except Exception as err2:
        print("Could not list models either:", str(err2)[:200])
"""),
    md("## Run (resumable)"),
    code("""
import agent_lib as A
path = A.run_all(llm, MODEL, "data", EMB_PATH, OUT_DIR, variants=VARIANTS, limit=LIMIT)
print("\\nAnswers saved in", path)
"""),
    md("## Results so far"),
    code("""
import importlib
ev = importlib.import_module("09_agent_eval")
results = ev.main(OUT_DIR, data_dir="data")
"""),
]

nb = {"cells": cells,
      "metadata": {"colab": {"provenance": []},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as fh:
    json.dump(nb, fh, indent=1)
print("wrote", os.path.normpath(OUT), "with", len(cells), "cells")
