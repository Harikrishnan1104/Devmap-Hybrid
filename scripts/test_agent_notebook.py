"""Runs colab/phase3_agent.ipynb cell by cell in a clean folder. Gemini cannot be
reached from here, so the notebook's gemini_llm is swapped for the scripted fake
from test_agent.py and LIMIT is set to 3 rows per variant."""
import json, os, shutil, sys, tempfile
import devmap
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from test_agent import FakeLLM
nb = json.load(open(os.path.join(HERE, "..", "colab", "phase3_agent.ipynb")))
work = tempfile.mkdtemp(); old = os.getcwd()
os.makedirs(os.path.join(work, "data")); os.makedirs(os.path.join(work, "embeddings"))
for f in devmap.FILES.values():
    shutil.copy(os.path.join(devmap.DATA_DIR, f), os.path.join(work, "data"))
shutil.copy(os.path.join(HERE, "..", "results", "colab_all", "embeddings", "embeddings__graphcodebert.npz"),
            os.path.join(work, "embeddings"))
os.chdir(work); sys.path.insert(0, work)
for m in ["devmap", "embed_lib", "features", "agent_lib", "06_repeats", "09_agent_eval"]:
    sys.modules.pop(m, None)
ns = {"__name__": "__main__"}
try:
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if src.startswith("%%writefile "):
            first, body = src.split("\n", 1)
            name = first.split()[1]
            assert open(os.path.join(HERE, name)).read().strip() == body.strip(), f"{name} differs"
            open(name, "w").write(body)
            continue
        if "pip" in src and "install" in src:
            continue
        if "A.gemini_llm(" in src:
            import agent_lib
            agent_lib.gemini_llm = lambda *a, **k: FakeLLM()
        if "A.run_all(" in src:
            ns["LIMIT"] = 3
        exec(compile(src, f"<cell {i}>", "exec"), ns)
    import pandas as pd
    d = pd.read_csv(os.path.join("phase3", "senior_answers.csv"))
    assert len(d) == 2 * 4 * 3, len(d)
    assert "unfinished" in " ".join(ns["results"].get("note", pd.Series(dtype=str)).fillna("").tolist())
    print("PHASE 3 NOTEBOOK CHECK PASSED (fake LLM, 3 rows per variant)")
finally:
    os.chdir(old); shutil.rmtree(work, ignore_errors=True)
