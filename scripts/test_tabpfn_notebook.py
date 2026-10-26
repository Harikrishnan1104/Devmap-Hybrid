"""Runs colab/phase2_tabpfn.ipynb cell by cell in a clean folder. TabPFN's gated
weights cannot be downloaded here, so the run uses the Random Forest learner
and one seed; everything else is the notebook's own code."""
import json, os, shutil, sys, tempfile
import devmap
HERE = os.path.dirname(os.path.abspath(__file__))
nb = json.load(open(os.path.join(HERE, "..", "colab", "phase2_tabpfn.ipynb")))
work = tempfile.mkdtemp(); old = os.getcwd()
os.makedirs(os.path.join(work, "data")); os.makedirs(os.path.join(work, "embeddings"))
for f in devmap.FILES.values():
    shutil.copy(os.path.join(devmap.DATA_DIR, f), os.path.join(work, "data"))
shutil.copy(os.path.join(HERE, "..", "results", "colab_all", "embeddings", "embeddings__graphcodebert.npz"),
            os.path.join(work, "embeddings"))
os.chdir(work); sys.path.insert(0, work)
for m in ["devmap", "embed_lib", "features", "05_hybrid", "08_decision_makers"]:
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
            continue                                   # install + dataset download: done above
        if "login()" in src:
            continue                                   # Hugging Face login: not possible here
        if "dm.main(" in src:
            ns.update(LEARNERS=["rf"], SEEDS=[1], INPUTS=["hand", "hybrid"])
        exec(compile(src, f"<cell {i}>", "exec"), ns)
    files = sorted(os.listdir("phase2"))
    assert "decision_makers_tabpfn_predictions.csv" in files, files
    import pandas as pd
    p = pd.read_csv("phase2/decision_makers_tabpfn_predictions.csv")
    assert (p.groupby(["machine", "exam", "student"]).size() == 680).all()
    print("TABPFN NOTEBOOK CHECK PASSED (run with the Random Forest stand-in)")
finally:
    os.chdir(old); shutil.rmtree(work, ignore_errors=True)
