"""
Offline check of embed_lib.py, 05_hybrid.py and colab/extract_embeddings.ipynb.
The real pretrained models cannot be downloaded here, so a tiny untrained
stand-in model makes the embeddings. This proves the plumbing works; it says
nothing about how good the real embeddings are.
"""
import importlib
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

import devmap
import embed_lib
from test_train_lib import StubTokenizer, make_encoder

HERE = os.path.dirname(os.path.abspath(__file__))
hybrid = importlib.import_module("05_hybrid")
quiet = lambda *a, **k: None


def check_embed(tmp):
    path = embed_lib.embed("stub", devmap.DATA_DIR, tmp, tokenizer=StubTokenizer(), make_encoder=make_encoder,
                           device="cpu", max_len=48, log=quiet)
    z = np.load(path)
    assert z["mean"].shape == (249, 32) and z["cls"].shape == (249, 32) and len(set(z["ids"])) == 249
    assert np.isfinite(z["mean"]).all()
    for machine in ["amd", "nvidia"]:                       # every row of both sheets finds its vector
        vec = embed_lib.load_embeddings(path)
        assert all(i in vec for i in embed_lib.code_ids(devmap.load(machine)))
    again = embed_lib.embed("stub", devmap.DATA_DIR, tmp, log=quiet)   # existing file is kept
    assert again == path
    print("embeddings: 249 kernels, every row of both machines matched: ok")
    return path


def check_pieces(tmp):
    tok = StubTokenizer()
    ids, texts = embed_lib.unique_codes()
    longest = max(texts, key=len)
    parts = embed_lib.split_into_pieces(longest, tok, 48)
    assert " ".join(parts).split(" ") == longest.split(" "), "pieces lost or reordered words"
    assert all(embed_lib.n_tokens(tok, p) <= 48 or len(p.split(" ")) <= 24 for p in parts)
    assert len(parts) > 10
    path = embed_lib.embed("stub-pieces", devmap.DATA_DIR, tmp, tokenizer=tok, make_encoder=make_encoder,
                           device="cpu", max_len=48, pieces=True, log=quiet)
    z = np.load(path)
    assert z["mean"].shape == (249, 32) and np.isfinite(z["mean"]).all()
    assert (z["n_pieces"] > 1).sum() > 100 and z["n_pieces"].max() == max(
        len(embed_lib.split_into_pieces(t, tok, 48)) for t in texts)
    print(f"reading in pieces: every word kept in order; {int((z['n_pieces'] > 1).sum())} kernels "
          f"split, up to {int(z['n_pieces'].max())} pieces: ok")


def check_no_exam_influence(path):
    """The squeeze (PCA) and the forest must only learn from study rows: changing
    the vectors of exam-only kernels may change exam answers, but must not change
    what the model learned. Check by predicting the same study rows twice."""
    df = devmap.load("amd").set_index("row_id", drop=False)
    r = devmap.build_exam(df, "unseen_app", 1)[0]
    study, ex = df.loc[r["study"]], df.loc[r["exam"]]
    vec = embed_lib.load_embeddings(path)
    probe = study.iloc[:50]
    a = hybrid.predict_round(study, probe, vec, hybrid.EXTRAS["emb+ratios"], 1)
    noisy = dict(vec)
    for i in set(embed_lib.code_ids(ex)):
        noisy[i] = vec[i] * 1000 + 7                          # wreck the exam-only kernels
    b = hybrid.predict_round(study, probe, noisy, hybrid.EXTRAS["emb+ratios"], 1)
    assert (a == b).all()
    print("exam kernels have no influence on what is learned: ok")


def check_hybrid(tmp, emb_dir):
    out = os.path.join(tmp, "results")
    summary = hybrid.main(emb_dir, models=("stub",), seeds=[1], results=out, log=quiet)
    preds = pd.read_csv(os.path.join(out, "hybrid_predictions.csv"))
    sizes = preds.groupby(["machine", "exam", "student"]).size()
    assert len(sizes) == 2 * 4 * len(hybrid.EXTRAS) and (sizes == 680).all(), sizes
    print("hybrid: every variant answered all 680 rows on every exam and machine: ok")


def check_notebook(tmp):
    nb = json.load(open(os.path.join(HERE, "..", "colab", "extract_embeddings.ipynb")))
    work = os.path.join(tmp, "nb")
    os.makedirs(os.path.join(work, "data"))
    for f in devmap.FILES.values():
        shutil.copy(os.path.join(devmap.DATA_DIR, f), os.path.join(work, "data"))
    old = os.getcwd()
    os.chdir(work)
    sys.path.insert(0, work)
    ns = {"__name__": "__main__"}
    try:
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            src = "".join(cell["source"])
            if src.startswith("%%writefile "):
                first, body = src.split("\n", 1)
                with open(os.path.join(HERE, first.split()[1])) as ref:
                    assert ref.read().strip() == body.strip(), f"{first} differs from tested script"
                continue
            if "git" in src and "clone" in src:
                continue                                       # data already copied in
            if "embed_lib.embed(" in src:                      # stand-in model, real loop
                ns["MODELS"] = ["stub"]
                real = embed_lib.embed
                embed_lib.embed = lambda m, d, o, **k: real(m, d, o, tokenizer=StubTokenizer(),
                                                           make_encoder=make_encoder, device="cpu",
                                                           max_len=48, log=quiet)
                ns["embed_lib"] = embed_lib
                src = src.replace("import embed_lib\n", "")
                try:
                    exec(compile(src, f"<cell {i}>", "exec"), ns)
                finally:
                    embed_lib.embed = real
                continue
            exec(compile(src, f"<cell {i}>", "exec"), ns)
        assert os.path.exists("embeddings/embeddings__stub.npz") and os.path.exists("embeddings.zip")
    finally:
        os.chdir(old)
        sys.path.remove(work)
    print("extract_embeddings notebook runs end to end and carries the tested code: ok")


def main():
    tmp = tempfile.mkdtemp()
    try:
        emb_dir = os.path.join(tmp, "emb")
        path = check_embed(emb_dir)
        check_pieces(os.path.join(tmp, "emb_pieces"))
        check_no_exam_influence(path)
        check_hybrid(tmp, emb_dir)
        check_notebook(tmp)
        print("ALL HYBRID CHECKS PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
