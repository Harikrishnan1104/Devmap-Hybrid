"""Offline check of 10_end_to_end.py with a tiny untrained model (plumbing only)."""
import importlib, os, shutil, tempfile
import pandas as pd
import devmap, train_lib
from test_train_lib import StubTokenizer, make_encoder

E = importlib.import_module("10_end_to_end")

def main():
    out = tempfile.mkdtemp()
    quiet = lambda *a, **k: None
    s = {"max_len": 48, "epochs": 2, "lr": 1e-3, "train_batch": 16}
    try:
        E.run(seeds=(1,), out_dir=out, settings=s, tokenizer=StubTokenizer(), make_encoder=make_encoder,
              device="cpu", log=quiet)
        for m in ["amd", "nvidia"]:
            p = train_lib.result_file(out, E.LABEL, m, "unseen_app", 1, 1)
            d = pd.read_csv(p)
            assert len(d) == 680 and d.row_id.nunique() == 680 and not d.code_seen_in_study.any()
            # same piles as the fine-tuned GraphCodeBERT runs
            ref = pd.read_csv(os.path.join("..", "results", "colab_all",
                              f"preds__graphcodebert+fast__{m}__unseen_app__split1__train1.csv"))
            a = d.groupby("round")["row_id"].apply(frozenset); b = ref.groupby("round")["row_id"].apply(frozenset)
            assert a.equals(b), "piles differ from graphcodebert+fast"
            print(m, "accuracy (untrained stub)", round(100 * (d.label == d.pred).mean(), 1))
        # resume: drop a round, rerun, expect identical predictions
        p = train_lib.result_file(out, E.LABEL, "amd", "unseen_app", 1, 1)
        full = pd.read_csv(p); full[full["round"] != "pile3"].to_csv(p, index=False)
        E.run(machines=("amd",), seeds=(1,), out_dir=out, settings=s, tokenizer=StubTokenizer(),
              make_encoder=make_encoder, device="cpu", log=quiet)
        again = pd.read_csv(p).sort_values("row_id").reset_index(drop=True)
        full = full.sort_values("row_id").reset_index(drop=True)
        assert (again.pred.values == full.pred.values).all(), "resume changed predictions"
        print("resume gives identical predictions: ok")
        for f in os.listdir(os.path.join("..", "results", "colab_all")):
            if f.startswith("preds__graphcodebert+fast__") and "unseen_app" in f:
                shutil.copy(os.path.join("..", "results", "colab_all", f), out)
        print(E.summary(out).to_string(index=False))
        print("ALL E2E CHECKS PASSED")
    finally:
        shutil.rmtree(out, ignore_errors=True)

if __name__ == "__main__":
    main()
