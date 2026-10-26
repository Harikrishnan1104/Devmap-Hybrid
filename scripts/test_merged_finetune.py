"""Offline check of 14_merged_finetune.py with a tiny untrained model (plumbing only)."""
import importlib, shutil, tempfile
import pandas as pd
from test_train_lib import StubTokenizer, make_encoder

M = importlib.import_module("14_merged_finetune")
mg = importlib.import_module("13_merged")

def main():
    out = tempfile.mkdtemp(); quiet = lambda *a, **k: None
    s = {"max_len": 48, "epochs": 1, "lr": 1e-3, "train_batch": 32}
    try:
        df = M.merged_table()
        rounds = M.merged_rounds(df)
        same = mg.rounds_for(mg.merged_table(), "unseen_app", 1)
        assert [r["exam"] for r in rounds] == [r["exam"] for r in same], "piles differ from the RF merged test"
        for r in rounds:
            ex, st = df.loc[r["exam"]], df.loc[r["study"]]
            assert not set(ex.app) & set(st.app) and not set(ex.program_id) & set(st.program_id)
            assert set(ex.base_row[ex.machine == "amd"]) == set(ex.base_row[ex.machine == "nvidia"]), "copies split"
        print("piles: both copies of every kernel together, no shared code or application: ok")
        assert M.text_with_gpu(df.head(1))[0].startswith("AMD Radeon HD 7970 - ")
        kw = dict(seeds=(1,), out_dir=out, settings=s, tokenizer=StubTokenizer(), make_encoder=make_encoder,
                  device="cpu", log=quiet)
        M.run(**kw)
        for way in ["merged", "separate"]:
            d = pd.read_csv(M.result_path(out, way, 1))
            assert len(d) == 1360 and d.merged_row.nunique() == 1360, (way, len(d))
        p = M.result_path(out, "merged", 1)
        full = pd.read_csv(p); full[full["round"] != "pile2"].to_csv(p, index=False)
        log = []
        M.run(**{**kw, "log": log.append})
        assert sum("already done" in l for l in log) == 9, log     # 4 merged + 5 separate skipped
        again = pd.read_csv(p).sort_values("merged_row").reset_index(drop=True)
        assert (again.pred.values == full.sort_values("merged_row").pred.values).all()
        print("resume: only the missing pile is redone, same predictions: ok")
        print(M.summary(out).to_string(index=False))
        print("ALL MERGED FINE-TUNE CHECKS PASSED")
    finally:
        shutil.rmtree(out, ignore_errors=True)

if __name__ == "__main__":
    main()
