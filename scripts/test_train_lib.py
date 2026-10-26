"""
Offline check of train_lib.py. The pretrained models cannot be downloaded in
the workspace this was written in, so this swaps in a tiny untrained model and
a throwaway tokenizer. It proves the plumbing (splits, training loop, saving,
resuming, summary tables) runs; it says nothing about real accuracy.
"""
import os
import re
import shutil
import tempfile

import torch
from transformers import RobertaConfig, RobertaModel

import devmap
import train_lib


class StubTokenizer:
    """Mimics the two ways train_lib calls a Hugging Face tokenizer:
    padded to max_length as tensors (paper's way), or unpadded lists (fast way)."""
    pad_token_id = 0

    def __call__(self, texts, padding=False, truncation=True, max_length=512, return_tensors=None):
        ids, mask = [], []
        for t in texts:
            toks = [1] + [3 + (hash(w) % 1990) for w in re.findall(r"\w+|[^\w\s]", t)][: max_length - 1]
            m = [1] * len(toks)
            if padding == "max_length":
                m = m + [0] * (max_length - len(toks))
                toks = toks + [0] * (max_length - len(toks))
            ids.append(toks)
            mask.append(m)
        if return_tensors == "pt":
            return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}
        return {"input_ids": ids, "attention_mask": mask}


def make_encoder():
    cfg = RobertaConfig(vocab_size=2000, hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                        intermediate_size=64, max_position_embeddings=70, pad_token_id=0)
    return RobertaModel(cfg)


def main():
    out = tempfile.mkdtemp()
    settings = {"max_len": 48, "epochs": 2, "lr": 1e-3, "train_batch": 16}
    quiet = lambda *a, **k: None
    try:
        for exam in devmap.EXAMS:
            d = train_lib.run_experiment("stub", "nvidia", exam, devmap.DATA_DIR, out, settings=settings,
                                         tokenizer=StubTokenizer(), make_encoder=make_encoder,
                                         device="cpu", log=quiet)
            assert len(d) == 680 and d["row_id"].nunique() == 680, (exam, len(d))
            if exam != "old":
                assert not d["code_seen_in_study"].any(), exam
            print(f"{exam:15s} rows {len(d)}  accuracy {100 * (d.label == d.pred).mean():.1f}%  "
                  f"rounds {d['round'].nunique()}  seen-code rows {int(d.code_seen_in_study.sum())}")

        # resume: delete two rounds from a file, rerun, and expect the same predictions back
        path = train_lib.result_file(out, "stub", "nvidia", "old", 1, 1)
        import pandas as pd
        full = pd.read_csv(path)
        full[full["round"].isin(["pile1", "pile2", "pile3"])].to_csv(path, index=False)
        again = train_lib.run_experiment("stub", "nvidia", "old", devmap.DATA_DIR, out, settings=settings,
                                         tokenizer=StubTokenizer(), make_encoder=make_encoder,
                                         device="cpu", log=quiet)
        a = full.sort_values("row_id")["pred"].to_numpy()
        b = again.sort_values("row_id")["pred"].to_numpy()
        print("resume ok, rows:", len(again), "| same predictions after resume:", bool((a == b).all()))
        assert len(again) == 680

        # a round saved with different exam rows (old splitting code) must be redone, not skipped
        app_path = train_lib.result_file(out, "stub", "nvidia", "unseen_app", 1, 1)
        saved = pd.read_csv(app_path)
        bad = saved[saved["round"] != "pile2"]
        bad = bad[~((bad["round"] == "pile1") & (bad.index % 7 == 0))]   # pile1 now missing some rows
        bad.to_csv(app_path, index=False)
        msgs = []
        redo = train_lib.run_experiment("stub", "nvidia", "unseen_app", devmap.DATA_DIR, out, settings=settings,
                                        tokenizer=StubTokenizer(), make_encoder=make_encoder,
                                        device="cpu", log=msgs.append)
        assert len(redo) == 680 and redo["row_id"].nunique() == 680
        assert any("pile1: saved with different exam rows" in m for m in msgs)
        assert sum("already done" in m for m in msgs) == 3      # piles 3-5 kept
        print("mismatched round redone, matching rounds kept: ok")

        # the real Colab path: no tokenizer/encoder passed in, so train_lib loads them by their
        # Hugging Face name. Fake the download and check the name asked for is the model's name,
        # also when old rounds have to be redone (this is the bug that broke Run 2 on Colab).
        import transformers
        asked = []
        real_tok, real_model = transformers.AutoTokenizer.from_pretrained, transformers.AutoModel.from_pretrained
        transformers.AutoTokenizer.from_pretrained = lambda n, *a, **k: (asked.append(n), StubTokenizer())[1]
        transformers.AutoModel.from_pretrained = lambda n, *a, **k: (asked.append(n), make_encoder())[1]
        try:
            train_lib.MODELS["stubhub"] = "fake-org/fake-model"
            hub_path = train_lib.result_file(out, "stubhub", "nvidia", "unseen_app", 1, 1)
            old = saved.assign(model="stubhub")                  # all 5 piles, correct rows
            old = old[~old["round"].isin(["pile3", "pile4"])]     # two piles not done yet
            old = old[~((old["round"] == "pile1") & (old.index % 5 == 0))]   # pile1 from different piles
            old.to_csv(hub_path, index=False)
            d = train_lib.run_experiment("stubhub", "nvidia", "unseen_app", devmap.DATA_DIR, out,
                                         settings=settings, device="cpu", log=quiet)
            assert len(d) == 680
            assert asked and set(asked) == {"fake-org/fake-model"}, asked
            print("model loaded by its own name during redo:", sorted(set(asked)), "ok")
        finally:
            transformers.AutoTokenizer.from_pretrained, transformers.AutoModel.from_pretrained = real_tok, real_model
            train_lib.MODELS.pop("stubhub", None)

        # fast mode: runs, answers every row, and is saved under its own label
        fast = {**settings, "fast": True}
        for exam in ["old", "unseen_app"]:
            d = train_lib.run_experiment("stub", "nvidia", exam, devmap.DATA_DIR, out, settings=fast,
                                         tokenizer=StubTokenizer(), make_encoder=make_encoder,
                                         device="cpu", log=quiet)
            assert len(d) == 680 and d["row_id"].nunique() == 680 and set(d["model"]) == {"stub+fast"}
            print(f"fast {exam:10s} rows {len(d)}  accuracy {100 * (d.label == d.pred).mean():.1f}%")
        assert os.path.exists(train_lib.result_file(out, "stub+fast", "nvidia", "old", 1, 1))

        # fast mode actually learns: bigger stand-in, one round, compare with always-guessing
        def bigger():
            return RobertaModel(RobertaConfig(vocab_size=2000, hidden_size=64, num_hidden_layers=2,
                                              num_attention_heads=4, intermediate_size=128,
                                              max_position_embeddings=140, pad_token_id=0))
        df = devmap.load("amd").set_index("row_id", drop=False)
        r = devmap.build_exam(df, "old")[0]
        st, ex = df.loc[r["study"]], df.loc[r["exam"]]
        cfg = {**train_lib.DEFAULT_SETTINGS, "max_len": 128, "epochs": 12, "lr": 5e-4,
               "train_batch": 16, "fast": True}
        pred, _ = train_lib.train_and_predict(st, ex, StubTokenizer(), bigger, "cpu", cfg, 1, log=quiet)
        acc = 100 * (pred == ex["label"].to_numpy()).mean()
        guess = 100 * max(ex["label"].mean(), 1 - ex["label"].mean())
        print(f"fast-mode learning check: exam accuracy {acc:.1f}% vs always-guessing {guess:.1f}%")
        assert acc > guess + 10

        tables = train_lib.summarise(out)
        for name, t in tables.items():
            print(f"\n[{name}]")
            print(t.to_string())
    finally:
        shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    main()
