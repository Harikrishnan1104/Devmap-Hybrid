"""
Faculty point 7b, GPU part: fine-tune GraphCodeBERT on AMD + NVIDIA merged
into one dataset (1,360 rows).

The same code has different answers on the two machines (37.5% of rows), so
the model is told which GPU each row is for, at the start of its input text:
    "<GPU name> - <transfer> - <wgsize> - <code>"
Piles are dealt by application over the merged table (both copies of a kernel
always on the same side; exam code never in the study rows).

On the SAME piles, two ways are trained with the same recipe as
graphcodebert+fast (Adam 1e-5, 10 passes, batch 4, fast mode):
  merged     one model for both machines
  separate   one model per machine (the usual way), for a fair comparison
Order: seed 1 for all of them first, then seeds 2 and 3. Every pile is saved at
once (file per way and seed) and finished piles are skipped on a rerun.
The Random Forest version of this test (13_merged.py, same piles for seed 1):
separate 80.9 / 67.4, merged 80.2 / 66.7 (AMD / NVIDIA, hand features).
"""
import os

import numpy as np
import pandas as pd

import devmap
import train_lib

GPU_NAME = {"amd": "AMD Radeon HD 7970", "nvidia": "NVIDIA GeForce GTX 970"}
EXAM = "unseen_app_merged"


def merged_table(data_dir=devmap.DATA_DIR):
    parts = []
    for flag, m in enumerate(["amd", "nvidia"]):
        d = devmap.load(m, data_dir).copy()
        d["base_row"], d["machine"], d["nvidia"] = d["row_id"], m, flag
        d["row_id"] = d["row_id"] + 1000 * flag
        parts.append(d)
    return pd.concat(parts, ignore_index=True).set_index("row_id", drop=False)


def merged_rounds(df, seed=1):
    pile = devmap.group_piles(df["app"], df["label"], 5, seed)
    out = []
    for i in range(5):
        ex, st = df[pile == i], df[pile != i]
        st = st[~st["program_id"].isin(ex["program_id"])]
        out.append({"name": f"pile{i + 1}", "study": st.index.tolist(), "exam": ex.index.tolist()})
    return out


def text_with_gpu(d):
    text = (d["machine"].map(GPU_NAME) + " - " + d["transfer"].astype(str) + " - "
            + d["wgsize"].astype(str) + " - " + d["src"].astype(str))
    return [" ".join(t.split()) for t in text]


def result_path(out_dir, way, seed):
    return os.path.join(out_dir, f"preds__graphcodebert+{way}__both__{EXAM}__split1__train{seed}.csv")


def run(seeds=(1, 2, 3), ways=("merged", "separate"), data_dir=devmap.DATA_DIR, out_dir="results",
        settings=None, tokenizer=None, make_encoder=None, device=None, log=print):
    import torch
    settings = {**train_lib.DEFAULT_SETTINGS, "fast": True, **(settings or {})}
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if tokenizer is None or make_encoder is None:
        from transformers import AutoModel, AutoTokenizer
        hub = train_lib.MODELS["graphcodebert"]
        tokenizer = AutoTokenizer.from_pretrained(hub)
        make_encoder = lambda: AutoModel.from_pretrained(hub)
    os.makedirs(out_dir, exist_ok=True)
    df = merged_table(data_dir)
    rounds = merged_rounds(df, seed=1)
    for seed in seeds:
        for way in ways:
            path = result_path(out_dir, way, seed)
            done = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
            log(f"\n=== GraphCodeBERT {way} | both machines | merged piles | training seed {seed} ===")
            for r in rounds:
                want = set(df.loc[r["exam"], "row_id"])
                if len(done) and set(done.loc[done["round"] == r["name"], "merged_row"]) == want:
                    log(f"   round {r['name']}: already done, skipping")
                    continue
                if len(done):
                    done = done[done["round"] != r["name"]]
                study, ex = df.loc[r["study"]], df.loc[r["exam"]]
                prob = np.zeros(len(ex))
                groups = [(None, np.ones(len(ex), bool), np.ones(len(study), bool))] if way == "merged" else \
                    [(m, (ex["machine"] == m).to_numpy(), (study["machine"] == m).to_numpy()) for m in ["amd", "nvidia"]]
                for m, e_mask, s_mask in groups:
                    log(f"   round {r['name']}{'' if m is None else ' ' + m}: study {int(s_mask.sum())} rows, "
                        f"exam {int(e_mask.sum())} rows")
                    _, p = train_lib.train_and_predict_fast(study[s_mask], ex[e_mask], tokenizer, make_encoder,
                                                            device, settings, seed, log, text_fn=text_with_gpu)
                    prob[e_mask] = p
                part = pd.DataFrame({
                    "model": f"graphcodebert+{way}", "machine": ex["machine"].to_numpy(), "exam": EXAM,
                    "split_seed": 1, "train_seed": seed, "round": r["name"],
                    "merged_row": ex["row_id"].to_numpy(), "row_id": ex["base_row"].to_numpy(),
                    "benchmark": ex["benchmark"].to_numpy(), "suite": ex["suite"].to_numpy(),
                    "app": ex["app"].to_numpy(), "label": ex["label"].to_numpy(),
                    "pred": (prob >= 0.5).astype(int), "prob_gpu": prob})
                done = pd.concat([done, part], ignore_index=True)
                done.to_csv(path, index=False)
                acc = (part.label == part.pred).groupby(part.machine).mean() * 100
                log(f"   round {r['name']}: exam accuracy AMD {acc.get('amd', np.nan):.1f}%, "
                    f"NVIDIA {acc.get('nvidia', np.nan):.1f}%")


def summary(out_dir):
    import glob
    files = glob.glob(os.path.join(out_dir, f"preds__graphcodebert+*__both__{EXAM}__*.csv"))
    if not files:
        return pd.DataFrame()
    P = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    P["correct"] = (P.label == P.pred).astype(int)
    complete = P.groupby(["model", "train_seed"]).merged_row.transform("nunique") == 1360
    per_run = (100 * P[complete].groupby(["model", "machine", "train_seed"]).correct.mean())
    t = per_run.groupby(["model", "machine"]).agg(["mean", "count"]).round(1).unstack("machine")
    t.columns = [f"{m} {'accuracy' if s == 'mean' else 'runs'}" for s, m in t.columns]
    return t.reset_index()
