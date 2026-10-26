"""
Faculty point 10: open-source alternatives to Jev, fine-tuned on DevMap.

Jev (TypeSafe AI, Sept 2026) is a closed "System One" classifier: it reads a text
("state") plus a question with fixed options and returns a probability per option.
It cannot be fine-tuned. Its open-source alternatives can, and each is trained here
with ITS OWN official fine-tuning code, on the same folds as all our other models:

  laya   Convai's Laya: ModernBERT-large (421M) + decision head, Apache-2.0.
         Fine-tuned with laya.train.finetune (the package's own loop: RLCD loss,
         calibration on a 10% slice of the training rows).
  kev    Jared Palmer's Kev-0.8B: Qwen3.5-0.8B-Base + LoRA + pointer head, Apache-2.0.
         Fine-tuned with `python -m kev.train --init_from jaredpalmer/kev-0.8b`
         (the repo's documented "delta" fine-tune), then scored with kev's loader.

Every row becomes the same typed question:
  state    = which GPU, data copied, work-group size, then the kernel code
  question = "Which device runs this OpenCL kernel faster for this job?"  options CPU / GPU
Exam: unseen_app (shuffle seed 1), 5 piles, so the results line up row by row with
graphcodebert+fast, graphcodebert+e2e, the hybrid and the Random Forest.
Results: preds__<laya|kev>__<machine>__unseen_app__split1__train<seed>.csv, one
file per model/machine/seed, saved after every pile; finished piles are skipped.
Per-pile model checkpoints are written to local disk only and deleted after use.
"""
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

import devmap

EXAM = "unseen_app"
GPU_NAME = {"amd": "AMD Radeon HD 7970", "nvidia": "NVIDIA GeForce GTX 970"}
QID = "device"
QUESTION = {
    "type": "choice",
    "instructions": "Which device runs this OpenCL kernel faster for this job: the CPU or the GPU?",
    "criteria": {"CPU": "the Intel Core i7-3820 CPU is faster", "GPU": "the GPU is faster"},
}


def make_state(row, machine):
    code = " ".join(str(row["src"]).split())
    return (f"Hardware: Intel Core i7-3820 CPU versus {GPU_NAME[machine]} GPU. "
            f"Data copied between CPU and GPU: {int(row['transfer'])} bytes. "
            f"Work-group size: {int(row['wgsize'])}. OpenCL kernel: {code}")


def label_name(y):
    return "GPU" if int(y) == 1 else "CPU"


# ----------------------------------------------------------------------------------- Laya

class LayaRunner:
    label = "laya"

    def __init__(self, base="convaiinnovations/laya", epochs=4, micro_batch=8, grad_accum=1, max_len=512,
                 device="auto", log=print):
        self.base, self.epochs, self.micro_batch, self.grad_accum = base, epochs, micro_batch, grad_accum
        self.max_len, self.device, self.log = max_len, device, log

    def fit_predict(self, study, exam, machine, seed, workdir):
        import laya
        from laya.train import TrainConfig, finetune
        os.makedirs(workdir, exist_ok=True)
        data = os.path.join(workdir, "train.jsonl")
        with open(data, "w") as f:
            for _, r in study.iterrows():
                f.write(json.dumps({"state": make_state(r, machine), "questions": {QID: QUESTION},
                                    "expected": {QID: label_name(r["label"])}}) + "\n")
        out = os.path.join(workdir, "model")
        cfg = TrainConfig(epochs=self.epochs, micro_batch=self.micro_batch, grad_accum=self.grad_accum,
                          seed=seed, max_len=self.max_len, log_every=50)
        finetune(data, self.base, out, cfg, device=self.device)
        agent = laya.load(out, device=None if self.device == "auto" else self.device)
        states = [make_state(r, machine) for _, r in exam.iterrows()]
        results = agent.predict_batch(states, {QID: QUESTION}, batch_size=16)
        del agent
        return np.array([gpu_probability(res) for res in results])


def gpu_probability(result):
    """P(GPU) from a Laya/Jev-style answer: the option probabilities if given, else the chosen option."""
    ans = result["answers"][QID]
    probs = ans.get("probabilities") or {}
    if "GPU" in probs or "CPU" in probs:
        g, c = float(probs.get("GPU", 0.0)), float(probs.get("CPU", 0.0))
        return g / (g + c) if g + c > 0 else 0.5
    return 1.0 if ans.get("choice") == "GPU" else 0.0


# ------------------------------------------------------------------------------------ Kev

class KevRunner:
    label = "kev"

    def __init__(self, kev_dir="kev", base="Qwen/Qwen3.5-0.8B-Base", init_from="jaredpalmer/kev-0.8b",
                 epochs=2, lr=2e-5, batch=1, accum=8, max_state=1024, dtype="fp32", device="auto",
                 extra_args=(), log=print):
        self.kev_dir, self.base, self.init_from = os.path.abspath(kev_dir), base, init_from
        self.epochs, self.lr, self.batch, self.accum = epochs, lr, batch, accum
        self.max_state, self.dtype, self.device, self.extra_args, self.log = max_state, dtype, device, list(extra_args), log
        self._tok = None

    def _env(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = self.kev_dir + os.pathsep + env.get("PYTHONPATH", "")
        return env

    def _kev(self):
        if self.kev_dir not in sys.path:
            sys.path.insert(0, self.kev_dir)
        from kev import checkpoint, data, model
        return checkpoint, data, model

    def _states(self, df, machine):
        """States cut so the whole state fits the training context (kev drops, never truncates, longer records)."""
        _, _, kmodel = self._kev()
        if self._tok is None:
            self._tok = kmodel.load_tokenizer(self.base)
        budget = self.max_state - 64          # room for kev's delimiters and the state token
        out = []
        for _, r in df.iterrows():
            s = make_state(r, machine)
            ids = self._tok(s, add_special_tokens=False)["input_ids"]
            out.append(s if len(ids) <= budget else self._tok.decode(ids[:budget]))
        return out

    def fit_predict(self, study, exam, machine, seed, workdir):
        os.makedirs(workdir, exist_ok=True)
        data = os.path.join(workdir, "train.jsonl")
        with open(data, "w") as f:
            for s, (_, r) in zip(self._states(study, machine), study.iterrows()):
                f.write(json.dumps({"state": s, "questions": {QID: {**QUESTION, "label": label_name(r["label"])}}}) + "\n")
        out = os.path.join(workdir, "run")
        cmd = [sys.executable, "-m", "kev.train", "--data", data, "--base", self.base,
               "--epochs", str(self.epochs), "--lr", str(self.lr), "--batch", str(self.batch), "--accum", str(self.accum),
               "--dtype", self.dtype, "--device", self.device, "--max_state", str(self.max_state),
               "--p_none", "0", "--p_none_distract", "0", "--p_distract", "0",
               "--seed", str(seed), "--out", out] + (["--init_from", self.init_from] if self.init_from else []) + self.extra_args
        self.log("      " + " ".join(cmd[1:]))
        proc = subprocess.Popen(cmd, cwd=self.kev_dir, env=self._env(), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        tail = []
        for line in proc.stdout:
            line = line.rstrip()
            tail = (tail + [line])[-40:]
            if any(k in line for k in ("epoch", "step", "dropped", "training requests", "delta:", "device=", "Error", "error")):
                self.log("      " + line[:200])
        if proc.wait() != 0:
            raise RuntimeError("kev.train failed:\n" + "\n".join(tail))
        return self.predict(out, exam, machine)

    def predict(self, run_dir, exam, machine):
        import torch
        kcheck, kdata, kmodel = self._kev()
        dev = torch.device("cuda" if torch.cuda.is_available() and self.device != "cpu" else "cpu")
        tok, m = kcheck.Checkpoint(run_dir).load(dev, kcheck.LoadOptions())
        ctx = kmodel.training_context(self.max_state)
        probs = []
        with torch.no_grad():
            for s in self._states(exam, machine):
                req = {"state": s, "questions": {QID: {**QUESTION, "label": "CPU", "src": "devmap_choice"}}}
                rec = kdata.materialize(req)
                p = m.probs(m.encode(tok, rec, max_state=ctx["max_state"], max_branch=ctx["max_branch"]))[0]
                keys = rec["questions"][0]["keys"]
                probs.append(float(p[keys.index("GPU")]))
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return np.array(probs)


# ------------------------------------------------------------------------------------ run

def result_path(out_dir, label, machine, seed):
    return os.path.join(out_dir, f"preds__{label}__{machine}__{EXAM}__split1__train{seed}.csv")


def run(runners, machines=("amd", "nvidia"), seeds=(1,), data_dir=devmap.DATA_DIR, out_dir="results",
        work_root="jev_work", log=print):
    os.makedirs(out_dir, exist_ok=True)
    for seed in seeds:
        for runner in runners:
            for machine in machines:
                df = devmap.load(machine, data_dir).set_index("row_id", drop=False)
                rounds = devmap.build_exam(df, EXAM, seed=1)
                path = result_path(out_dir, runner.label, machine, seed)
                done = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
                log(f"\n=== {runner.label} | {machine} | exam '{EXAM}' | training seed {seed} ===")
                for r in rounds:
                    if len(done) and set(done.loc[done["round"] == r["name"], "row_id"]) == set(r["exam"]):
                        log(f"   round {r['name']}: already done, skipping")
                        continue
                    if len(done):
                        done = done[done["round"] != r["name"]]
                    study, ex = df.loc[r["study"]], df.loc[r["exam"]]
                    log(f"   round {r['name']}: study {len(study)} rows, exam {len(ex)} rows")
                    work = os.path.join(work_root, f"{runner.label}_{machine}_{seed}_{r['name']}")
                    t0 = time.time()
                    try:
                        prob = runner.fit_predict(study, ex, machine, seed, work)
                    finally:
                        shutil.rmtree(work, ignore_errors=True)
                    part = pd.DataFrame({
                        "model": runner.label, "machine": machine, "exam": EXAM, "split_seed": 1, "train_seed": seed,
                        "round": r["name"], "row_id": ex["row_id"].to_numpy(), "benchmark": ex["benchmark"].to_numpy(),
                        "suite": ex["suite"].to_numpy(), "app": ex["app"].to_numpy(),
                        "program_id": ex["program_id"].to_numpy(),
                        "code_seen_in_study": ex["program_id"].isin(set(study["program_id"])).to_numpy(),
                        "transfer": ex["transfer"].to_numpy(), "wgsize": ex["wgsize"].to_numpy(),
                        "label": ex["label"].to_numpy(), "pred": (prob >= 0.5).astype(int), "prob_gpu": prob})
                    done = pd.concat([done, part], ignore_index=True)
                    done.to_csv(path, index=False)
                    log(f"   round {r['name']}: exam accuracy {100 * (part.label == part.pred).mean():.1f}%  "
                        f"({(time.time() - t0) / 60:.1f} min)")
                log(f"   overall accuracy {100 * (done.label == done.pred).mean():.1f}%")


def summary(out_dir, labels=("laya", "kev", "graphcodebert+fast", "graphcodebert+e2e", "codeberta+fast")):
    import glob
    rows = []
    for f in glob.glob(os.path.join(out_dir, f"preds__*__{EXAM}__split1__train*.csv")):
        d = pd.read_csv(f)
        if len(d) == 680 and d["model"].iloc[0] in labels:
            rows.append({"model": d["model"].iloc[0], "machine": d["machine"].iloc[0],
                         "seed": d["train_seed"].iloc[0], "acc": 100 * (d.label == d.pred).mean()})
    if not rows:
        return pd.DataFrame()
    t = pd.DataFrame(rows).groupby(["model", "machine"]).acc.agg(["mean", "count"]).round(1).unstack("machine")
    t.columns = [f"{m} {'accuracy' if s == 'mean' else 'runs'}" for s, m in t.columns]
    return t.reset_index()
