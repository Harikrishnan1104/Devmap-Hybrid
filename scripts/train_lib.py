"""
Re-creates the base paper's Transformer training, step for step, and lets the
same model sit any of our four exams.

Copied from the authors' public notebooks (github.com/marco-siino/GM_SOURCE_CODE):
  * input text  = "<transfer> - <wgsize> - <code>", extra spaces removed
  * at most 512 tokens, padded to 512
  * model       = pretrained encoder -> Linear(768,768) -> ReLU -> Dropout(0.3)
                  -> Linear(768,2), reading the first token's output
  * training    = Adam, learning rate 1e-05, batch of 4, 10 passes (epochs),
                  scored once after the last pass
What we add: a fixed random seed per run (theirs was not fixed), per-row
predictions saved to disk, and the choice of exam.
"""
import glob
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import devmap

# Short name -> pretrained model on the Hugging Face hub.
MODELS = {
    "codeberta": "huggingface/CodeBERTa-language-id",                            # paper's best
    "distilbert": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",  # as in their notebook
    "codebert": "microsoft/codebert-base",                                       # their notebook loads a local copy
    "graphcodebert": "microsoft/graphcodebert-base",                             # OURS - not in the paper
}

# The paper's own scores on the old exam (Tables 3 and 4), per pile and average.
PAPER_OLD_EXAM = {
    ("codeberta", "amd"): [0.956, 0.897, 0.956, 0.890, 0.941],
    ("codeberta", "nvidia"): [0.934, 0.868, 0.904, 0.875, 0.875],
    ("distilbert", "amd"): [0.934, 0.897, 0.971, 0.875, 0.963],
    ("distilbert", "nvidia"): [0.890, 0.868, 0.875, 0.824, 0.838],
    ("codebert", "amd"): [0.956, 0.890, 0.956, 0.897, 0.897],
    ("codebert", "nvidia"): [0.912, 0.824, 0.882, 0.853, 0.860],
}

DEFAULT_SETTINGS = {
    "max_len": 512,
    "train_batch": 4,
    "eval_batch": 16,      # only affects speed, not the answers
    "epochs": 10,
    "lr": 1e-05,
    "dropout": 0.3,
    "fast": False,         # True = our speed-ups (see train_and_predict_fast); False = exactly the paper's way
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_text(df):
    """The paper's input format: the two job numbers, then the code."""
    text = df["transfer"].astype(str) + " - " + df["wgsize"].astype(str) + " - " + df["src"].astype(str)
    return [" ".join(t.split()) for t in text]


def encode(tokenizer, texts, max_len):
    enc = tokenizer(list(texts), padding="max_length", truncation=True,
                    max_length=max_len, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


class Classifier(nn.Module):
    """Same head as the paper's CodeBERTaClass."""

    def __init__(self, encoder, dropout=0.3):
        super().__init__()
        self.l1 = encoder
        hidden = encoder.config.hidden_size
        self.pre_classifier = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, 2)

    def forward(self, input_ids, attention_mask):
        hidden_state = self.l1(input_ids=input_ids, attention_mask=attention_mask)[0]
        pooler = hidden_state[:, 0]
        pooler = torch.relu(self.pre_classifier(pooler))
        pooler = self.dropout(pooler)
        return self.classifier(pooler)


def _pad(seqs, pad_id):
    """Pad a handful of token lists only up to the longest one among them."""
    longest = max(len(x) for x in seqs)
    ids = torch.full((len(seqs), longest), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), longest), dtype=torch.long)
    for i, x in enumerate(seqs):
        ids[i, :len(x)] = torch.tensor(x, dtype=torch.long)
        mask[i, :len(x)] = 1
    return ids, mask


def train_and_predict_fast(study, exam, tokenizer, make_encoder, device, settings, seed, log=print, text_fn=None):
    """Same model, same settings, same number of passes as the paper's way, with
    two speed-ups that do not change what the model is asked to learn:
      1. each batch is padded only to its own longest input, not always to 512
         tokens, so less work is spent on empty filler;
      2. on a GPU, the arithmetic runs in half precision (16-bit numbers)."""
    set_seed(seed)
    text_fn = text_fn or make_text
    pad_id = tokenizer.pad_token_id if getattr(tokenizer, "pad_token_id", None) is not None else 0
    tr_seqs = tokenizer(text_fn(study), truncation=True, max_length=settings["max_len"])["input_ids"]
    ex_seqs = tokenizer(text_fn(exam), truncation=True, max_length=settings["max_len"])["input_ids"]
    y_tr = study["label"].to_numpy()

    use_amp = str(device).startswith("cuda")
    model = Classifier(make_encoder(), settings["dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["lr"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = nn.CrossEntropyLoss()
    order_gen = torch.Generator().manual_seed(seed)
    bs = settings["train_batch"]

    for epoch in range(settings["epochs"]):
        model.train()
        total_loss, correct, seen = 0.0, 0, 0
        t0 = time.time()
        order = torch.randperm(len(tr_seqs), generator=order_gen).tolist()
        for start in range(0, len(order), bs):
            idx = order[start:start + bs]
            ids, mask = _pad([tr_seqs[i] for i in idx], pad_id)
            y = torch.tensor(y_tr[idx], dtype=torch.long)
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                out = model(ids, mask)
                loss = loss_fn(out.float(), y)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * y.size(0)
            correct += (out.argmax(dim=1) == y).sum().item()
            seen += y.size(0)
        log(f"      pass {epoch + 1:2d}/{settings['epochs']}  loss {total_loss / seen:.4f}  "
            f"study accuracy {100 * correct / seen:5.1f}%  ({time.time() - t0:.0f}s)")

    model.eval()
    by_length = sorted(range(len(ex_seqs)), key=lambda i: len(ex_seqs[i]))   # similar lengths together
    prob_gpu = np.zeros(len(ex_seqs))
    with torch.no_grad():
        for start in range(0, len(by_length), settings["eval_batch"]):
            idx = by_length[start:start + settings["eval_batch"]]
            ids, mask = _pad([ex_seqs[i] for i in idx], pad_id)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                out = model(ids.to(device), mask.to(device))
            prob_gpu[idx] = torch.softmax(out.float(), dim=1)[:, 1].cpu().numpy()
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return (prob_gpu >= 0.5).astype(int), prob_gpu


def train_and_predict(study, exam, tokenizer, make_encoder, device, settings, seed, log=print):
    """Train on the study rows, then answer the exam rows.
    Returns (predicted labels, probability of GPU) in exam-row order."""
    if settings.get("fast"):
        return train_and_predict_fast(study, exam, tokenizer, make_encoder, device, settings, seed, log)
    set_seed(seed)
    ids_tr, mask_tr = encode(tokenizer, make_text(study), settings["max_len"])
    y_tr = torch.tensor(study["label"].to_numpy(), dtype=torch.long)
    ids_ex, mask_ex = encode(tokenizer, make_text(exam), settings["max_len"])

    model = Classifier(make_encoder(), settings["dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["lr"])
    loss_fn = nn.CrossEntropyLoss()
    shuffle_gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(ids_tr, mask_tr, y_tr), batch_size=settings["train_batch"],
                        shuffle=True, generator=shuffle_gen)

    for epoch in range(settings["epochs"]):
        model.train()
        total_loss, correct, seen = 0.0, 0, 0
        t0 = time.time()
        for ids, mask, y in loader:
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            out = model(ids, mask)
            loss = loss_fn(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)
            correct += (out.argmax(dim=1) == y).sum().item()
            seen += y.size(0)
        log(f"      pass {epoch + 1:2d}/{settings['epochs']}  loss {total_loss / seen:.4f}  "
            f"study accuracy {100 * correct / seen:5.1f}%  ({time.time() - t0:.0f}s)")

    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(ids_ex), settings["eval_batch"]):
            out = model(ids_ex[i:i + settings["eval_batch"]].to(device),
                        mask_ex[i:i + settings["eval_batch"]].to(device))
            probs.append(torch.softmax(out.float(), dim=1)[:, 1].cpu())
    prob_gpu = torch.cat(probs).numpy()
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return (prob_gpu >= 0.5).astype(int), prob_gpu


def result_file(out_dir, model_key, machine, exam, split_seed, train_seed):
    return os.path.join(out_dir, f"preds__{model_key}__{machine}__{exam}__split{split_seed}__train{train_seed}.csv")


def run_experiment(model_key, machine, exam, data_dir, out_dir, split_seed=1, train_seed=1,
                   settings=None, tokenizer=None, make_encoder=None, device=None, log=print,
                   trainer=None, label=None):
    """One model, one machine, one exam, all rounds. Saves after every round,
    and if the file already has some rounds (e.g. Colab disconnected) it
    carries on from where it stopped. `trainer` (default train_and_predict) and
    `label` (default "<model_key>[+fast]") let other experiments reuse this loop."""
    trainer = trainer or train_and_predict
    settings = {**DEFAULT_SETTINGS, **(settings or {})}
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)

    if tokenizer is None or make_encoder is None:
        from transformers import AutoModel, AutoTokenizer
        hub_name = MODELS[model_key]
        tokenizer = AutoTokenizer.from_pretrained(hub_name)
        make_encoder = lambda hub_name=hub_name: AutoModel.from_pretrained(hub_name)   # name fixed now

    label = label or model_key + ("+fast" if settings.get("fast") else "")   # e.g. "codeberta+fast"
    df = devmap.load(machine, data_dir).set_index("row_id", drop=False)
    rounds = devmap.build_exam(df, exam, seed=split_seed)
    path = result_file(out_dir, label, machine, exam, split_seed, train_seed)
    done = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

    log(f"\n=== {label} | {machine} | exam '{exam}' | shuffle seed {split_seed} | training seed {train_seed} ===")
    # A saved round only counts if it holds exactly this round's exam rows. Rounds saved
    # with different piles (e.g. an older version of the splitting code) are redone.
    done_rounds = set()
    if len(done):
        wanted = {r["name"]: set(r["exam"]) for r in rounds}
        keep = []
        for round_name, part in done.groupby("round", sort=False):
            want = wanted.get(round_name)
            if want is not None and set(part["row_id"]) == want and len(part) == len(want):
                done_rounds.add(round_name)
                keep.append(part)
            else:
                log(f"   round {round_name}: saved with different exam rows, will redo it")
        if keep:
            done = pd.concat(keep, ignore_index=True)
            done.to_csv(path, index=False)
        else:
            done = pd.DataFrame()
            os.remove(path)
    for r in rounds:
        if r["name"] in done_rounds:
            log(f"   round {r['name']}: already done, skipping")
            continue
        study, ex = df.loc[r["study"]], df.loc[r["exam"]]
        log(f"   round {r['name']}: study {len(study)} rows, exam {len(ex)} rows")
        pred, prob = trainer(study, ex, tokenizer, make_encoder, device, settings, train_seed, log)
        part = pd.DataFrame({
            "model": label, "machine": machine, "exam": exam,
            "split_seed": split_seed, "train_seed": train_seed, "round": r["name"],
            "row_id": ex["row_id"].to_numpy(), "benchmark": ex["benchmark"].to_numpy(),
            "suite": ex["suite"].to_numpy(), "app": ex["app"].to_numpy(),
            "program_id": ex["program_id"].to_numpy(),
            "code_seen_in_study": ex["program_id"].isin(set(study["program_id"])).to_numpy(),
            "transfer": ex["transfer"].to_numpy(), "wgsize": ex["wgsize"].to_numpy(),
            "label": ex["label"].to_numpy(), "pred": pred, "prob_gpu": prob,
        })
        done = pd.concat([done, part], ignore_index=True)
        done.to_csv(path, index=False)
        log(f"   round {r['name']}: exam accuracy {100 * (part['label'] == part['pred']).mean():.1f}%")
    log(f"   overall accuracy {100 * (done['label'] == done['pred']).mean():.1f}%  -> saved {os.path.basename(path)}")
    return done


def load_predictions(out_dir):
    files = sorted(glob.glob(os.path.join(out_dir, "preds__*.csv")))
    if not files:
        return pd.DataFrame()
    preds = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    preds["correct"] = (preds["label"] == preds["pred"]).astype(int)
    preds["run"] = "split" + preds["split_seed"].astype(str) + "_train" + preds["train_seed"].astype(str)
    return preds


def summarise(out_dir):
    """Tables of everything finished so far. Returns a dict of DataFrames."""
    preds = load_predictions(out_dir)
    if preds.empty:
        return {}
    exam_order = {e: i for i, e in enumerate(devmap.EXAMS)}

    per_run = preds.groupby(["model", "machine", "exam", "run"])["correct"].mean().reset_index()
    main = per_run.groupby(["model", "machine", "exam"])["correct"].agg(
        accuracy="mean", spread="std", runs="count").reset_index()
    main["accuracy"] = (100 * main["accuracy"]).round(1)
    main["spread"] = (100 * main["spread"].fillna(0)).round(1)
    base = lambda m: m.split("+")[0]
    main["paper"] = [
        round(100 * float(np.mean(PAPER_OLD_EXAM[(base(m), k)])), 1)
        if e == "old" and (base(m), k) in PAPER_OLD_EXAM else np.nan
        for m, k, e in zip(main["model"], main["machine"], main["exam"])]
    main = main.sort_values(["model", "machine", "exam"], key=lambda c: c.map(exam_order) if c.name == "exam" else c)

    old = preds[preds["exam"] == "old"]
    piles = seen_new = pd.DataFrame()
    if len(old):
        piles = (100 * old.groupby(["model", "machine", "round"])["correct"].mean()).round(1).unstack()
        piles["average"] = piles.mean(axis=1).round(1)
        present = {(m.split("+")[0], k) for m, k in piles.index}
        paper_rows = {(m, k): [round(100 * v, 1) for v in vals] + [round(100 * float(np.mean(vals)), 1)]
                      for (m, k), vals in PAPER_OLD_EXAM.items() if (m, k) in present}
        for (m, k), vals in paper_rows.items():
            piles.loc[(m + " (paper)", k), :] = vals
        piles = piles.sort_index()
        seen_new = (100 * old.groupby(["model", "machine", "code_seen_in_study"])["correct"].mean()).round(1).unstack()
        seen_new = seen_new.rename(columns={False: "new code", True: "seen code"})

    suite = preds[preds["exam"] == "unseen_suite"]
    per_suite = pd.DataFrame()
    if len(suite):
        per_suite = (100 * suite.groupby(["model", "machine", "suite"])["correct"].mean()).round(1).unstack()

    return {"main": main, "old_exam_piles": piles, "old_exam_seen_vs_new": seen_new, "per_collection": per_suite}
