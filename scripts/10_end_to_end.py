"""
Faculty question 9: "Why not let the Transformer make the decision itself
(tweak its last layer) instead of adding a Random Forest?"

Our hybrid = frozen GraphCodeBERT reading the code + counts + ratios, decided
by a Random Forest. Here the SAME inputs go into ONE neural network, trained
end to end, and its own last layer makes the decision:

    code text -> GraphCodeBERT -> first-token vector -> Linear -> ReLU --+
                                                                        |-> join -> Dropout -> Linear -> CPU/GPU
    counts + ratios (log, standardised) -> Linear(., 32) -> ReLU -------+

  * The text is the code only. The numbers go in as numbers, not as text.
  * Same training recipe as our fine-tuned GraphCodeBERT (Adam, 10 passes,
    batch 4, learning rate 1e-5, fast mode). The two new number layers start
    from scratch, so they get a larger learning rate (1e-3); everything else
    keeps 1e-5.
  * Same exam (unseen_app, shuffle seed 1) and same piles, 3 training seeds,
    so it lines up row by row with graphcodebert+fast (72.5 / 64.5) and with
    the hybrid (75.8 / 72.1).
Results are saved as preds__graphcodebert+e2e__<machine>__unseen_app__split1__train<seed>.csv
in the same format as all other fine-tuned runs, and resume after a disconnect.
"""
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

import devmap
import features as F
import train_lib

LABEL = "graphcodebert+e2e"
NEW_LAYER_LR = 1e-3


def number_inputs(df):
    """The hybrid's extra inputs: six code counts + five scale-free ratios."""
    X = pd.concat([F.counts(df), F.ratios(df)], axis=1).astype(float)
    return np.log1p(X.clip(lower=0).to_numpy())


class NumberClassifier(nn.Module):
    def __init__(self, encoder, n_numbers, dropout=0.3, number_width=32):
        super().__init__()
        self.l1 = encoder
        hidden = encoder.config.hidden_size
        self.pre_classifier = nn.Linear(hidden, hidden)
        self.numbers = nn.Sequential(nn.Linear(n_numbers, number_width), nn.ReLU())
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden + number_width, 2)

    def new_layers(self):
        return list(self.numbers.parameters()) + list(self.classifier.parameters())

    def forward(self, input_ids, attention_mask, numbers):
        h = self.l1(input_ids=input_ids, attention_mask=attention_mask)[0][:, 0]
        h = torch.relu(self.pre_classifier(h))
        joined = torch.cat([h, self.numbers(numbers)], dim=1)
        return self.classifier(self.dropout(joined))


def train_and_predict_e2e(study, exam, tokenizer, make_encoder, device, settings, seed, log=print):
    """Like train_lib.train_and_predict_fast, plus the number branch."""
    train_lib.set_seed(seed)
    pad_id = tokenizer.pad_token_id if getattr(tokenizer, "pad_token_id", None) is not None else 0
    code = lambda d: [" ".join(str(s).split()) for s in d["src"]]
    tr_seqs = tokenizer(code(study), truncation=True, max_length=settings["max_len"])["input_ids"]
    ex_seqs = tokenizer(code(exam), truncation=True, max_length=settings["max_len"])["input_ids"]
    n_tr, n_ex = number_inputs(study), number_inputs(exam)
    mu, sd = n_tr.mean(axis=0), n_tr.std(axis=0) + 1e-6          # scaled with study rows only
    n_tr = torch.tensor((n_tr - mu) / sd, dtype=torch.float32)
    n_ex = torch.tensor((n_ex - mu) / sd, dtype=torch.float32)
    y_tr = study["label"].to_numpy()

    use_amp = str(device).startswith("cuda")
    model = NumberClassifier(make_encoder(), n_tr.shape[1], settings["dropout"]).to(device)
    new_ids = {id(p) for p in model.new_layers()}
    old = [p for p in model.parameters() if id(p) not in new_ids]
    optimizer = torch.optim.Adam([{"params": old, "lr": settings["lr"]},
                                  {"params": model.new_layers(), "lr": settings.get("new_layer_lr", NEW_LAYER_LR)}])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = nn.CrossEntropyLoss()
    order_gen = torch.Generator().manual_seed(seed)
    bs = settings["train_batch"]

    for epoch in range(settings["epochs"]):
        model.train()
        total_loss, correct, seen, t0 = 0.0, 0, 0, time.time()
        order = torch.randperm(len(tr_seqs), generator=order_gen).tolist()
        for start in range(0, len(order), bs):
            idx = order[start:start + bs]
            ids, mask = train_lib._pad([tr_seqs[i] for i in idx], pad_id)
            y = torch.tensor(y_tr[idx], dtype=torch.long).to(device)
            nums = n_tr[idx].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                out = model(ids.to(device), mask.to(device), nums)
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
    by_length = sorted(range(len(ex_seqs)), key=lambda i: len(ex_seqs[i]))
    prob_gpu = np.zeros(len(ex_seqs))
    with torch.no_grad():
        for start in range(0, len(by_length), settings["eval_batch"]):
            idx = by_length[start:start + settings["eval_batch"]]
            ids, mask = train_lib._pad([ex_seqs[i] for i in idx], pad_id)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                out = model(ids.to(device), mask.to(device), n_ex[idx].to(device))
            prob_gpu[idx] = torch.softmax(out.float(), dim=1)[:, 1].cpu().numpy()
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return (prob_gpu >= 0.5).astype(int), prob_gpu


def run(machines=("amd", "nvidia"), seeds=(1, 2, 3), data_dir=devmap.DATA_DIR, out_dir="results",
        settings=None, tokenizer=None, make_encoder=None, device=None, log=print):
    settings = {**train_lib.DEFAULT_SETTINGS, "fast": True, **(settings or {})}
    for seed in seeds:                       # seed 1 for both machines first, then the repeats
        for machine in machines:
            train_lib.run_experiment("graphcodebert", machine, "unseen_app", data_dir, out_dir,
                                     split_seed=1, train_seed=seed, settings=settings,
                                     tokenizer=tokenizer, make_encoder=make_encoder, device=device,
                                     log=log, trainer=train_and_predict_e2e, label=LABEL)


def summary(out_dir, hybrid_csv=None):
    """Accuracy per run, next to fine-tuned GraphCodeBERT if its files are in out_dir."""
    preds = train_lib.load_predictions(out_dir)
    preds = preds[(preds["exam"] == "unseen_app") & (preds["split_seed"] == 1)
                  & preds["model"].isin([LABEL, "graphcodebert+fast", "codeberta+fast"])]
    per_run = (100 * preds.groupby(["model", "machine", "train_seed"])["correct"].mean()).round(1)
    table = per_run.groupby(["model", "machine"]).agg(["mean", "count"]).round(1)
    table.columns = ["accuracy (mean of runs)", "runs"]
    return table.reset_index()
