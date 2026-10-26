"""
Row-by-row comparison of a Transformer run (from the Colab notebook) against
the baselines, on exactly the same exam piles.

  python 03_compare.py <folder with preds__*.csv from Colab> [model label]

For every machine x exam it reports:
  - accuracy of the Transformer and of each baseline on the same rows
  - McNemar's exact test: of the rows where the two disagree (one right, one
    wrong), is the split lopsided enough that it is unlikely to be luck?
  - a 95% range for the accuracy difference from an application-level
    bootstrap: re-draw whole applications with replacement 2000 times. Rows
    from one application are alike, so re-drawing single rows would make us
    look more certain than we are.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
BASELINES = ["forest", "job-numbers", "majority"]
EXAM_ORDER = ["old", "unseen_program", "unseen_app", "unseen_suite"]


def mcnemar_p(a_correct, b_correct):
    only_a = int(((a_correct == 1) & (b_correct == 0)).sum())
    only_b = int(((a_correct == 0) & (b_correct == 1)).sum())
    if only_a + only_b == 0:
        return 1.0, only_a, only_b
    return binomtest(only_a, only_a + only_b, 0.5).pvalue, only_a, only_b


def app_bootstrap(diff, apps, n=2000, seed=0):
    """95% range of mean(diff) when whole applications are re-drawn."""
    rng = np.random.RandomState(seed)
    groups = pd.Series(diff).groupby(np.asarray(apps))
    sums, counts = groups.sum().to_numpy(), groups.size().to_numpy()
    k = len(sums)
    stats = []
    for _ in range(n):
        pick = rng.randint(0, k, k)
        stats.append(sums[pick].sum() / counts[pick].sum())
    return np.percentile(stats, [2.5, 97.5])


def load_transformer(pred_dir, label, train_seed=1):
    files = sorted(glob.glob(os.path.join(pred_dir, f"preds__{label}__*__split1__train{train_seed}.csv")))
    if not files:
        return pd.DataFrame()
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d["correct"] = (d["label"] == d["pred"]).astype(int)
    return d


def main(pred_dir, model_label, other_models=()):
    tr = load_transformer(pred_dir, model_label)
    if not len(tr):
        sys.exit(f"no prediction files for {model_label} in {pred_dir}")
    base = pd.read_csv(os.path.join(RESULTS, "baseline_predictions.csv"))
    base = base[base["run"] == "split1_model1"]              # same piles as the Transformer run
    students = list(other_models) + BASELINES
    for other in other_models:                               # other Transformers become opponents too
        o = load_transformer(pred_dir, other)
        o["student"] = other
        base = pd.concat([base, o[["machine", "exam", "student", "round", "row_id", "correct"]]],
                         ignore_index=True)

    rows, per_app = [], []
    for (machine, exam), t in tr.groupby(["machine", "exam"]):
        t = t.set_index("row_id").sort_index()
        assert t.index.is_unique, (machine, exam)
        if len(t) < 680:
            print(f"(skipping {machine} {exam}: only {t['round'].nunique()} rounds finished)")
            continue
        for student in students:
            b = base[(base.machine == machine) & (base.exam == exam) & (base.student == student)]
            b = b.set_index("row_id").sort_index()
            if len(b) != 680:
                continue
            assert (b.index == t.index).all()
            assert (b["round"] == t["round"]).all(), f"piles differ: {machine} {exam} {student}"
            p, only_t, only_b = mcnemar_p(t["correct"].to_numpy(), b["correct"].to_numpy())
            lo, hi = app_bootstrap(t["correct"].to_numpy() - b["correct"].to_numpy(), t["app"].to_numpy())
            rows.append({
                "machine": machine, "exam": exam, "vs": student,
                model_label: round(100 * t["correct"].mean(), 1),
                "baseline": round(100 * b["correct"].mean(), 1),
                "difference": round(100 * (t["correct"].mean() - b["correct"].mean()), 1),
                "95% range": f"{100 * lo:+.1f} to {100 * hi:+.1f}",
                "only model right": only_t, "only baseline right": only_b,
                "McNemar p": round(p, 4),
            })
            if exam == "unseen_app" and student == "forest":
                g = pd.DataFrame({"app": t["app"], "suite": t["suite"], "model": t["correct"],
                                  "forest": b["correct"].to_numpy()}).groupby(["app", "suite"])
                a = g.agg(rows=("model", "size"), model=("model", "mean"), forest=("forest", "mean"))
                a.insert(0, "machine", machine)
                per_app.append(a.reset_index())

    out = pd.DataFrame(rows)
    out["exam"] = pd.Categorical(out["exam"], EXAM_ORDER, ordered=True)
    out = out.sort_values(["machine", "exam", "vs"]).reset_index(drop=True)
    pd.set_option("display.width", 250)
    print(out.to_string(index=False))
    out.to_csv(os.path.join(RESULTS, f"compare_{model_label}.csv"), index=False)

    if per_app:
        a = pd.concat(per_app)
        a["model"] = (100 * a["model"]).round(1)
        a["forest"] = (100 * a["forest"]).round(1)
        a.to_csv(os.path.join(RESULTS, f"unseen_app_per_app_{model_label}.csv"), index=False)
        print("\nUNSEEN APP - by collection (rows, accuracy %)")
        s = pd.concat(per_app)
        s = s.assign(m=s["model"] * s["rows"], f=s["forest"] * s["rows"]).groupby(["machine", "suite"]).agg(
            rows=("rows", "sum"), m=("m", "sum"), f=("f", "sum"))
        s[model_label] = (100 * s["m"] / s["rows"]).round(1)
        s["forest"] = (100 * s["f"] / s["rows"]).round(1)
        print(s[["rows", model_label, "forest"]].to_string())


if __name__ == "__main__":
    # usage: 03_compare.py <pred folder> [model label] [other Transformer labels to compare against ...]
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "codeberta+fast", sys.argv[3:])
