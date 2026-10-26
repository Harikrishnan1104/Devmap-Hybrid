"""
Significance tests for the hybrid model on the unseen-app and unseen-suite exams.

The hybrid variants (05_hybrid.py) and the Random Forests (04_name_tag_test.py)
were run on exactly the same 5 sets of piles (runs split{s}_model{s}), so every
exam row is answered once per run by both. For each pair:
  - difference in accuracy averaged over the 5 runs
  - 95% range from an application-level bootstrap on each row's average
    correctness over the runs
  - McNemar's exact test per run, and how many of the 5 runs give p < 0.05
Also compares the hybrid with the fine-tuned Transformers on the one set of
piles they share (split1), using the fine-tuned models' 3 training runs.

  python 07_hybrid_tests.py <folder with fine-tuned preds__*.csv>
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

import devmap

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
from importlib import import_module
rep = import_module("06_repeats")

PAIRS = [
    ("graphcodebert:emb+ratios", "forest"),
    ("graphcodebert:emb+ratios", "counts+ratios"),
    ("graphcodebert:emb+ratios", "graphcodebert:emb+raw"),
    ("graphcodebert:emb+ratios", "codeberta:emb+ratios"),
    ("graphcodebert:emb", "forest"),
    ("graphcodebert:emb+raw", "forest"),
]


def runs_of(preds, machine, exam, student):
    d = preds[(preds.machine == machine) & (preds.exam == exam) & (preds.student == student)]
    out = {}
    for run, g in d.groupby("run"):
        g = g.sort_values("row_id")
        out[run] = g["correct"].to_numpy()
    return out


def main(ft_dir):
    preds = pd.concat([pd.read_csv(os.path.join(RESULTS, "hybrid_predictions.csv")),
                       pd.read_csv(os.path.join(RESULTS, "name_tag_predictions.csv"))], ignore_index=True)
    if "correct" not in preds:
        preds["correct"] = (preds["label"] == preds["pred"]).astype(int)
    rows = []
    for machine in ["amd", "nvidia"]:
        apps = devmap.load(machine).sort_values("row_id")["app"].to_numpy()
        for exam in ["unseen_app", "unseen_suite"]:
            for a, b in PAIRS:
                ra, rb = runs_of(preds, machine, exam, a), runs_of(preds, machine, exam, b)
                common = sorted(set(ra) & set(rb))
                diff = np.mean([ra[r] - rb[r] for r in common], axis=0)
                lo, hi = rep.app_bootstrap(diff, apps)
                ps = [rep.mcnemar_p(ra[r], rb[r]) for r in common]
                rows.append({"machine": machine, "exam": exam, "model": a, "vs": b,
                             "model %": round(100 * np.mean([ra[r].mean() for r in common]), 1),
                             "vs %": round(100 * np.mean([rb[r].mean() for r in common]), 1),
                             "difference": round(100 * diff.mean(), 1),
                             "95% range": f"{100 * lo:+.1f} to {100 * hi:+.1f}",
                             "runs with p<0.05": f"{sum(p < 0.05 for p in ps)}/{len(ps)}"})
        # hybrid vs fine-tuned, on the shared piles (split1) of the unseen_app exam
        hyb = runs_of(preds, machine, "unseen_app", "graphcodebert:emb+ratios")["split1_model1"]
        for ft in ["graphcodebert+fast", "codeberta+fast"]:
            ft_runs = rep.transformer_runs(ft_dir, ft, machine)
            diff = hyb - np.mean(list(ft_runs.values()), axis=0)
            lo, hi = rep.app_bootstrap(diff, apps)
            ps = [rep.mcnemar_p(hyb, r) for r in ft_runs.values()]
            rows.append({"machine": machine, "exam": "unseen_app (piles 1)",
                         "model": "graphcodebert:emb+ratios", "vs": f"{ft} (fine-tuned)",
                         "model %": round(100 * hyb.mean(), 1),
                         "vs %": round(100 * np.mean([r.mean() for r in ft_runs.values()]), 1),
                         "difference": round(100 * diff.mean(), 1),
                         "95% range": f"{100 * lo:+.1f} to {100 * hi:+.1f}",
                         "runs with p<0.05": f"{sum(p < 0.05 for p in ps)}/{len(ps)}"})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS, "hybrid_tests.csv"), index=False)
    pd.set_option("display.width", 220)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
