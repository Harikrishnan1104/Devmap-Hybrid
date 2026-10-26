"""
Repeat-run analysis for the most important exam (unseen_app).

  python 06_repeats.py <folder with preds__*.csv>

Every Transformer was trained 3 times (training seeds 1-3) on the SAME piles
(shuffle seed 1). For a fair comparison the Random Forest is also trained 5
times on those same piles here (forest seeds 1-5).

For each pair of models it reports:
  - mean accuracy and spread (std) over runs
  - the 95% range of the accuracy difference from an application-level
    bootstrap, using each row's average correctness over that model's runs
  - McNemar's exact test for each matching pair of runs (run k vs run k)
Writes results/repeats_summary.csv and results/repeats_pairs.csv.
"""
import glob
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestClassifier

import devmap
import features as F

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
EXAM = "unseen_app"


def mcnemar_p(a, b):
    only_a, only_b = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    return 1.0 if only_a + only_b == 0 else binomtest(only_a, only_a + only_b, 0.5).pvalue


def app_bootstrap(diff, apps, n=2000, seed=0):
    rng = np.random.RandomState(seed)
    g = pd.Series(diff).groupby(np.asarray(apps))
    sums, counts = g.sum().to_numpy(), g.size().to_numpy()
    k = len(sums)
    stats = [sums[p].sum() / counts[p].sum() for p in (rng.randint(0, k, k) for _ in range(n))]
    return np.percentile(stats, [2.5, 97.5])


def transformer_runs(pred_dir, label, machine):
    """{run name: correctness per row (sorted by row_id)}"""
    runs = {}
    for f in sorted(glob.glob(os.path.join(pred_dir, f"preds__{label}__{machine}__{EXAM}__split1__train*.csv"))):
        d = pd.read_csv(f).sort_values("row_id")
        if len(d) == 680:
            runs["train" + f.split("__train")[-1].split(".")[0]] = (d["label"] == d["pred"]).astype(int).to_numpy()
    return runs


def baseline_runs(machine, name, seeds=(1, 2, 3, 4, 5)):
    df = devmap.load(machine).set_index("row_id", drop=False)
    rounds = devmap.build_exam(df, EXAM, seed=1)
    feats = {"forest": F.forest, "job-numbers": F.job_numbers}[name] if name != "majority" else None
    runs = {}
    for s in (seeds if feats is not None else (1,)):
        correct = pd.Series(0, index=df.index)
        for r in rounds:
            study, ex = df.loc[r["study"]], df.loc[r["exam"]]
            if feats is None:
                pred = np.full(len(ex), int(study["label"].mean() >= 0.5))
            else:
                rf = RandomForestClassifier(n_estimators=300, random_state=s, n_jobs=-1)
                rf.fit(feats(study), study["label"])
                pred = rf.predict(feats(ex))
            correct.loc[ex.index] = (pred == ex["label"].to_numpy()).astype(int)
        runs[f"seed{s}"] = correct.sort_index().to_numpy()
    return runs


def main(pred_dir, models=("graphcodebert+fast", "codeberta+fast")):
    summary, pairs = [], []
    for machine in ["amd", "nvidia"]:
        apps = devmap.load(machine).sort_values("row_id")["app"].to_numpy()
        students = {m: transformer_runs(pred_dir, m, machine) for m in models}
        for b in ["forest", "job-numbers", "majority"]:
            students[b] = baseline_runs(machine, b)
        for name, runs in students.items():
            accs = [100 * r.mean() for r in runs.values()]
            summary.append({"machine": machine, "model": name, "runs": len(accs),
                            "mean": round(np.mean(accs), 1), "spread": round(np.std(accs, ddof=1), 1) if len(accs) > 1 else 0.0,
                            "each run": ", ".join(f"{a:.1f}" for a in accs)})
        for a, b in itertools.combinations(students, 2):
            if a not in models:
                continue
            ra, rb = students[a], students[b]
            diff = np.mean(list(ra.values()), axis=0) - np.mean(list(rb.values()), axis=0)
            lo, hi = app_bootstrap(diff, apps)
            ps = [mcnemar_p(x, y) for x, y in zip(ra.values(), itertools.cycle(rb.values()))]
            pairs.append({"machine": machine, "model": a, "vs": b,
                          "difference": round(100 * diff.mean(), 1),
                          "95% range": f"{100 * lo:+.1f} to {100 * hi:+.1f}",
                          "McNemar p per run": ", ".join(f"{p:.3f}" for p in ps),
                          "runs with p<0.05": f"{sum(p < 0.05 for p in ps)}/{len(ps)}"})
    s, p = pd.DataFrame(summary), pd.DataFrame(pairs)
    s.to_csv(os.path.join(RESULTS, "repeats_summary.csv"), index=False)
    p.to_csv(os.path.join(RESULTS, "repeats_pairs.csv"), index=False)
    pd.set_option("display.width", 220)
    print(f"EXAM: {EXAM}, all models on the same piles\n")
    print(s.to_string(index=False))
    print()
    print(p.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
