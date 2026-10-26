"""
The "name tag" test. No GPU needed.

Idea: the two raw job numbers (transfer, wgsize) are nearly unique to each
application, so a model can use them to recognise WHICH application it is
looking at instead of learning WHY it runs faster on the CPU or the GPU.
If that is true, the raw numbers should be very useful on exams where the
application was seen before, and much less useful on exams with new
applications. Models given scale-free ratios instead should hold up better.

Random Forest variants (feature sets defined in features.py), all on exactly
the same piles as every other model, 5 runs each:
  forest          - counts + raw job numbers + Grewe's four  (original baseline)
  counts          - the six code counts only, no job information at all
  counts+ratios   - code counts + scale-free ratios, NO raw job numbers
  grewe4          - only the four features Grewe et al. (2013) designed
  job-numbers     - only the two raw job numbers

Writes results/name_tag_summary.csv and results/name_tag_predictions.csv.
"""
import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import devmap
import features as F

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
SEEDS = [1, 2, 3, 4, 5]

FEATURE_SETS = {
    "forest": F.forest,
    "counts": F.counts,
    "counts+ratios": lambda d: pd.concat([F.counts(d), F.ratios(d)], axis=1),
    "grewe4": F.grewe4,
    "job-numbers": F.job_numbers,
}


def main(seeds=SEEDS, results=RESULTS):
    rows = []
    for machine in ["amd", "nvidia"]:
        df = devmap.load(machine).set_index("row_id", drop=False)
        for exam in devmap.EXAMS:
            reshuffled = exam in devmap.RESHUFFLED_EXAMS
            for seed in seeds:                      # reshuffled exams: new piles AND new forest per seed
                rounds = devmap.build_exam(df, exam, seed=seed if reshuffled else 1)
                for name, feats in FEATURE_SETS.items():
                    for r in rounds:
                        study, ex = df.loc[r["study"]], df.loc[r["exam"]]
                        rf = RandomForestClassifier(n_estimators=300, random_state=seed)
                        rf.fit(feats(study), study["label"])
                        rows.append(pd.DataFrame({
                            "machine": machine, "exam": exam, "student": name,
                            "run": f"split{seed if reshuffled else 1}_model{seed}", "round": r["name"],
                            "row_id": ex["row_id"].to_numpy(), "app": ex["app"].to_numpy(),
                            "label": ex["label"].to_numpy(), "pred": rf.predict(feats(ex)),
                        }))
    preds = pd.concat(rows, ignore_index=True)
    preds["correct"] = (preds["label"] == preds["pred"]).astype(int)
    os.makedirs(results, exist_ok=True)
    preds.to_csv(os.path.join(results, "name_tag_predictions.csv"), index=False)

    per_run = preds.groupby(["machine", "exam", "student", "run"])["correct"].mean().reset_index()
    summary = per_run.groupby(["machine", "exam", "student"])["correct"].agg(
        accuracy="mean", spread="std", runs="count").reset_index()
    summary.to_csv(os.path.join(results, "name_tag_summary.csv"), index=False)

    pd.set_option("display.width", 200)
    cols = [(m, e) for m in ["amd", "nvidia"] for e in devmap.EXAMS]
    for value, title in [("accuracy", "Accuracy (%), mean over runs"),
                         ("spread", "Spread between runs (std, % points)")]:
        t = summary.pivot_table(index="student", columns=["machine", "exam"], values=value) * 100
        print("\n" + title)
        print(t.loc[list(FEATURE_SETS)].reindex(columns=cols).round(1).to_string())
    return summary


if __name__ == "__main__":
    main()
