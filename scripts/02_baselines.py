"""
Step 5 of the plan: the "quick students". No GPU needed.

  majority    - always answers whatever was most common in the study rows.
  memoriser   - if the exam row's code was in the study rows, repeats the
                answer seen most often for that code; otherwise falls back to
                the majority answer. Understands nothing about code.
  memoriser+  - same, but when the code was seen with several job sizes it
                copies the answer of the closest job size.
  job-numbers - a Random Forest that sees ONLY the two job numbers, no code.
  forest      - a Random Forest on the hand-counted code features that ship
                with the dataset, plus the two job numbers.

Every student sits all four exams on both machines. Results go to
results/baseline_summary.csv and results/baseline_predictions.csv.
"""
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import devmap

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
SEEDS = [1, 2, 3, 4, 5]


def forest_features(df):
    """Raw counts plus the four ratios Grewe et al. built from them."""
    X = df[devmap.STATIC_FEATURES + devmap.JOB_FEATURES].astype(float).copy()
    mem = X["mem"].replace(0, np.nan)
    X["F1_transfer_per_op"] = X["transfer"] / (X["comp"] + X["mem"]).replace(0, np.nan)
    X["F2_coalesced_share"] = X["coalesced"] / mem
    X["F3_localmem_x_wgsize"] = (X["localmem"] / mem) * X["wgsize"]
    X["F4_comp_per_mem"] = X["comp"] / mem
    return X.fillna(0.0)


def predict_majority(study, exam, seed):
    return np.full(len(exam), int(study["label"].mean() >= 0.5))


def predict_memoriser(study, exam, seed):
    fallback = int(study["label"].mean() >= 0.5)
    by_program = study.groupby("program_id")["label"].mean()
    out = []
    for pid in exam["program_id"]:
        if pid in by_program.index and by_program[pid] != 0.5:
            out.append(int(by_program[pid] > 0.5))
        else:
            out.append(fallback)
    return np.array(out)


def predict_memoriser_plus(study, exam, seed):
    fallback = int(study["label"].mean() >= 0.5)
    groups = {pid: g for pid, g in study.groupby("program_id")}
    out = []
    for _, row in exam.iterrows():
        g = groups.get(row["program_id"])
        if g is None:
            out.append(fallback)
            continue
        dist = (np.abs(np.log1p(g["transfer"]) - np.log1p(row["transfer"]))
                + np.abs(np.log1p(g["wgsize"]) - np.log1p(row["wgsize"])))
        out.append(int(g.loc[dist.idxmin(), "label"]))
    return np.array(out)


def predict_job_numbers(study, exam, seed):
    rf = RandomForestClassifier(n_estimators=300, random_state=seed)
    rf.fit(study[devmap.JOB_FEATURES].astype(float), study["label"])
    return rf.predict(exam[devmap.JOB_FEATURES].astype(float))


def predict_forest(study, exam, seed):
    rf = RandomForestClassifier(n_estimators=300, random_state=seed)
    rf.fit(forest_features(study), study["label"])
    return rf.predict(forest_features(exam))


STUDENTS = {
    "majority": (predict_majority, False),       # (function, depends on seed?)
    "memoriser": (predict_memoriser, False),
    "memoriser+": (predict_memoriser_plus, False),
    "job-numbers": (predict_job_numbers, True),
    "forest": (predict_forest, True),
}


def main():
    os.makedirs(RESULTS, exist_ok=True)
    all_preds = []
    for machine in ["amd", "nvidia"]:
        df = devmap.load(machine).set_index("row_id", drop=False)
        for exam in devmap.EXAMS:
            # 'old' and 'unseen_suite' have one fixed split; the others are reshuffled per seed
            reshuffled = exam in devmap.RESHUFFLED_EXAMS
            split_seeds = SEEDS if reshuffled else [1]
            for split_seed in split_seeds:
                rounds = devmap.build_exam(df, exam, seed=split_seed)
                for student, (fn, seeded) in STUDENTS.items():
                    model_seeds = SEEDS if (seeded and not reshuffled) else [split_seed]
                    for model_seed in model_seeds:
                        for r in rounds:
                            study, ex = df.loc[r["study"]], df.loc[r["exam"]]
                            pred = fn(study, ex, model_seed)
                            seen = ex["program_id"].isin(set(study["program_id"])).to_numpy()
                            all_preds.append(pd.DataFrame({
                                "machine": machine, "exam": exam, "student": student,
                                "run": f"split{split_seed}_model{model_seed}",
                                "round": r["name"], "row_id": ex["row_id"].to_numpy(),
                                "suite": ex["suite"].to_numpy(),
                                "code_seen_in_study": seen,
                                "label": ex["label"].to_numpy(), "pred": pred,
                            }))
    preds = pd.concat(all_preds, ignore_index=True)
    preds["correct"] = (preds["label"] == preds["pred"]).astype(int)
    preds.to_csv(os.path.join(RESULTS, "baseline_predictions.csv"), index=False)

    # one accuracy per run (all exam rows pooled), then mean and spread over runs
    per_run = preds.groupby(["machine", "exam", "student", "run"])["correct"].mean().reset_index()
    summary = per_run.groupby(["machine", "exam", "student"])["correct"].agg(
        accuracy="mean", spread="std", runs="count").reset_index()
    summary["spread"] = summary["spread"].fillna(0.0)
    summary.to_csv(os.path.join(RESULTS, "baseline_summary.csv"), index=False)

    pd.set_option("display.width", 160)
    for value, title in [("accuracy", "Accuracy (%), average over runs"),
                         ("spread", "Spread between runs (standard deviation, % points)")]:
        table = summary.pivot_table(index="student", columns=["machine", "exam"], values=value)
        table = table.loc[list(STUDENTS)] * 100
        table = table.reindex(columns=[(m, e) for m in ["amd", "nvidia"] for e in devmap.EXAMS])
        print("\n" + title)
        print(table.round(1).to_string())

    print("\nOld exam only: accuracy (%) on exam rows whose code was SEEN in study vs NEW")
    old = preds[preds["exam"] == "old"]
    t7 = old.groupby(["machine", "student", "code_seen_in_study"])["correct"].mean().unstack() * 100
    t7.columns = ["new code", "seen code"]
    print(t7.round(1).to_string())

    print("\nUnseen-collection exam: accuracy (%) per held-out collection")
    f2 = preds[preds["exam"] == "unseen_suite"]
    per_suite = f2.groupby(["machine", "student", "suite"])["correct"].mean().unstack() * 100
    print(per_suite.round(1).to_string())


if __name__ == "__main__":
    main()
