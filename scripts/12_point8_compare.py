"""
Faculty point 8: is the Random Forest the best decision-maker?
Combines every learner tried (Phase 2: rf, xgboost, lightgbm, tabpfn; point 8:
logreg, svm, knn, mlp) on the unseen_app exam, 5 runs each, and tests every
learner against the Random Forest on the same inputs, row by row
(McNemar per run, application-level bootstrap of the mean difference).
Writes results/point8_table.csv and results/point8_tests.csv.
"""
import importlib, os
import numpy as np, pandas as pd
import devmap

rep = importlib.import_module("06_repeats")
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
files = ["decision_makers_predictions.csv", "decision_makers_point8_predictions.csv",
         "colab_phase2/decision_makers_tabpfn_predictions.csv"]
P = pd.concat([pd.read_csv(os.path.join(R, f)) for f in files if os.path.exists(os.path.join(R, f))])
P = P[P.exam == "unseen_app"].drop_duplicates(["machine", "student", "run", "row_id"])
P["correct"] = (P.label == P.pred).astype(int)
P["learner"], P["inputs"] = P.student.str.split(":").str[0], P.student.str.split(":").str[1]

acc = (100 * P.groupby(["inputs", "learner", "machine", "run"]).correct.mean()).groupby(["inputs", "learner", "machine"]).mean()
table = acc.unstack("machine").round(1)
table["mean"] = table.mean(axis=1).round(1)
table = table.sort_values(["inputs", "mean"], ascending=[True, False])
table.to_csv(os.path.join(R, "point8_table.csv"))

def runs(machine, student):
    d = P[(P.machine == machine) & (P.student == student)]
    return {r: g.sort_values("row_id").correct.to_numpy() for r, g in d.groupby("run")}

rows = []
for machine in ["amd", "nvidia"]:
    apps = devmap.load(machine).sort_values("row_id")["app"].to_numpy()
    for inp in ["hand", "hybrid"]:
        ref = runs(machine, f"rf:{inp}")
        for learner in sorted(P[P.inputs == inp].learner.unique()):
            if learner == "rf":
                continue
            other = runs(machine, f"{learner}:{inp}")
            common = sorted(set(ref) & set(other))
            diff = np.mean([other[r] - ref[r] for r in common], axis=0)
            lo, hi = rep.app_bootstrap(diff, apps)
            ps = [rep.mcnemar_p(other[r], ref[r]) for r in common]
            rows.append({"machine": machine, "inputs": inp, "learner": learner, "vs": "rf",
                         "difference": round(100 * diff.mean(), 1), "95% range": f"{100*lo:+.1f} to {100*hi:+.1f}",
                         "runs p<0.05": f"{sum(p < 0.05 for p in ps)}/{len(ps)}"})
tests = pd.DataFrame(rows)
tests.to_csv(os.path.join(R, "point8_tests.csv"), index=False)
if __name__ == "__main__":
    pd.set_option("display.width", 200)
    print(table.to_string()); print(); print(tests.to_string(index=False))
