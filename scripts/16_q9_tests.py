"""Faculty question 9: graphcodebert+e2e (numbers as numbers, decides by itself) vs the others,
row by row on the same unseen_app piles (split 1). Writes results/q9_tests.csv."""
import glob, importlib, os
import numpy as np, pandas as pd
import devmap
rep = importlib.import_module("06_repeats")
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

def ft_runs(folder, label, m):
    fs = sorted(glob.glob(os.path.join(R, folder, f"preds__{label}__{m}__unseen_app__split1__train*.csv")))
    return [pd.read_csv(f).sort_values("row_id").reset_index(drop=True) for f in fs]

def rows_runs(csv, student, m, run="split1_model1"):
    d = pd.read_csv(os.path.join(R, csv)); d = d[(d.exam == "unseen_app") & (d.student == student) & (d.machine == m) & (d.run == run)]
    return d.sort_values("row_id").reset_index(drop=True)

def speed(m, preds):
    df = devmap.load(m).set_index("row_id"); best = np.minimum(df.runtime_cpu, df.runtime_gpu)
    static = df.runtime_cpu if m == "amd" else df.runtime_gpu
    out = []
    for p in preds:
        p = p.set_index("row_id").pred.reindex(df.index)
        ch = np.where(p == 1, df.runtime_gpu, df.runtime_cpu)
        out.append((np.exp(np.mean(np.log(best / ch))), np.exp(np.mean(np.log(static / ch)))))
    return np.mean(out, axis=0)

rows = []
for m in ["amd", "nvidia"]:
    e2e = ft_runs("colab_q9", "graphcodebert+e2e", m)
    gcb = ft_runs("colab_all", "graphcodebert+fast", m)
    hyb = rows_runs("hybrid_predictions.csv", "graphcodebert:emb+ratios", m)
    rf = rows_runs("baseline_predictions.csv", "forest", m)
    apps = e2e[0].app.values; non = ~pd.Series(apps).str.startswith("npb").values
    E = np.mean([(d.label == d.pred).values for d in e2e], axis=0)
    others = {"graphcodebert fine-tuned (numbers as text)": np.mean([(d.label == d.pred).values for d in gcb], axis=0),
              "hybrid (GCB + RF)": hyb.correct.values.astype(float), "Random Forest": rf.correct.values.astype(float)}
    for name, O in others.items():
        diff = E - O; lo, hi = rep.app_bootstrap(diff, apps)
        o_runs = gcb if name.startswith("graphcodebert") else [hyb if name.startswith("hybrid") else rf]
        o_runs = [(d.label == d.pred).values.astype(int) for d in o_runs]
        ps = [rep.mcnemar_p((e.label == e.pred).values.astype(int), o) for e in e2e for o in o_runs[:1]] if len(o_runs) == 1 else \
             [rep.mcnemar_p((e.label == e.pred).values.astype(int), o) for e, o in zip(e2e, o_runs)]
        rows.append({"machine": m, "vs": name, "e2e %": round(100 * E.mean(), 1), "other %": round(100 * O.mean(), 1),
                     "difference": round(100 * diff.mean(), 1), "95% range": f"{100*lo:+.1f} to {100*hi:+.1f}",
                     "runs p<0.05": f"{sum(p < 0.05 for p in ps)}/{len(ps)}",
                     "e2e NPB %": round(100 * E[~non].mean(), 1), "e2e non-NPB %": round(100 * E[non].mean(), 1)})
    sp = speed(m, e2e)
    print(f"{m}: e2e per run {[round(100*(d.label==d.pred).mean(),1) for d in e2e]}; share of best speed {sp[0]:.3f}; "
          f"speed-up over {'always-CPU' if m=='amd' else 'always-GPU'} {sp[1]:.3f}")
out = pd.DataFrame(rows); out.to_csv(os.path.join(R, "q9_tests.csv"), index=False)
pd.set_option("display.width", 250); print(out.to_string(index=False))
