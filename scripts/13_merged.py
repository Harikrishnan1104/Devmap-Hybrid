"""
Faculty point 7b: merge the AMD and NVIDIA tables into one dataset (1,360 rows).

The two machines give different answers for 37.5% of identical rows, so the
merged table gets a "which GPU" column (nvidia = 0 for AMD, 1 for NVIDIA).
Piles are dealt by application over the merged table, so both copies of a
kernel are always on the same side (no leak between the machines).

On the SAME piles, three ways are compared (Random Forest, 300 trees):
  separate      one model per machine, as in all our other results
  merged        one model for both machines, with the "which GPU" column
  merged-noflag one model for both machines, without that column
Inputs: hand (counts + job numbers + Grewe's four) and hybrid (GraphCodeBERT
embedding squeezed to 16 numbers + counts + ratios). Exams: unseen_app (5 shuffle
seeds) and the paper's random 5-fold ("old") for contrast.
"""
import importlib, os, sys
import numpy as np, pandas as pd
from sklearn.model_selection import KFold

import devmap, embed_lib

dm = importlib.import_module("08_decision_makers")
rep = importlib.import_module("06_repeats")
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def merged_table(data_dir=devmap.DATA_DIR):
    parts = []
    for flag, m in enumerate(["amd", "nvidia"]):
        d = devmap.load(m, data_dir).copy()
        d["base_row"], d["machine"], d["nvidia"] = d["row_id"], m, flag
        d["row_id"] = d["row_id"] + 1000 * flag
        parts.append(d)
    return pd.concat(parts, ignore_index=True).set_index("row_id", drop=False)


def rounds_for(df, exam, seed):
    if exam == "old":
        sh = df.sample(frac=1, random_state=1)
        return [{"name": f"pile{i+1}", "study": sh.iloc[tr].index.tolist(), "exam": sh.iloc[te].index.tolist()}
                for i, (tr, te) in enumerate(KFold(5).split(sh))]
    pile = devmap.group_piles(df["app"], df["label"], 5, seed)
    out = []
    for i in range(5):
        ex, st = df[pile == i], df[pile != i]
        st = st[~st["program_id"].isin(ex["program_id"])]
        out.append({"name": f"pile{i+1}", "study": st.index.tolist(), "exam": ex.index.tolist()})
    return out


def main(emb_dir, seeds=(1, 2, 3, 4, 5), kinds=("hand", "hybrid"), log=print):
    vectors = embed_lib.load_embeddings(os.path.join(emb_dir, "embeddings__graphcodebert.npz"))
    df = merged_table()
    rows = []
    for exam in ["unseen_app", "old"]:
        for seed in (seeds if exam == "unseen_app" else [1]):
            for r in rounds_for(df, exam, seed):
                study, ex = df.loc[r["study"]], df.loc[r["exam"]]
                for kind in kinds:
                    Xs, Xe = dm.inputs(study, ex, kind, vectors, seed)
                    preds = {}
                    for name, cols_s, cols_e, fit_rows in [
                        ("merged", Xs.assign(nvidia=study["nvidia"]), Xe.assign(nvidia=ex["nvidia"]), None),
                        ("merged-noflag", Xs, Xe, None)]:
                        m = dm.make_learner("rf", seed).fit(cols_s.to_numpy(float), study["label"].to_numpy())
                        preds[name] = m.predict(cols_e.to_numpy(float))
                    sep = np.zeros(len(ex), dtype=int)
                    for flag in [0, 1]:
                        s, e = (study["nvidia"] == flag).to_numpy(), (ex["nvidia"] == flag).to_numpy()
                        Xs1, Xe1 = dm.inputs(study[s], ex[e], kind, vectors, seed)   # squeeze refit per machine
                        m = dm.make_learner("rf", seed).fit(Xs1.to_numpy(float), study["label"].to_numpy()[s])
                        sep[e] = m.predict(Xe1.to_numpy(float))
                    preds["separate"] = sep
                    for name, p in preds.items():
                        rows.append(pd.DataFrame({"exam": exam, "run": f"split{seed}", "round": r["name"],
                                                  "inputs": kind, "way": name, "machine": ex["machine"].to_numpy(),
                                                  "row_id": ex["base_row"].to_numpy(), "app": ex["app"].to_numpy(),
                                                  "label": ex["label"].to_numpy(), "pred": np.asarray(p).astype(int)}))
            log(f"  {exam} seed {seed}: done")
    P = pd.concat(rows, ignore_index=True)
    P["correct"] = (P.label == P.pred).astype(int)
    P.to_csv(os.path.join(R, "merged_predictions.csv"), index=False)

    acc = (100 * P.groupby(["exam", "inputs", "way", "machine", "run"]).correct.mean()) \
        .groupby(["exam", "inputs", "way", "machine"]).mean().unstack("machine").round(1)
    acc["mean"] = acc.mean(axis=1).round(1)
    acc.to_csv(os.path.join(R, "merged_summary.csv"))

    tests = []
    U = P[P.exam == "unseen_app"]
    for machine in ["amd", "nvidia"]:
        apps = devmap.load(machine).sort_values("row_id")["app"].to_numpy()
        for kind in kinds:
            get = lambda w: {r: g.sort_values("row_id").correct.to_numpy() for r, g in
                             U[(U.machine == machine) & (U.inputs == kind) & (U.way == w)].groupby("run")}
            ref = get("separate")
            for w in ["merged", "merged-noflag"]:
                o = get(w)
                diff = np.mean([o[k] - ref[k] for k in ref], axis=0)
                lo, hi = rep.app_bootstrap(diff, apps)
                ps = [rep.mcnemar_p(o[k], ref[k]) for k in ref]
                tests.append({"machine": machine, "inputs": kind, "way": w, "vs": "separate",
                              "difference": round(100 * diff.mean(), 1), "95% range": f"{100*lo:+.1f} to {100*hi:+.1f}",
                              "runs p<0.05": f"{sum(p < 0.05 for p in ps)}/{len(ps)}"})
    tests = pd.DataFrame(tests)
    tests.to_csv(os.path.join(R, "merged_tests.csv"), index=False)
    pd.set_option("display.width", 200)
    log(acc.to_string()); log(""); log(tests.to_string(index=False))
    return acc, tests


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(R, "colab_all", "embeddings"))
