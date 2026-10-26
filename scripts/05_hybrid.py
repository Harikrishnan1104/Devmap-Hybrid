"""
The hybrid model: a pretrained code model's understanding of the code,
plus hand-made features, fed to a Random Forest. No GPU needed here; the
code embeddings come from colab/extract_embeddings.ipynb.

  python 05_hybrid.py <folder with embeddings__*.npz> [models ...]

Why a hybrid: the fine-tuned Transformers read the raw job numbers as text
next to the code, where they can act as a name tag for the application.
Here the code model only ever sees the code, and the job size reaches the
classifier either as raw numbers (name-tag prone) or as scale-free ratios.

Variants, each with every embedding model, on exactly the same piles as all
other models, 5 runs each:
  emb             code embedding only
  emb+counts      + the six code counts                       (no job info)
  emb+ratios      + counts + scale-free ratios               OUR HYBRID
  emb+raw         + counts + raw job numbers + Grewe's four  (name-tag prone)

No leakage: in every round the embedding is squeezed to 16 numbers with a
PCA that is fitted on the STUDY rows' kernels only, and the forest is trained
on study rows only. The pretrained code model itself never saw DevMap labels.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import devmap
import embed_lib
import features as F

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
SEEDS = [1, 2, 3, 4, 5]
PCA_DIMS = 16

EXTRAS = {
    "emb": lambda d: pd.DataFrame(index=d.index),
    "emb+counts": F.counts,
    "emb+ratios": lambda d: pd.concat([F.counts(d), F.ratios(d)], axis=1),
    "emb+raw": F.forest,
}


def embedding_matrix(df, vectors):
    ids = embed_lib.code_ids(df)
    missing = [i for i in set(ids) if i not in vectors]
    if missing:
        raise KeyError(f"{len(missing)} kernels have no embedding - wrong embeddings file?")
    return ids, np.stack([vectors[i] for i in ids])


def predict_round(study, exam, vectors, extra, seed):
    s_ids, s_vec = embedding_matrix(study, vectors)
    _, e_vec = embedding_matrix(exam, vectors)
    # fit the squeeze on each distinct STUDY kernel once, so popular kernels do not dominate
    _, first = np.unique(s_ids, return_index=True)
    dims = min(PCA_DIMS, len(first) - 1)
    squeeze = make_pipeline(StandardScaler(), PCA(n_components=dims, random_state=seed))
    squeeze.fit(s_vec[first])
    cols = [f"emb{i}" for i in range(dims)]
    Xs = pd.concat([pd.DataFrame(squeeze.transform(s_vec), index=study.index, columns=cols),
                    extra(study)], axis=1)
    Xe = pd.concat([pd.DataFrame(squeeze.transform(e_vec), index=exam.index, columns=cols),
                    extra(exam)], axis=1)
    rf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    rf.fit(Xs, study["label"])
    return rf.predict(Xe)


def available_models(emb_dir):
    return sorted(f[len("embeddings__"):-len(".npz")] for f in os.listdir(emb_dir)
                  if f.startswith("embeddings__") and f.endswith(".npz"))


def main(emb_dir, models=None, seeds=SEEDS, results=RESULTS, pooling="mean",
         data_dir=devmap.DATA_DIR, log=print, variants=None, tag=""):
    """models=None -> every embeddings__*.npz in emb_dir. variants=None -> all of
    EXTRAS. tag is added to the output file names (hybrid<tag>_*.csv) so a later
    run never overwrites an earlier one."""
    models = models or available_models(emb_dir)
    extras = {k: v for k, v in EXTRAS.items() if variants is None or k in variants}
    rows = []
    for model in models:
        path = os.path.join(emb_dir, f"embeddings__{model}.npz")
        if not os.path.exists(path):
            log(f"(no embeddings for {model}, skipping)")
            continue
        vectors = embed_lib.load_embeddings(path, pooling)
        for machine in ["amd", "nvidia"]:
            df = devmap.load(machine, data_dir).set_index("row_id", drop=False)
            for exam in devmap.EXAMS:
                reshuffled = exam in devmap.RESHUFFLED_EXAMS
                for seed in seeds:
                    rounds = devmap.build_exam(df, exam, seed=seed if reshuffled else 1)
                    for variant, extra in extras.items():
                        for r in rounds:
                            study, ex = df.loc[r["study"]], df.loc[r["exam"]]
                            rows.append(pd.DataFrame({
                                "machine": machine, "exam": exam, "student": f"{model}:{variant}",
                                "run": f"split{seed if reshuffled else 1}_model{seed}", "round": r["name"],
                                "row_id": ex["row_id"].to_numpy(), "app": ex["app"].to_numpy(),
                                "label": ex["label"].to_numpy(),
                                "pred": predict_round(study, ex, vectors, extra, seed),
                            }))
            log(f"   {model} {machine}: done")
    preds = pd.concat(rows, ignore_index=True)
    preds["correct"] = (preds["label"] == preds["pred"]).astype(int)
    os.makedirs(results, exist_ok=True)
    preds.to_csv(os.path.join(results, f"hybrid{tag}_predictions.csv"), index=False)

    per_run = preds.groupby(["machine", "exam", "student", "run"])["correct"].mean().reset_index()
    summary = per_run.groupby(["machine", "exam", "student"])["correct"].agg(
        accuracy="mean", spread="std", runs="count").reset_index()
    summary["spread"] = summary["spread"].fillna(0.0)
    ref = os.path.join(results, "name_tag_summary.csv")       # Random Forest references, same piles
    if os.path.exists(ref):
        r = pd.read_csv(ref)
        summary = pd.concat([summary, r[r["student"].isin(["forest", "counts+ratios"])]],
                            ignore_index=True)
    summary.to_csv(os.path.join(results, f"hybrid{tag}_summary.csv"), index=False)

    pd.set_option("display.width", 220)
    cols = [(m, e) for m in ["amd", "nvidia"] for e in devmap.EXAMS]
    t = (summary.pivot_table(index="student", columns=["machine", "exam"], values="accuracy") * 100)
    print("\nAccuracy (%), mean over runs")
    print(t.reindex(columns=cols).round(1).to_string())
    s = (summary.pivot_table(index="student", columns=["machine", "exam"], values="spread") * 100)
    print("\nSpread between runs (std, % points)")
    print(s.reindex(columns=cols).round(1).to_string())
    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("emb_dir")
    ap.add_argument("--models", nargs="*", help="default: every embeddings file in emb_dir")
    ap.add_argument("--variants", nargs="*", choices=list(EXTRAS), help="default: all")
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. _phase1")
    a = ap.parse_args()
    main(a.emb_dir, models=a.models, variants=a.variants, tag=a.tag)
