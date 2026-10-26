"""
Phase 2: does a newer "decision-maker" beat the Random Forest?

The final CPU/GPU decision in our best models is made by a Random Forest. Here
the same inputs are given to newer learners built for small tables:
  rf        Random Forest, 300 trees                     (our original choice)
  xgboost   gradient-boosted trees (XGBoost)
  lightgbm  gradient-boosted trees (LightGBM)
  logreg    logistic regression (a straight-line rule)      \
  svm       support vector machine, RBF kernel              | faculty point 8: simple
  knn       k-nearest neighbours, k = 5                     | classic alternatives, all on
  mlp       small neural network, 2 hidden layers (64, 32)  / signed-log + standardised inputs
  tabpfn    TabPFN v2, a pretrained Transformer for small tables (Hollmann et al.,
            Nature 2025). Its weights download from Hugging Face, which this
            workspace cannot reach, so it runs on Colab.

Inputs (all leakage-safe, as in 04/05):
  hand      counts + raw runtime parameters + Grewe's four  (= the "forest" baseline)
  ratios    counts + scale-free ratios
  hybrid    GraphCodeBERT embedding squeezed to 16 numbers (PCA fitted on study
            kernels only) + counts + ratios                (= our best hybrid)

Every learner's settings were fixed before any results were seen (no tuning).
Same exams, same 5 sets of piles, as every other model.

  python 08_decision_makers.py <folder with embeddings__graphcodebert.npz> [--learners rf xgboost lightgbm tabpfn]
"""
import argparse
import importlib
import os

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
hybrid = importlib.import_module("05_hybrid")


def _scaled(model):
    """Counts and sizes span many orders of magnitude; distance- and weight-based
    learners need them on a log scale and standardised (fitted on study rows only)."""
    from sklearn.preprocessing import FunctionTransformer
    signed_log = FunctionTransformer(lambda X: np.sign(X) * np.log1p(np.abs(X)))
    return make_pipeline(signed_log, StandardScaler(), model)


def make_learner(name, seed):
    if name == "logreg":
        from sklearn.linear_model import LogisticRegression
        return _scaled(LogisticRegression(C=1.0, max_iter=5000))
    if name == "svm":
        from sklearn.svm import SVC
        return _scaled(SVC(C=1.0, kernel="rbf", gamma="scale", random_state=seed))
    if name == "knn":
        from sklearn.neighbors import KNeighborsClassifier
        return _scaled(KNeighborsClassifier(n_neighbors=5))
    if name == "mlp":
        from sklearn.neural_network import MLPClassifier
        return _scaled(MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=2000, random_state=seed))
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, random_state=seed, n_jobs=-1, verbosity=0)
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=300, num_leaves=15, learning_rate=0.05, min_child_samples=10,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                              random_state=seed, n_jobs=-1, verbose=-1)
    if name == "tabpfn":
        # TabPFN version 2, the model published in Nature (Hollmann et al., 2025). Its
        # weights are a free download (Prior Labs licence: Apache 2.0 + attribution);
        # newer versions need a gated Hugging Face sign-up.
        import torch
        from tabpfn import TabPFNClassifier
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:                                    # newer tabpfn packages: ask for the v2 model explicitly
            from tabpfn.constants import ModelVersion
        except ImportError:                     # tabpfn 2.0.x: v2 is the default model
            return TabPFNClassifier(device=device, random_state=seed)
        return TabPFNClassifier.create_default_for_version(ModelVersion.V2, device=device, random_state=seed)
    raise ValueError(name)


def inputs(study, exam, kind, vectors, seed):
    """(study table, exam table) for one round."""
    if kind == "hand":
        return F.forest(study), F.forest(exam)
    ratios = lambda d: pd.concat([F.counts(d), F.ratios(d)], axis=1)
    if kind == "ratios":
        return ratios(study), ratios(exam)
    if kind == "hybrid":
        s_ids, s_vec = hybrid.embedding_matrix(study, vectors)
        _, e_vec = hybrid.embedding_matrix(exam, vectors)
        _, first = np.unique(s_ids, return_index=True)        # squeeze fitted on study kernels only
        dims = min(PCA_DIMS, len(first) - 1)
        squeeze = make_pipeline(StandardScaler(), PCA(n_components=dims, random_state=seed)).fit(s_vec[first])
        cols = [f"emb{i}" for i in range(dims)]
        Xs = pd.concat([pd.DataFrame(squeeze.transform(s_vec), index=study.index, columns=cols), ratios(study)], axis=1)
        Xe = pd.concat([pd.DataFrame(squeeze.transform(e_vec), index=exam.index, columns=cols), ratios(exam)], axis=1)
        return Xs, Xe
    raise ValueError(kind)


def main(emb_dir, learners=("rf", "xgboost", "lightgbm"), kinds=("hand", "ratios", "hybrid"),
         seeds=SEEDS, results=RESULTS, tag="", log=print, data_dir=devmap.DATA_DIR):
    vectors = embed_lib.load_embeddings(os.path.join(emb_dir, "embeddings__graphcodebert.npz"))
    rows = []
    for machine in ["amd", "nvidia"]:
        df = devmap.load(machine, data_dir).set_index("row_id", drop=False)
        for exam in devmap.EXAMS:
            reshuffled = exam in devmap.RESHUFFLED_EXAMS
            for seed in seeds:
                rounds = devmap.build_exam(df, exam, seed=seed if reshuffled else 1)
                for r in rounds:
                    study, ex = df.loc[r["study"]], df.loc[r["exam"]]
                    for kind in kinds:
                        Xs, Xe = inputs(study, ex, kind, vectors, seed)
                        for name in learners:
                            model = make_learner(name, seed)
                            model.fit(Xs.to_numpy(dtype=float), study["label"].to_numpy())
                            pred = np.asarray(model.predict(Xe.to_numpy(dtype=float))).astype(int)
                            rows.append(pd.DataFrame({
                                "machine": machine, "exam": exam, "student": f"{name}:{kind}",
                                "run": f"split{seed if reshuffled else 1}_model{seed}", "round": r["name"],
                                "row_id": ex["row_id"].to_numpy(), "app": ex["app"].to_numpy(),
                                "label": ex["label"].to_numpy(), "pred": pred,
                            }))
            log(f"   {machine} {exam}: done")
    preds = pd.concat(rows, ignore_index=True)
    preds["correct"] = (preds["label"] == preds["pred"]).astype(int)
    os.makedirs(results, exist_ok=True)
    preds.to_csv(os.path.join(results, f"decision_makers{tag}_predictions.csv"), index=False)
    per_run = preds.groupby(["machine", "exam", "student", "run"])["correct"].mean().reset_index()
    summary = per_run.groupby(["machine", "exam", "student"])["correct"].agg(
        accuracy="mean", spread="std", runs="count").reset_index()
    summary["spread"] = summary["spread"].fillna(0.0)
    summary.to_csv(os.path.join(results, f"decision_makers{tag}_summary.csv"), index=False)
    pd.set_option("display.width", 220)
    cols = [(m, e) for m in ["amd", "nvidia"] for e in devmap.EXAMS]
    for value, title in [("accuracy", "Accuracy (%), mean over runs"), ("spread", "Spread between runs (std)")]:
        t = summary.pivot_table(index="student", columns=["machine", "exam"], values=value) * 100
        print("\n" + title)
        print(t.reindex(columns=cols).round(1).to_string())
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("emb_dir")
    ap.add_argument("--learners", nargs="*", default=["rf", "xgboost", "lightgbm"])
    ap.add_argument("--kinds", nargs="*", default=["hand", "ratios", "hybrid"])
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    main(a.emb_dir, learners=a.learners, kinds=a.kinds, tag=a.tag)
