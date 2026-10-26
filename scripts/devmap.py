"""
Shared helpers: load the DevMap dataset and build the four "exams".

Words used here (same as in our chats):
  row      = one line of the dataset (code + two job numbers + answer)
  program  = one piece of OpenCL code (a "kernel"); several rows can share it
  study    = the rows a model learns from   (training set)
  exam     = the rows it is scored on       (test set)

  application = the benchmark a program was taken from (e.g. "npb-3.3-BT");
                one application contains several programs. 71 in total.
  collection  = the benchmark suite an application belongs to (NPB, Rodinia,
                ...). 7 in total.

The four exams, from easiest to hardest:
  old             - the base paper's exam: shuffle the ROWS, cut into 5 piles.
                    Rows of the same program can land on both sides.
  unseen_program  - shuffle the PROGRAMS, cut into 5 piles. Every row of a
                    program is on one side only.
  unseen_app      - shuffle the APPLICATIONS, cut into 5 piles. Needed because
                    programs of one application share the same job numbers,
                    so those numbers give the application away.
  unseen_suite    - hold out one whole collection at a time, 7 rounds.
"""
import os
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
FILES = {"amd": "dataset-devmap-amd.csv", "nvidia": "dataset-devmap-nvidia.csv"}

# The hand-counted features that ship with the dataset (Grewe et al.):
#   comp      = number of compute operations in the code
#   rational  = number of relational (comparison) operations
#   mem       = number of memory accesses
#   localmem  = number of accesses to local memory
#   coalesced = number of memory accesses that are neatly side by side
#   atomic    = number of atomic operations (always 0 here)
#   transfer  = how much data is sent to the device  (job number 1)
#   wgsize    = how the work is batched               (job number 2)
STATIC_FEATURES = ["comp", "rational", "mem", "localmem", "coalesced", "atomic"]
JOB_FEATURES = ["transfer", "wgsize"]

SUITES = ["amd-sdk", "npb", "nvidia-sdk", "parboil", "polybench", "rodinia", "shoc"]


def suite_of(benchmark_name):
    """Which collection a program comes from, read off its name."""
    b = benchmark_name.lower()
    if b.startswith("amd"):
        return "amd-sdk"
    if b.startswith("npb"):
        return "npb"
    if b.startswith("nvidia"):
        return "nvidia-sdk"
    if b.startswith("parboil"):
        return "parboil"
    if b.startswith("polybench"):
        return "polybench"
    if b.startswith("rodinia"):
        return "rodinia"
    if b.startswith("shoc"):
        return "shoc"
    raise ValueError("unknown collection for " + benchmark_name)


def app_of(benchmark_name):
    """'npb-3.3-BT-exact_rhs1' -> 'npb-3.3-BT' (collection, version, application)."""
    m = re.match(r"^(.*?-\d+(?:\.\d+)*)-([^-]+)-(.*)$", benchmark_name)
    if not m:
        raise ValueError("cannot read application from " + benchmark_name)
    return m.group(1) + "-" + m.group(2)


def code_key(src):
    """Two rows are 'the same program' when their code text is identical
    once extra spaces and line breaks are ignored."""
    return re.sub(r"\s+", " ", str(src)).strip()


def load(machine, data_dir=DATA_DIR):
    """Load one answer sheet ('amd' or 'nvidia') with a few helper columns."""
    df = pd.read_csv(os.path.join(data_dir, FILES[machine]), sep=",")
    df = df.loc[:, [c for c in df.columns if not c.startswith("Unnamed")]].copy()
    df["row_id"] = np.arange(len(df))           # stable id = original file order
    df["suite"] = df["benchmark"].map(suite_of)
    df["app"] = df["benchmark"].map(app_of)
    keys = df["src"].map(code_key)
    df["program_id"] = pd.factorize(keys)[0]     # same code text -> same id
    df["label"] = (df["oracle"] == "GPU").astype(int)   # 0 = CPU, 1 = GPU
    return df


# ----------------------------------------------------------------------------
# The four exams. Each returns a list of rounds; a round is a dict with
# 'name', 'study' (row_ids) and 'exam' (row_ids).
# ----------------------------------------------------------------------------
def old_exam(df):
    """Exactly what the base paper's notebooks do:
    df.sample(frac=1, random_state=1) and then KFold(5, shuffle=False)."""
    shuffled = df.sample(frac=1, random_state=1).reset_index(drop=True)
    rounds = []
    for i, (tr, te) in enumerate(KFold(n_splits=5, shuffle=False).split(shuffled)):
        rounds.append({
            "name": f"pile{i + 1}",
            "study": shuffled.iloc[tr]["row_id"].tolist(),
            "exam": shuffled.iloc[te]["row_id"].tolist(),
        })
    return rounds


def group_piles(groups, labels, n_piles=5, seed=1):
    """Deal whole groups (programs or applications) into n_piles piles.

    Our own version instead of scikit-learn's StratifiedGroupKFold, because
    that one gives different piles in different library versions (Colab's
    1.6 put 364 rows in one AMD pile and 15 in another). This uses only
    numpy's fixed random generator, so the piles are the same everywhere.

    How it deals: shuffle the groups (seed), then take them biggest first and
    put each one in the pile where it keeps the piles most even, both in
    size and in their share of GPU answers. Returns one pile number per row."""
    groups = np.asarray(groups)
    labels = np.asarray(labels)
    names, inverse = np.unique(groups, return_inverse=True)
    sizes = np.bincount(inverse)
    gpus = np.bincount(inverse, weights=labels)
    order = np.random.RandomState(seed).permutation(len(names))
    order = order[np.argsort(-sizes[order], kind="stable")]       # biggest first, ties in shuffled order
    target_rows = len(groups) / n_piles
    target_gpu = max(labels.sum() / n_piles, 1.0)
    pile_rows = np.zeros(n_piles)
    pile_gpu = np.zeros(n_piles)
    pile_of_group = np.empty(len(names), dtype=int)
    for g in order:
        costs = []
        for p in range(n_piles):
            rows, gpu = pile_rows.copy(), pile_gpu.copy()
            rows[p] += sizes[g]
            gpu[p] += gpus[g]
            costs.append((((rows - target_rows) / target_rows) ** 2).sum()
                         + (((gpu - target_gpu) / target_gpu) ** 2).sum())
        best = int(np.argmin(costs))                               # ties -> lowest pile number
        pile_of_group[g] = best
        pile_rows[best] += sizes[g]
        pile_gpu[best] += gpus[g]
    return pile_of_group[inverse]


def fair_exam_unseen_programs(df, seed=1):
    """5 rounds; all rows of a program stay on the same side. Piles are also
    kept roughly balanced in size and CPU/GPU answers."""
    pile = group_piles(df["program_id"], df["label"], 5, seed)
    rounds = []
    for i in range(5):
        tr, te = np.where(pile != i)[0], np.where(pile == i)[0]
        rounds.append({
            "name": f"pile{i + 1}",
            "study": df.iloc[tr]["row_id"].tolist(),
            "exam": df.iloc[te]["row_id"].tolist(),
        })
    return rounds


def fair_exam_unseen_apps(df, seed=1):
    """5 rounds; all programs of an application stay on the same side. If the
    exact same code also exists under another application, those study rows
    are dropped so the exam code is truly unseen."""
    pile = group_piles(df["app"], df["label"], 5, seed)
    rounds = []
    for i in range(5):
        tr, te = np.where(pile != i)[0], np.where(pile == i)[0]
        exam = df.iloc[te]
        study = df.iloc[tr]
        study = study[~study["program_id"].isin(exam["program_id"])]
        rounds.append({
            "name": f"pile{i + 1}",
            "study": study["row_id"].tolist(),
            "exam": exam["row_id"].tolist(),
        })
    return rounds


def fair_exam_unseen_collection(df):
    """7 rounds; each round holds out one whole collection. If a held-out
    program's exact code also exists in another collection, those study rows
    are dropped so the exam code is truly unseen."""
    rounds = []
    for suite in SUITES:
        exam = df[df["suite"] == suite]
        study = df[(df["suite"] != suite) & (~df["program_id"].isin(exam["program_id"]))]
        rounds.append({
            "name": suite,
            "study": study["row_id"].tolist(),
            "exam": exam["row_id"].tolist(),
        })
    return rounds


def build_exam(df, exam, seed=1):
    if exam == "old":
        return old_exam(df)
    if exam == "unseen_program":
        return fair_exam_unseen_programs(df, seed)
    if exam == "unseen_app":
        return fair_exam_unseen_apps(df, seed)
    if exam == "unseen_suite":
        return fair_exam_unseen_collection(df)
    raise ValueError(exam)


EXAMS = ["old", "unseen_program", "unseen_app", "unseen_suite"]
RESHUFFLED_EXAMS = ["unseen_program", "unseen_app"]   # these change with the shuffle seed


def leak_report(df, rounds):
    """For each round: how many exam rows have code that is also in the
    study rows (seen) vs. not (new)."""
    pid = df.set_index("row_id")["program_id"]
    out = []
    for r in rounds:
        study_programs = set(pid.loc[r["study"]])
        seen = sum(1 for x in r["exam"] if pid.loc[x] in study_programs)
        out.append({"round": r["name"], "study_rows": len(r["study"]),
                    "exam_rows": len(r["exam"]), "exam_seen_code": seen,
                    "exam_new_code": len(r["exam"]) - seen})
    return pd.DataFrame(out)
