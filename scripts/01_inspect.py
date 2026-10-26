"""Step 2 of the plan: look at the dataset before building on it."""
import pandas as pd

import devmap

pd.set_option("display.width", 160)

for machine in ["amd", "nvidia"]:
    df = devmap.load(machine)
    print(f"\n================ {machine.upper()} ================")
    print("rows:", len(df), "| program names:", df["benchmark"].nunique(),
          "| distinct code texts:", df["program_id"].nunique(),
          "| applications:", df["app"].nunique())
    print("answers:", df["oracle"].value_counts().to_dict())

    per_suite = df.groupby("suite").agg(
        rows=("row_id", "size"),
        names=("benchmark", "nunique"),
        codes=("program_id", "nunique"),
        gpu_share=("label", "mean"),
    )
    print(per_suite.round(2))

    rows_per_program = df.groupby("program_id").size()
    print("programs with exactly 1 row:", int((rows_per_program == 1).sum()),
          "| rows in programs with 2+ rows:", int(rows_per_program[rows_per_program > 1].sum()))

    answers_per_program = df.groupby("program_id")["label"].nunique()
    multi = rows_per_program[rows_per_program > 1].index
    always_same = int((answers_per_program.loc[multi] == 1).sum())
    print("programs with 2+ rows:", len(multi), "| of these, answer never changes:", always_same,
          "| answer changes with job size:", len(multi) - always_same)

    shared = df.groupby("program_id")["benchmark"].nunique()
    shared = shared[shared > 1]
    print("code texts used under more than one program name:", len(shared))
    for pid in shared.index:
        sub = df[df["program_id"] == pid]
        print("   ", sorted(sub["benchmark"].unique()), "suites:", sorted(sub["suite"].unique()))

    for exam in devmap.EXAMS:
        rep = devmap.leak_report(df, devmap.build_exam(df, exam))
        print(f"\n-- exam '{exam}' --")
        print(rep.to_string(index=False))
        print("   total exam rows:", int(rep["exam_rows"].sum()),
              "| seen code:", int(rep["exam_seen_code"].sum()),
              "| new code:", int(rep["exam_new_code"].sum()))
