"""
Offline checks for the Phase 3 junior + senior system. Gemini cannot be reached
from here, so scripted fake LLMs stand in for it.
"""
import importlib
import json
import os
import re
import shutil
import tempfile

import numpy as np
import pandas as pd

import agent_lib as A
import devmap

HERE = os.path.dirname(os.path.abspath(__file__))
EMB = os.path.join(HERE, "..", "results", "colab_all", "embeddings", "embeddings__graphcodebert.npz")
ev = importlib.import_module("09_agent_eval")
quiet = lambda *a, **k: None


class FakeLLM:
    """Agent prompts: calls fact_sheet, then similar_kernels, then answers by a
    simple rule on the fact sheet. Other prompts: answers by the same rule if the
    fact sheet is present, otherwise says GPU."""

    def __init__(self):
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        usage = {"prompt_tokens": len(prompt) // 4, "output_tokens": 30}
        m = re.search(r"Arithmetic per global memory access: ([0-9.]+)", prompt)
        rule = "GPU" if (m and float(m.group(1)) > 1.0) else "CPU"
        if "to call a tool" in prompt:
            if "You called fact_sheet" not in prompt:
                return json.dumps({"action": "tool", "tool": "fact_sheet"}), usage
            if "You called similar_kernels" not in prompt:
                return json.dumps({"action": "tool", "tool": "similar_kernels"}), usage
            return json.dumps({"action": "answer", "device": rule, "confidence": 0.7, "reason": "rule"}), usage
        return "Sure! " + json.dumps({"device": rule if m else "GPU", "confidence": 0.6, "reason": "x"}), usage


def check_junior_and_tools():
    df = devmap.load("amd").set_index("row_id", drop=False)
    rounds = devmap.build_exam(df, A.EXAM, seed=A.SPLIT_SEED)
    j = A.junior_predictions(df, rounds)
    assert len(j) == 680 and j["row_id"].nunique() == 680
    hard = j[j["hard"]]
    assert 100 < len(hard) < 300, len(hard)
    sim = A.SimilarKernels(A.embed_lib.load_embeddings(EMB))
    exam_apps = {r["name"]: set(df.loc[r["exam"], "app"]) for r in rounds}
    for _, jr in hard.head(40).iterrows():
        row = df.loc[int(jr["row_id"])]
        study = df.loc[next(r["study"] for r in rounds if r["name"] == jr["round"])]
        text = sim(row, study)
        names = re.findall(r"^- (\S+) \(similarity", text, flags=re.M)
        assert len(names) == 3
        assert not ({devmap.app_of(n) for n in names} & exam_apps[jr["round"]]), "retrieved an exam application!"
    fs = A.fact_sheet(df.loc[int(hard.iloc[0]["row_id"])])
    assert "Arithmetic per global memory access" in fs and "Work-group size" in fs
    print(f"junior: {len(hard)} hard rows on AMD; similar-kernel search never returns the exam application: ok")


def check_parsing():
    assert A.parse_answer(A.parse_json('blah {"device": "gpu", "confidence": "0.8", "reason": "r"} blah'))["device"] == "GPU"
    assert A.parse_answer(A.parse_json("no json here")) is None
    assert A.parse_answer({"device": "TPU"}) is None
    print("reply parsing: ok")


def check_variants_and_fallback():
    df = devmap.load("nvidia").set_index("row_id", drop=False)
    rounds = devmap.build_exam(df, A.EXAM, seed=A.SPLIT_SEED)
    j = A.junior_predictions(df, rounds)
    jr = j[j["hard"]].iloc[0]
    row = df.loc[int(jr["row_id"])]
    study = df.loc[next(r["study"] for r in rounds if r["name"] == jr["round"])]
    sim = A.SimilarKernels(A.embed_lib.load_embeddings(EMB))
    llm = FakeLLM()
    agent = A.run_case(llm, "agent", "nvidia", row, jr, study, sim)
    assert agent["llm_ok"] and agent["tools_used"] == "fact_sheet+similar_kernels" and agent["calls"] == 3
    assert "You called similar_kernels. Result:" in llm.prompts[-1] and "Most similar kernels" in llm.prompts[-1]
    for v in ["code", "facts", "all_tools"]:
        r = A.run_case(FakeLLM(), v, "nvidia", row, jr, study, sim)
        assert r["llm_ok"] and r["calls"] == 1, (v, r)
    code_prompt = FakeLLM()
    A.run_case(code_prompt, "code", "nvidia", row, jr, study, sim)
    assert "Arithmetic per global memory access" not in code_prompt.prompts[0] and "GTX 970" in code_prompt.prompts[0]
    garbage = lambda p: ("I think it's the GPU, probably.", {})
    r = A.run_case(garbage, "facts", "nvidia", row, jr, study, sim)
    assert not r["llm_ok"] and r["device"] == ("GPU" if jr["junior_pred"] == 1 else "CPU") and r["calls"] == 2
    looping = lambda p: (json.dumps({"action": "tool", "tool": "fact_sheet"}), {})
    r = A.run_case(looping, "agent", "nvidia", row, jr, study, sim)
    assert not r["llm_ok"] and r["calls"] <= 4
    print("all four variants, tool use, and fallback to the junior on bad replies: ok")


def check_run_resume_quota_and_eval():
    out = tempfile.mkdtemp()
    try:
        A.run_all(FakeLLM(), "fake", devmap.DATA_DIR, EMB, out, limit=10, log=quiet)
        d = pd.read_csv(os.path.join(out, "senior_answers.csv"))
        assert len(d) == 2 * 4 * 10
        # quota hit part-way: run stops cleanly, saved rows stay
        n = {"k": 0}

        def flaky(prompt):
            n["k"] += 1
            if n["k"] > 25:
                raise A.QuotaExhausted("daily limit")
            return FakeLLM()(prompt)
        A.run_all(flaky, "fake", devmap.DATA_DIR, EMB, out, log=quiet)
        d = pd.read_csv(os.path.join(out, "senior_answers.csv"))
        assert not d.duplicated(["machine", "row_id", "variant"]).any()
        partial = len(d)
        # resume to the end
        A.run_all(FakeLLM(), "fake", devmap.DATA_DIR, EMB, out, log=quiet)
        d = pd.read_csv(os.path.join(out, "senior_answers.csv"))
        hard = sum(int(pd.read_csv(os.path.join(out, f"junior_{m}.csv"))["hard"].sum()) for m in ["amd", "nvidia"])
        assert len(d) == hard * 4 and not d.duplicated(["machine", "row_id", "variant"]).any()
        print(f"run: stops cleanly at a quota limit ({partial} answers saved), resumes without repeats, "
              f"finishes all {hard} hard rows x 4 variants: ok")
        res = ev.main(out, log=quiet)
        assert set(res["variant"]) == {"junior only", "code", "facts", "all_tools", "agent"}
        assert res["overall accuracy"].notna().all()
        print("evaluation table: ok")
        print(res[["machine", "variant", "hard-row accuracy", "overall accuracy"]].to_string(index=False))
    finally:
        shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    check_parsing()
    check_junior_and_tools()
    check_variants_and_fallback()
    check_run_resume_quota_and_eval()
    print("ALL AGENT CHECKS PASSED")
