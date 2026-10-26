"""
Phase 3: a "junior + senior" device-mapping system.

  junior  - a Random Forest on the hand-counted features (fast, cheap). It answers
            every exam row and says how sure it is.
  senior  - a large language model (LLM). It only looks at the rows where the
            junior is less than THRESHOLD sure, and may use three tools:
              fact_sheet      the kernel's counts and job sizes in plain words,
                              plus derived ratios (work per memory access, ...)
              junior_opinion  the Random Forest's answer and how sure it is
              similar_kernels the 3 most similar kernels from the STUDY rows
                              (never the exam application), with their measured
                              answers at the nearest job size
The senior never runs anything on hardware.

Four senior variants, all scored on the same hard rows:
  code        the kernel code and job sizes only (no tools)
  facts       + the fact sheet
  all_tools   + fact sheet + junior opinion + similar kernels, all given at once
  agent       the LLM decides which tools to call (up to 3), one at a time, then answers

Final system answer = junior on rows it is sure about + senior on the hard rows.
If the senior fails (bad reply, error), the junior's answer is kept and the
failure is recorded.

Every LLM reply is saved to a CSV as soon as it arrives, so an interrupted run
(Colab disconnect, free-tier daily limit) resumes where it stopped.
"""
import csv
import json
import os
import re
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import devmap
import embed_lib
import features as F

THRESHOLD = 0.7           # junior hands a row to the senior when max(p, 1-p) < THRESHOLD
JUNIOR_SEED = 1
EXAM, SPLIT_SEED = "unseen_app", 1
VARIANTS = ["all_tools", "agent", "facts", "code"]     # run order: most important first
MAX_TOOL_CALLS = 3
CODE_CHARS = 6000          # longer kernels are shown truncated (about 2,000 tokens)

PLATFORMS = {
    "amd": "CPU: Intel Core i7-3820 (4 cores, 3.6 GHz). GPU: AMD Radeon HD 7970 'Tahiti' (discrete, PCIe).",
    "nvidia": "CPU: Intel Core i7-3820 (4 cores, 3.6 GHz). GPU: NVIDIA GeForce GTX 970 (discrete, PCIe).",
}

TASK = """You decide whether an OpenCL kernel runs faster on the CPU or on the GPU of this machine.
{platform}
Each case is one kernel launched with a given job size:
- transfer: bytes copied between CPU memory and GPU memory for this launch
- work-group size: number of work-items per OpenCL work-group
The correct answer was measured: whichever device actually ran this launch faster.
Typical considerations: data-transfer cost over PCIe, amount of parallel work, arithmetic per
memory access, coalesced versus scattered memory access, local-memory use, branching."""

ANSWER_FORMAT = """Reply with ONE JSON object and nothing else:
{"device": "CPU" or "GPU", "confidence": number between 0 and 1, "reason": "at most two sentences citing the evidence you used"}"""

AGENT_FORMAT = """You may use tools before answering. Each tool takes no arguments and is about THIS case:
- fact_sheet: the kernel's operation counts, job sizes and derived ratios in plain words
- junior_opinion: what a Random Forest trained on measured cases predicts, and how sure it is
- similar_kernels: the 3 most similar kernels with measured answers (from other applications)
Reply with ONE JSON object and nothing else, either
  to call a tool:  {"action": "tool", "tool": "fact_sheet" | "junior_opinion" | "similar_kernels"}
  to answer:       {"action": "answer", "device": "CPU" or "GPU", "confidence": number between 0 and 1, "reason": "at most two sentences citing the evidence you used"}
Each tool can be called once; after {max_calls} tool calls you must answer."""


# ----------------------------------------------------------------------------
# junior
# ----------------------------------------------------------------------------
def junior_predictions(df, rounds, seed=JUNIOR_SEED):
    """Random Forest on the hand features, trained per round on study rows only.
    Returns one row per exam row with p_gpu, prediction and confidence."""
    out = []
    for r in rounds:
        study, ex = df.loc[r["study"]], df.loc[r["exam"]]
        rf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
        rf.fit(F.forest(study), study["label"])
        p = rf.predict_proba(F.forest(ex))[:, 1]
        out.append(pd.DataFrame({"round": r["name"], "row_id": ex["row_id"].to_numpy(),
                                 "label": ex["label"].to_numpy(), "p_gpu": p}))
    j = pd.concat(out, ignore_index=True)
    j["junior_pred"] = (j["p_gpu"] >= 0.5).astype(int)
    j["junior_conf"] = np.maximum(j["p_gpu"], 1 - j["p_gpu"])
    j["hard"] = j["junior_conf"] < THRESHOLD
    return j


# ----------------------------------------------------------------------------
# tools
# ----------------------------------------------------------------------------
def _bytes(n):
    n = float(n)
    for unit in ["bytes", "KB", "MB", "GB"]:
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "bytes" else f"{n:.1f} {unit}"
        n /= 1024


def fact_sheet(row):
    ops = row["comp"] + row["mem"]
    ratio = lambda a, b: (a / b) if b else 0.0
    return "\n".join([
        f"Arithmetic operations: {int(row['comp'])}; comparisons: {int(row['rational'])}",
        f"Global memory accesses: {int(row['mem'])}; local-memory accesses: {int(row['localmem'])}; atomic operations: {int(row['atomic'])}",
        f"Coalesced memory accesses: {int(row['coalesced'])} (counted separately by the static analyser, so it can exceed the global count)",
        f"Arithmetic per global memory access: {ratio(row['comp'], row['mem']):.2f}",
        f"Share of memory accesses to local memory: {100 * ratio(row['localmem'], row['mem'] + row['localmem']):.0f}%",
        f"Data transferred for this launch: {_bytes(row['transfer'])} ({int(row['transfer'])} bytes)",
        f"Work-group size: {int(row['wgsize'])}",
        f"Bytes transferred per counted operation: {ratio(row['transfer'], ops):.1f}",
    ])


def junior_opinion(jrow, n_study):
    dev = "GPU" if jrow["junior_pred"] == 1 else "CPU"
    return (f"A Random Forest trained on {n_study} measured cases from other applications predicts {dev}, "
            f"with probability {jrow['junior_conf']:.2f} (it is only sure above {THRESHOLD:.2f}).")


class SimilarKernels:
    """Nearest study kernels by GraphCodeBERT embedding (cosine similarity).
    Only study rows are searched, so the exam application can never be retrieved."""

    def __init__(self, vectors, k=3):
        self.vectors, self.k = vectors, k

    def __call__(self, row, study):
        q = self.vectors[embed_lib.code_id(row["src"])]
        cand = study.assign(cid=embed_lib.code_ids(study))
        ids = cand["cid"].unique()
        mat = np.stack([self.vectors[i] for i in ids])
        sims = mat @ q / (np.linalg.norm(mat, axis=1) * np.linalg.norm(q) + 1e-12)
        lines = []
        for i in np.argsort(-sims)[: self.k]:
            rows = cand[cand["cid"] == ids[i]]
            near = rows.iloc[np.argsort(np.abs(np.log1p(rows["transfer"]) - np.log1p(row["transfer"])))[:2]]
            cases = "; ".join(
                f"transfer {_bytes(n['transfer'])}, work-group {int(n['wgsize'])}: "
                f"{'GPU' if n['label'] == 1 else 'CPU'} faster "
                f"(CPU/GPU time ratio {n['runtime_cpu'] / n['runtime_gpu']:.2f})"
                for _, n in near.iterrows())
            lines.append(f"- {rows.iloc[0]['benchmark']} (similarity {sims[i]:.2f}): {cases}")
        return "Most similar kernels from other applications, with measured results:\n" + "\n".join(lines)


def kernel_block(row):
    code = devmap.code_key(row["src"])
    if len(code) > CODE_CHARS:
        code = code[:CODE_CHARS] + " ... [truncated]"
    return (f"Kernel code:\n{code}\n\n"
            f"Job size for this launch: transfer {_bytes(row['transfer'])} ({int(row['transfer'])} bytes), "
            f"work-group size {int(row['wgsize'])}.")


# ----------------------------------------------------------------------------
# prompts, parsing
# ----------------------------------------------------------------------------
def base_prompt(machine, row):
    return TASK.format(platform=PLATFORMS[machine]) + "\n\n" + kernel_block(row)


def parse_json(text):
    """First JSON object in the reply, or None."""
    if not text:
        return None
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def parse_answer(obj):
    if not isinstance(obj, dict):
        return None
    dev = str(obj.get("device", "")).strip().upper()
    if dev not in ("CPU", "GPU"):
        return None
    try:
        conf = float(obj.get("confidence", np.nan))
    except (TypeError, ValueError):
        conf = np.nan
    return {"device": dev, "confidence": conf, "reason": str(obj.get("reason", ""))[:500]}


# ----------------------------------------------------------------------------
# one case
# ----------------------------------------------------------------------------
def run_case(llm, variant, machine, row, jrow, study, similar):
    """Returns a dict with the answer, the tools used, tokens and any failure."""
    tools = {
        "fact_sheet": lambda: fact_sheet(row),
        "junior_opinion": lambda: junior_opinion(jrow, len(study)),
        "similar_kernels": lambda: similar(row, study),
    }
    usage = {"calls": 0, "prompt_tokens": 0, "output_tokens": 0}

    def ask(prompt):
        text, u = llm(prompt)
        usage["calls"] += 1
        usage["prompt_tokens"] += int(u.get("prompt_tokens", 0) or 0)
        usage["output_tokens"] += int(u.get("output_tokens", 0) or 0)
        return text

    base = base_prompt(machine, row)
    used = []
    if variant != "agent":
        extra = {"code": [], "facts": ["fact_sheet"],
                 "all_tools": ["fact_sheet", "junior_opinion", "similar_kernels"]}[variant]
        parts = [base] + [f"{name.replace('_', ' ').capitalize()}:\n{tools[name]()}" for name in extra]
        prompt = "\n\n".join(parts + [ANSWER_FORMAT])
        ans = parse_answer(parse_json(ask(prompt)))
        if ans is None:                                  # one retry with a reminder
            ans = parse_answer(parse_json(ask(prompt + "\nYour previous reply was not valid JSON. Reply with the JSON object only.")))
        used = extra
    else:
        transcript = [base, AGENT_FORMAT.replace("{max_calls}", str(MAX_TOOL_CALLS))]
        ans, bad = None, 0
        while ans is None:
            obj = parse_json(ask("\n\n".join(transcript)))
            if isinstance(obj, dict) and obj.get("action") == "tool" and obj.get("tool") in tools \
                    and obj.get("tool") not in used and len(used) < MAX_TOOL_CALLS:
                name = obj["tool"]
                used.append(name)
                transcript.append(f"You called {name}. Result:\n{tools[name]()}")
                continue
            ans = parse_answer(obj)
            if ans is None:
                bad += 1
                if bad > 1:
                    break
                transcript.append("That reply was not valid (unknown or repeated tool, or bad JSON). "
                                  + ("You must answer now." if len(used) >= MAX_TOOL_CALLS else "Call a tool or answer."))
    junior_dev = "GPU" if jrow["junior_pred"] == 1 else "CPU"
    return {
        "device": ans["device"] if ans else junior_dev,
        "llm_ok": ans is not None,
        "confidence": ans["confidence"] if ans else np.nan,
        "reason": ans["reason"] if ans else "",
        "tools_used": "+".join(used),
        **usage,
    }


# ----------------------------------------------------------------------------
# running everything, with resume
# ----------------------------------------------------------------------------
FIELDS = ["machine", "round", "row_id", "variant", "device", "llm_ok", "confidence", "reason",
          "tools_used", "calls", "prompt_tokens", "output_tokens", "seconds", "model"]


class QuotaExhausted(Exception):
    pass


def done_keys(path):
    if not os.path.exists(path):
        return set()
    d = pd.read_csv(path)
    return set(zip(d["machine"], d["row_id"], d["variant"]))


def run_all(llm, model_name, data_dir, emb_path, out_dir, machines=("amd", "nvidia"),
            variants=VARIANTS, limit=None, log=print):
    """Junior on every exam row (saved to junior_<machine>.csv), then each senior
    variant on the hard rows (appended to senior_answers.csv). Resumable."""
    os.makedirs(out_dir, exist_ok=True)
    similar = SimilarKernels(embed_lib.load_embeddings(emb_path))
    path = os.path.join(out_dir, "senior_answers.csv")
    done = done_keys(path)
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for machine in machines:
            df = devmap.load(machine, data_dir).set_index("row_id", drop=False)
            rounds = devmap.build_exam(df, EXAM, seed=SPLIT_SEED)
            junior = junior_predictions(df, rounds)
            junior.to_csv(os.path.join(out_dir, f"junior_{machine}.csv"), index=False)
            hard = junior[junior["hard"]]
            log(f"{machine}: junior {100 * (junior.junior_pred == junior.label).mean():.1f}% overall; "
                f"{len(hard)} of {len(junior)} rows go to the senior")
            study_of = {r["name"]: df.loc[r["study"]] for r in rounds}
            for variant in variants:
                todo = [j for _, j in hard.iterrows() if (machine, int(j["row_id"]), variant) not in done]
                if limit:
                    todo = todo[:limit]
                log(f"  {variant}: {len(todo)} rows to do")
                errors_in_a_row = 0
                for k, jrow in enumerate(todo, 1):
                    t0 = time.time()
                    try:
                        res = run_case(llm, variant, machine, df.loc[int(jrow["row_id"])], jrow,
                                       study_of[jrow["round"]], similar)
                        errors_in_a_row = 0
                    except QuotaExhausted:
                        log("  Free-tier limit reached. Everything finished so far is saved; "
                            "run the notebook again later (limits reset daily) and it continues.")
                        return path
                    except Exception as err:             # one bad case: record it, keep the junior's answer
                        errors_in_a_row += 1
                        if errors_in_a_row >= 5:
                            raise RuntimeError(f"5 errors in a row, stopping. Last error: {err}") from err
                        res = {"device": "GPU" if jrow["junior_pred"] == 1 else "CPU", "llm_ok": False,
                               "confidence": np.nan, "reason": f"ERROR: {type(err).__name__}: {str(err)[:200]}",
                               "tools_used": "", "calls": 0, "prompt_tokens": 0, "output_tokens": 0}
                    w.writerow({"machine": machine, "round": jrow["round"], "row_id": int(jrow["row_id"]),
                                "variant": variant, "seconds": round(time.time() - t0, 2),
                                "model": model_name, **res})
                    fh.flush()
                    if k % 20 == 0 or k == len(todo):
                        log(f"    {variant}: {k}/{len(todo)}")
    return path


# ----------------------------------------------------------------------------
# Gemini
# ----------------------------------------------------------------------------
def gemini_llm(api_key, model, pause=4.0, max_wait=600):
    """A function prompt -> (text, usage) that calls Gemini, waits `pause` seconds
    between calls (free-tier friendly) and backs off when rate-limited. Raises
    QuotaExhausted after waiting more than max_wait seconds in total for one call
    (typically the daily free limit); finished rows are already saved."""
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json",
                                         max_output_tokens=2048)
    state = {"last": 0.0}

    def call(prompt):
        waited, delay = 0.0, 10.0
        while True:
            gap = time.time() - state["last"]
            if gap < pause:
                time.sleep(pause - gap)
            try:
                state["last"] = time.time()
                resp = client.models.generate_content(model=model, contents=prompt, config=config)
                um = resp.usage_metadata
                usage = {"prompt_tokens": getattr(um, "prompt_token_count", 0) or 0,
                         "output_tokens": (getattr(um, "candidates_token_count", 0) or 0)
                         + (getattr(um, "thoughts_token_count", 0) or 0)}
                return (resp.text or ""), usage
            except Exception as err:            # rate limits and temporary server errors
                code = getattr(err, "code", None)
                if code in (429, 500, 502, 503, 504) or "RESOURCE_EXHAUSTED" in str(err):
                    if waited >= max_wait:
                        raise QuotaExhausted(str(err)[:300])
                    time.sleep(delay)
                    waited += delay
                    delay = min(delay * 2, 120)
                    continue
                raise
    return call
