"""
Scores the Phase 3 junior + senior system.

  python 09_agent_eval.py <folder with junior_<machine>.csv and senior_answers.csv>

For each machine and senior variant:
  - accuracy on the hard rows only (junior vs senior)
  - overall accuracy of the combined system (junior on easy rows + senior on hard rows)
  - McNemar's exact test and an application-level bootstrap 95% range for
    "combined system minus junior alone"
  - how often the senior's reply was unusable (junior's answer kept), which tools
    the agent chose, LLM calls and tokens per hard row
Only variants that have answered every hard row are scored (others are reported
as unfinished).
"""
import importlib
import os
import sys

import numpy as np
import pandas as pd

import devmap

rep = importlib.import_module("06_repeats")


def main(folder, data_dir=devmap.DATA_DIR, log=print):
    senior = pd.read_csv(os.path.join(folder, "senior_answers.csv"))
    senior = senior.drop_duplicates(["machine", "row_id", "variant"], keep="last")
    rows, tools = [], []
    for machine in ["amd", "nvidia"]:
        jpath = os.path.join(folder, f"junior_{machine}.csv")
        if not os.path.exists(jpath):
            continue
        j = pd.read_csv(jpath).sort_values("row_id").reset_index(drop=True)
        apps = devmap.load(machine, data_dir).set_index("row_id").loc[j["row_id"], "app"].to_numpy()
        hard = j["hard"].to_numpy()
        junior_ok = (j["junior_pred"] == j["label"]).to_numpy().astype(int)
        rows.append({"machine": machine, "variant": "junior only", "hard rows": int(hard.sum()),
                     "hard-row accuracy": round(100 * junior_ok[hard].mean(), 1),
                     "overall accuracy": round(100 * junior_ok.mean(), 1)})
        for variant, s in senior[senior["machine"] == machine].groupby("variant"):
            s = s.set_index("row_id")
            missing = set(j.loc[hard, "row_id"]) - set(s.index)
            if missing:
                rows.append({"machine": machine, "variant": variant,
                             "note": f"unfinished: {len(missing)} of {int(hard.sum())} hard rows still to do"})
                continue
            pred = j["junior_pred"].to_numpy().copy()
            hard_ids = j.loc[hard, "row_id"].to_numpy()
            pred[hard] = (s.loc[hard_ids, "device"] == "GPU").astype(int).to_numpy()
            ok = (pred == j["label"].to_numpy()).astype(int)
            lo, hi = rep.app_bootstrap(ok - junior_ok, apps)
            sh = s.loc[hard_ids]
            rows.append({
                "machine": machine, "variant": variant, "hard rows": int(hard.sum()),
                "hard-row accuracy": round(100 * ok[hard].mean(), 1),
                "overall accuracy": round(100 * ok.mean(), 1),
                "gain vs junior": round(100 * (ok.mean() - junior_ok.mean()), 1),
                "95% range": f"{100 * lo:+.1f} to {100 * hi:+.1f}",
                "McNemar p": round(rep.mcnemar_p(ok, junior_ok), 4),
                "unusable replies": int((~sh["llm_ok"].astype(bool)).sum()),
                "LLM calls / row": round(sh["calls"].mean(), 2),
                "tokens / row": int((sh["prompt_tokens"] + sh["output_tokens"]).mean()),
            })
            if variant == "agent":
                used = sh["tools_used"].fillna("")
                for name in ["fact_sheet", "junior_opinion", "similar_kernels"]:
                    tools.append({"machine": machine, "tool": name,
                                  "used in % of hard rows": round(100 * used.str.contains(name).mean(), 1)})
                tools.append({"machine": machine, "tool": "(no tool at all)",
                              "used in % of hard rows": round(100 * (used == "").mean(), 1)})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(folder, "agent_results.csv"), index=False)
    pd.set_option("display.width", 250)
    log(out.to_string(index=False))
    if tools:
        t = pd.DataFrame(tools)
        t.to_csv(os.path.join(folder, "agent_tool_use.csv"), index=False)
        log("\nWhich tools the agent chose:")
        log(t.to_string(index=False))
    return out


if __name__ == "__main__":
    main(sys.argv[1])
