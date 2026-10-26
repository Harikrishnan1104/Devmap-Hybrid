"""
Hand-made features shared by the name-tag test and the hybrid model.

  counts  - the six code counts that ship with DevMap (comp, rational, mem,
            localmem, coalesced, atomic)
  raw job numbers - transfer and wgsize exactly as stored. Nearly unique per
            application, so they can act as a "name tag".
  ratios  - scale-free versions that keep the useful information (how much
            data per operation, how much work per work-item) without being a
            name tag on their own:
              transfer_per_op = transfer / (comp + mem)   data moved per operation
              coalesced_share = coalesced / mem           share of tidy memory accesses
              local_share     = localmem / mem            share of local-memory accesses
              comp_per_mem    = comp / mem                arithmetic intensity
              ops_per_item    = (comp + mem) / wgsize     work per work-item in a batch
  grewe4  - the four features Grewe et al. (2013) designed. F3 still multiplies
            by the raw wgsize.
"""
import numpy as np
import pandas as pd

import devmap


def _safe(a, b):
    return (a / b.replace(0, np.nan)).fillna(0.0)


def counts(df):
    return df[devmap.STATIC_FEATURES].astype(float)


def job_numbers(df):
    return df[devmap.JOB_FEATURES].astype(float)


def ratios(df):
    X = df[devmap.STATIC_FEATURES + devmap.JOB_FEATURES].astype(float)
    ops = X["comp"] + X["mem"]
    return pd.DataFrame({
        "transfer_per_op": _safe(X["transfer"], ops),
        "coalesced_share": _safe(X["coalesced"], X["mem"]),
        "local_share": _safe(X["localmem"], X["mem"]),
        "comp_per_mem": _safe(X["comp"], X["mem"]),
        "ops_per_item": _safe(ops, X["wgsize"]),
    }, index=df.index)


def grewe4(df):
    X = df[devmap.STATIC_FEATURES + devmap.JOB_FEATURES].astype(float)
    r = ratios(df)
    return pd.DataFrame({
        "F1": r["transfer_per_op"], "F2": r["coalesced_share"],
        "F3": r["local_share"] * X["wgsize"], "F4": r["comp_per_mem"],
    }, index=df.index)


def forest(df):
    """Our original Random Forest baseline: counts + raw job numbers + Grewe's four."""
    return pd.concat([counts(df), job_numbers(df), grewe4(df)], axis=1)
