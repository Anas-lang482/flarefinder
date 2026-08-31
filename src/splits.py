"""Stage 5 -- holdout construction and confound control.

RULE 1: no site or year may appear in both train and test. This module is
the guard that makes that enforceable rather than aspirational.

It also fixes a measured methodological problem in the region holdout.

THE PROBLEM
    config holds out SAU and trains on IRQ+IRN. But Saudi flares are far
    smaller than Iranian ones (~850 vs ~7800 m3/h mean in 2024). So the
    held-out set differs from the training set in TWO ways at once: it is a
    different region, and it is a different size distribution. If the model
    scores worse on SAU, the pooled number cannot tell you which caused it.
    "Our model does not transfer across regions" and "our model is worse on
    small flares" are different claims with different implications, and a
    judge will ask which one you measured.

THE FIX, in three parts
    1. STRATIFY. Every region-holdout metric is reported per size bin.
       Comparing SAU-small against IRQ/IRN-small is like-for-like.
    2. MATCH. Build a size-matched training set by resampling the training
       regions to the held-out region's size-bin proportions. A gap that
       survives matching is geographic. A gap that disappears was size all
       along.
    3. QUANTIFY. Report the KS statistic between the two log-volume
       distributions, so the write-up states how confounded the raw
       comparison is instead of asserting it is fine.

None of this rescues an underpowered bin: `min_sites_per_bin` marks bins too
small to carry a metric, so they are reported as insufficient rather than as
a number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Config, load_config


# ---------------------------------------------------------------------------
# size bins
# ---------------------------------------------------------------------------
def assign_size_bins(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Attach the config size bins. RULE 2 depends on this column existing."""
    edges = list(cfg.evaluation["size_bins_m3_per_h"])
    labels = list(cfg.evaluation["size_bin_labels"])
    edges = [np.inf if e is None else float(e) for e in edges]

    out = df.copy()
    out["size_bin"] = pd.cut(
        out["m3_per_h"].astype(float),
        bins=edges,
        labels=labels,
        include_lowest=True,
        right=False,
    )
    return out


# ---------------------------------------------------------------------------
# splits (RULE 1)
# ---------------------------------------------------------------------------
def region_holdout(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    held = set(cfg.evaluation["held_out_regions"])
    test = df[df["region_code"].isin(held)].copy()
    train = df[~df["region_code"].isin(held)].copy()
    _assert_disjoint(train, test, "region")
    return train, test


def year_holdout(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    held = set(cfg.evaluation["held_out_years"])
    test = df[df["year"].isin(held)].copy()
    train = df[~df["year"].isin(held)].copy()
    _assert_disjoint(train, test, "year")
    return train, test


def site_holdout(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split on SITE, so no site leaks across the boundary via another year."""
    rng = np.random.default_rng(cfg.seeds["split"])
    keys = df["region_code"].astype(str) + ":" + df["site_id"].astype(str)
    uniq = pd.Index(keys.unique())
    n_test = int(round(len(uniq) * float(cfg.evaluation["test_fraction"])))
    test_ids = set(rng.choice(uniq.to_numpy(), size=n_test, replace=False))
    mask = keys.isin(test_ids)
    train, test = df[~mask].copy(), df[mask].copy()
    _assert_disjoint(train, test, "site")
    return train, test


def _assert_disjoint(train: pd.DataFrame, test: pd.DataFrame, kind: str) -> None:
    """RULE 1, enforced. A silent leak is worse than a crash."""
    if kind == "region":
        overlap = set(train["region_code"]) & set(test["region_code"])
    elif kind == "year":
        overlap = set(train["year"]) & set(test["year"])
    else:
        a = set(train["region_code"].astype(str) + ":" + train["site_id"].astype(str))
        b = set(test["region_code"].astype(str) + ":" + test["site_id"].astype(str))
        overlap = a & b
    if overlap:
        raise AssertionError(
            f"RULE 1 VIOLATION: {len(overlap)} {kind}(s) in both train and test: "
            f"{sorted(list(overlap))[:5]}"
        )


# ---------------------------------------------------------------------------
# confound control
# ---------------------------------------------------------------------------
def ks_statistic(a: pd.Series, b: pd.Series) -> float:
    """Two-sample KS statistic on log10 volume. No scipy dependency needed."""
    a = np.log10(np.asarray(a, dtype=float) + 1.0)
    b = np.log10(np.asarray(b, dtype=float) + 1.0)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    grid = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(np.sort(a), grid, side="right") / len(a)
    cdf_b = np.searchsorted(np.sort(b), grid, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def confound_report(train: pd.DataFrame, test: pd.DataFrame, cfg: Config) -> dict:
    """Quantify how much a region holdout is really a size holdout."""
    cc = cfg.evaluation["confound_control"]
    ks = ks_statistic(train["m3_per_h"], test["m3_per_h"])

    per_bin = pd.DataFrame(
        {
            "train_n": train.groupby("size_bin", observed=False).size(),
            "test_n": test.groupby("size_bin", observed=False).size(),
        }
    )
    per_bin["train_pct"] = 100 * per_bin["train_n"] / per_bin["train_n"].sum()
    per_bin["test_pct"] = 100 * per_bin["test_n"] / per_bin["test_n"].sum()
    per_bin["usable"] = per_bin["test_n"] >= int(cc["min_sites_per_bin"])

    return {
        "ks": ks,
        "ks_threshold": float(cc["max_acceptable_ks"]),
        "confounded": bool(ks > float(cc["max_acceptable_ks"])),
        "train_median_m3h": float(train["m3_per_h"].median()),
        "test_median_m3h": float(test["m3_per_h"].median()),
        "per_bin": per_bin,
        "unusable_bins": per_bin.index[~per_bin["usable"]].tolist(),
    }


def size_matched_train(
    train: pd.DataFrame, test: pd.DataFrame, cfg: Config
) -> pd.DataFrame:
    """Resample train so its size-bin proportions match test.

    Sampling WITHOUT replacement: a bootstrap-style resample with replacement
    would duplicate sites, and duplicated sites inflate apparent sample size
    and break the bootstrap CIs computed later. The cost is a smaller matched
    set, which is the honest trade.
    """
    cc = cfg.evaluation["confound_control"]
    rng = np.random.default_rng(cfg.seeds[cc["matching_seed_key"]])

    test_prop = test.groupby("size_bin", observed=False).size()
    test_prop = test_prop / test_prop.sum()
    train_counts = train.groupby("size_bin", observed=False).size()

    # Largest matched set achievable without replacement is set by whichever
    # bin is scarcest in train relative to the proportion test demands.
    feasible = [
        train_counts.get(b, 0) / p for b, p in test_prop.items() if p > 0
    ]
    total = int(np.floor(min(feasible))) if feasible else 0

    parts = []
    for b, p in test_prop.items():
        want = int(round(total * p))
        pool = train[train["size_bin"] == b]
        if want > 0 and len(pool) > 0:
            take = min(want, len(pool))
            parts.append(pool.iloc[rng.choice(len(pool), size=take, replace=False)])

    return (
        pd.concat(parts, ignore_index=True)
        if parts
        else train.iloc[0:0].copy()
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(config_path: str = "config.yaml") -> int:
    cfg = load_config(config_path)
    path = cfg.path(cfg["data"]["processed_dir"], "eog_site_year.parquet")
    if not path.exists():
        print(f"missing {path} -- run `python run.py download` first")
        return 1

    df = assign_size_bins(pd.read_parquet(path), cfg)
    train, test = region_holdout(df, cfg)

    print("=" * 72)
    print("REGION HOLDOUT -- CONFOUND DIAGNOSTIC")
    print("=" * 72)
    print(f"train regions : {sorted(train['region_code'].unique())}  n={len(train)}")
    print(f"test  regions : {sorted(test['region_code'].unique())}  n={len(test)}")

    rep = confound_report(train, test, cfg)
    print(f"\nmedian volume  train {rep['train_median_m3h']:>10.1f} m3/h")
    print(f"               test  {rep['test_median_m3h']:>10.1f} m3/h")
    print(f"\nKS(log10 volume) = {rep['ks']:.3f}   threshold = {rep['ks_threshold']:.3f}")
    if rep["confounded"]:
        print("  -> CONFOUNDED. A pooled region-holdout number is NOT reportable")
        print("     on its own. Report per size bin, plus the matched comparison.")
    else:
        print("  -> distributions comparable; pooled number is defensible.")

    print("\nsize-bin composition:")
    print(rep["per_bin"].to_string(float_format=lambda v: f"{v:.1f}"))
    if rep["unusable_bins"]:
        print(f"\n  bins with too few test sites to carry a metric: {rep['unusable_bins']}")
        print(f"  (min_sites_per_bin = {cfg.evaluation['confound_control']['min_sites_per_bin']})")

    matched = size_matched_train(train, test, cfg)
    print(f"\nsize-matched training set: n={len(matched)} (from {len(train)})")
    if len(matched):
        print(f"KS after matching = {ks_statistic(matched['m3_per_h'], test['m3_per_h']):.3f}")
        print("  Any performance gap that survives THIS comparison is geographic,")
        print("  not an artefact of Saudi flares being smaller.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
