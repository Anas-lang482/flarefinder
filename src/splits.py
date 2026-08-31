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


def leave_one_region_out(
    df: pd.DataFrame, cfg: Config
) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """One fold per region: hold that region out, train on the rest.

    Preferred over a single fixed holdout for the transfer claim. Three folds
    give three independent tests, and comparing them separates geography from
    size: a model trained on small Saudi flares that ALSO fails on large
    Iranian ones is failing at size, not at crossing a border. A single fold
    cannot make that distinction, and this project's one fixed fold happens
    to be the one whose very-large bin holds a single site.
    """
    regions = sorted(df["region_code"].dropna().unique())
    folds = []
    for r in regions:
        test = df[df["region_code"] == r].copy()
        train = df[df["region_code"] != r].copy()
        if train.empty or test.empty:
            continue
        _assert_disjoint(train, test, "region")
        folds.append((r, train, test))
    return folds


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
    # catalog.parquet, NOT eog_site_year.parquet. The latter carries EOG's
    # per-year site ids, which are re-issued annually, so a by-site split on
    # it puts the same physical flare in both train and test under two
    # different ids -- a silent RULE 1 violation. Only the catalogue has the
    # cross-year clustered ids that make a site split mean anything.
    path = cfg.path(cfg["data"]["processed_dir"], "catalog.parquet")
    if not path.exists():
        print(f"missing {path} -- run `python run.py join` first")
        return 1

    df = assign_size_bins(pd.read_parquet(path), cfg)

    mode = cfg.evaluation.get("region_holdout_mode", "fixed")
    if mode == "leave_one_out":
        folds = leave_one_region_out(df, cfg)
    else:
        train, test = region_holdout(df, cfg)
        folds = [(",".join(sorted(test["region_code"].unique())), train, test)]

    print("=" * 72)
    print(f"REGION HOLDOUT -- CONFOUND DIAGNOSTIC   (mode: {mode}, {len(folds)} fold(s))")
    print("=" * 72)

    summary = []
    for held, train, test in folds:
        rep = confound_report(train, test, cfg)
        matched = size_matched_train(train, test, cfg)
        ks_after = (
            ks_statistic(matched["m3_per_h"], test["m3_per_h"]) if len(matched) else float("nan")
        )

        print(f"\n--- fold: hold out {held} " + "-" * (48 - len(held)))
        print(f"train {sorted(train['region_code'].unique())} n={len(train)}   test n={len(test)}")
        print(
            f"median m3/h  train {rep['train_median_m3h']:>9.1f}   "
            f"test {rep['test_median_m3h']:>9.1f}"
        )
        print(
            f"KS {rep['ks']:.3f} (threshold {rep['ks_threshold']:.3f})"
            f"  ->  {'CONFOUNDED' if rep['confounded'] else 'comparable'}"
        )
        print(f"size-matched train n={len(matched)} (from {len(train)}), KS after = {ks_after:.3f}")
        print(rep["per_bin"][["train_n", "test_n", "test_pct", "usable"]].to_string(
            float_format=lambda v: f"{v:.1f}"
        ))
        if rep["unusable_bins"]:
            print(f"  UNUSABLE bins (test n < {cfg.evaluation['confound_control']['min_sites_per_bin']}): {rep['unusable_bins']}")

        summary.append(
            {
                "held_out": held,
                "test_n": len(test),
                "ks_raw": round(rep["ks"], 3),
                "ks_matched": round(ks_after, 3),
                "matched_n": len(matched),
                "confounded": rep["confounded"],
                "unusable_bins": ";".join(map(str, rep["unusable_bins"])) or "-",
            }
        )

    # The same diagnostic on the site and year splits. It was region-only at
    # first, on the assumption that only a region shift moves the size
    # distribution. That assumption failed once the Saudi data was recovered:
    # by-year came out at KS 0.209. Any split can be confounded, so check all.
    print()
    print("=" * 72)
    print("NON-REGION SPLITS -- same diagnostic")
    print("=" * 72)
    for _name, _fn in (("by-site", site_holdout), ("by-year", year_holdout)):
        _tr, _te = _fn(df, cfg)
        _rep = confound_report(_tr, _te, cfg)
        _m = size_matched_train(_tr, _te, cfg)
        _ka = ks_statistic(_m["m3_per_h"], _te["m3_per_h"]) if len(_m) else float("nan")
        _v = "CONFOUNDED" if _rep["confounded"] else "clean"
        print(
            f"  {_name:<8} train={len(_tr):>6} test={len(_te):>6}  "
            f"KS={_rep['ks']:.3f} -> {_v:<11} (matched n={len(_m)}, KS after {_ka:.3f})"
        )
        summary.append({
            "held_out": _name, "test_n": len(_te), "ks_raw": round(_rep["ks"], 3),
            "ks_matched": round(_ka, 3), "matched_n": len(_m),
            "confounded": _rep["confounded"],
            "unusable_bins": ";".join(map(str, _rep["unusable_bins"])) or "-",
        })

    print("\n" + "=" * 72)
    print("SUMMARY -- report every transfer claim against ALL folds, not one")
    print("=" * 72)
    print(pd.DataFrame(summary).to_string(index=False))
    print(
        "\nA gap that survives size-matching is geographic. A gap that appears\n"
        "in every fold regardless of which region is held out is about flare\n"
        "size, not about crossing a border."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
