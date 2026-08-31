"""Stage 6 -- the VIIRS volume calibration baseline.

RULE 6: this runs BEFORE anything is built. The project claims to replace
hand-fitted volume calibrations that are biased on small flares. That claim
is only meaningful if the failure is measured first, on our own data.

WHAT THIS CAN AND CANNOT SHOW -- read before quoting any number from it
    EOG's published volumes ARE the output of a hand-fitted VIIRS
    calibration. Fitting a calibration to predict them is therefore
    reproducing a formula, not testing it against truth. There is no
    independent metered volume in this dataset -- Asadi-Fard et al. 2024 had
    exactly one Iranian site with metering, and we have none.

    So this module measures ONE thing honestly:
        how well a standard physically-motivated power law reproduces EOG's
        published volumes, and whether that agreement DEGRADES WITH FLARE
        SIZE.
    A size-dependent residual means the published relationship is less
    determined for small flares than for large ones -- consistent with the
    bias Elvidge et al. 2024 report, and a genuine motivation for learning
    the mapping instead of fixing it by hand. It is NOT proof that EOG's
    volumes are wrong. Do not write that sentence in the paper.

THE FUNCTIONAL FORM
    Radiated power follows Stefan-Boltzmann, P = sigma * A * T^4, and the
    gas burned over a year scales with radiated power times the fraction of
    the time the flare is actually alight:
        volume  ~  A * T^4 * detection_freq
    We have temperature and detection frequency directly. We do NOT have
    source area: the EOG workbooks give ellipticity, a shape descriptor of
    the detection footprint, which is a weak proxy for area at best. So the
    model is fitted in log space with free exponents rather than imposing
    the theoretical 4:
        log10(V) = a + b*log10(T) + c*log10(detection_freq)
                     + d*log10(ellipticity)
    Fitting b rather than fixing b=4 is deliberate: if the recovered
    exponent lands near 4, that is evidence the form is right; forcing it
    would throw that check away.

    The published EOG slope constant (0.029353) appears in the workbook
    filenames. We cannot apply it directly -- it multiplies radiant heat,
    which the per-site sheets do not contain.

Run:
    python run.py baseline
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Config, load_config
from src.experiment import log_metrics
from src.splits import assign_size_bins, site_holdout

PREDICTORS = ["avg_temp_k", "detection_freq", "ellipticity"]
EPS = 1e-6


def _design_matrix(df: pd.DataFrame) -> np.ndarray:
    cols = [np.log10(np.clip(df[c].astype(float).to_numpy(), EPS, None)) for c in PREDICTORS]
    return np.column_stack([np.ones(len(df))] + cols)


def fit_power_law(train: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Least squares in log space. The classic hand-fitted calibration form."""
    ok = train[PREDICTORS + ["m3_per_h"]].notna().all(axis=1) & (train["m3_per_h"] > 0)
    t = train[ok]
    X = _design_matrix(t)
    y = np.log10(t["m3_per_h"].astype(float).to_numpy())
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef, {"n_fit": int(len(t)), "n_dropped": int((~ok).sum())}


def predict(coef: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    """Predicted m3/h. expm-style inverse: 10**x is never negative.

    This is the same structural guarantee RULE 4 demands of the learned
    model, and it holds here for free.
    """
    return 10.0 ** (_design_matrix(df) @ coef)


def _bootstrap_ci(values: np.ndarray, stat, n: int, seed: int, level: float) -> tuple[float, float]:
    """RULE 3: every headline number carries a bootstrap interval."""
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = [stat(values[rng.integers(0, len(values), len(values))]) for _ in range(n)]
    lo = (1 - level) / 2 * 100
    return (float(np.percentile(draws, lo)), float(np.percentile(draws, 100 - lo)))


def evaluate(test: pd.DataFrame, coef: np.ndarray, cfg: Config) -> pd.DataFrame:
    """Per-size-bin error. RULE 2: never a single pooled number."""
    bs = cfg.evaluation["bootstrap"]
    n_res, level = int(bs["n_resamples"]), float(bs["ci_level"])
    seed = int(cfg.seeds["bootstrap"])

    ok = test[PREDICTORS + ["m3_per_h"]].notna().all(axis=1) & (test["m3_per_h"] > 0)
    t = test[ok].copy()
    t["pred"] = predict(coef, t)

    # Residual in log10 space = log10(pred/actual). Positive means the
    # calibration OVERESTIMATES. Log space is the right frame: an error of
    # 100 m3/h means something entirely different on a 50 m3/h flare than on
    # a 50,000 m3/h one, and a linear residual would be dominated by the
    # largest flares -- exactly the sites this project is not about.
    t["log_resid"] = np.log10(t["pred"]) - np.log10(t["m3_per_h"].astype(float))

    rows = []
    for name, grp in t.groupby("size_bin", observed=False):
        r = grp["log_resid"].to_numpy()
        if len(r) == 0:
            continue
        med = float(np.median(r))
        lo, hi = _bootstrap_ci(r, np.median, n_res, seed, level)
        rows.append({
            "size_bin": str(name),
            "n": len(r),
            "median_log_bias": round(med, 4),
            "ci_lo": round(lo, 4),
            "ci_hi": round(hi, 4),
            # A log10 bias of +0.3 is a factor of 2 overestimate. Reporting
            # the factor as well, because "x2 too high" is legible and
            # "+0.30 dex" is not.
            "bias_factor": round(10 ** med, 3),
            "mae_log": round(float(np.mean(np.abs(r))), 4),
            "excludes_zero": not (lo <= 0 <= hi),
        })
    return pd.DataFrame(rows)


def main(config_path: str = "config.yaml") -> int:
    cfg = load_config(config_path)
    path = cfg.path(cfg["data"]["processed_dir"], "catalog.parquet")
    if not path.exists():
        print(f"missing {path} -- run `python run.py join` first")
        return 1

    df = assign_size_bins(pd.read_parquet(path), cfg)
    # by-site: the ONLY unconfounded split (KS 0.045). RULE 1.
    train, test = site_holdout(df, cfg)

    coef, info = fit_power_law(train)
    print("=" * 74)
    print("BASELINE -- hand-fitted VIIRS power-law calibration (RULE 6)")
    print("=" * 74)
    print(f"split       : by-site holdout, train n={len(train)}, test n={len(test)}")
    print(f"fitted on   : {info['n_fit']} rows ({info['n_dropped']} dropped: missing or zero volume)")
    print("\nfitted form : log10(V) = a + b*log10(T) + c*log10(det_freq) + d*log10(ellip)")
    for nm, c in zip(["a (intercept)", "b (log T)", "c (log det_freq)", "d (log ellip)"], coef):
        print(f"  {nm:<18} {c:+.4f}")
    print(
        f"\n  Stefan-Boltzmann predicts b ~ 4 if temperature drives radiated\n"
        f"  power. Recovered b = {coef[1]:+.2f}."
    )

    # Predictor relevance. Without this the fitted coefficients read as if
    # each predictor is doing work, and here two of the three are not.
    print()
    print("-" * 74)
    print("PREDICTOR RELEVANCE -- what is actually carrying the fit")
    print("-" * 74)
    ok_tr = train[PREDICTORS + ["m3_per_h"]].notna().all(axis=1) & (train["m3_per_h"] > 0)
    tt = train[ok_tr]
    ly = np.log10(tt["m3_per_h"].astype(float).to_numpy())
    for c in PREDICTORS:
        lx = np.log10(np.clip(tt[c].astype(float).to_numpy(), EPS, None))
        print(f"  corr(log {c:<15}, log volume) = {np.corrcoef(lx, ly)[0, 1]:+.3f}")
    lx = np.log10(np.clip(tt["detection_freq"].astype(float).to_numpy(), EPS, None))
    X1 = np.column_stack([np.ones(len(tt)), lx])
    c1, *_ = np.linalg.lstsq(X1, ly, rcond=None)
    r2_det = 1 - ((ly - X1 @ c1) ** 2).sum() / ((ly - ly.mean()) ** 2).sum()
    Xf = _design_matrix(tt)
    r2_full = 1 - ((ly - Xf @ coef) ** 2).sum() / ((ly - ly.mean()) ** 2).sum()
    print()
    print(f"  R2, detection_freq alone : {r2_det:.3f}")
    print(f"  R2, all three predictors : {r2_full:.3f}")
    print(f"  gain from adding temperature and ellipticity: {r2_full - r2_det:+.3f}")
    for line in [
        "",
        "  !! THIS IS NOT A REPRODUCTION OF THE EOG CALIBRATION.",
        "     EOG's formula multiplies RADIANT HEAT by a fitted slope",
        "     (0.029353, visible in the workbook filenames). Radiant heat and",
        "     source area are NOT in the per-site sheets, so they cannot enter",
        "     this fit. Temperature carries almost no signal on its own here",
        "     (corr +0.03), and its large negative coefficient is a suppression",
        "     artefact, not physics -- Stefan-Boltzmann would give roughly +4.",
        "     What is fitted is essentially a detection-frequency regression.",
        "     To reproduce the published calibration we need VNF per-detection",
        "     records, which carry radiant heat and source area. That is the",
        "     src/data_viirs.py stage, currently blocked on an access path.",
    ]:
        print(line)

    res = evaluate(test, coef, cfg)
    print("\n" + "-" * 74)
    print("ERROR BY FLARE SIZE (RULE 2) -- log10 bias, + means OVERESTIMATE")
    print(f"bootstrap {cfg.evaluation['bootstrap']['n_resamples']} resamples, "
          f"{int(100 * float(cfg.evaluation['bootstrap']['ci_level']))}% CI (RULE 3)")
    print("-" * 74)
    print(res.to_string(index=False))

    small = res[res["size_bin"].isin(["tiny", "sub-VIIRS"])]
    large = res[res["size_bin"].isin(["large", "very-large"])]
    print("\n" + "-" * 74)
    if not small.empty and not large.empty:
        ds = float(small["median_log_bias"].mean())
        dl = float(large["median_log_bias"].mean())
        print(f"small bins (tiny+sub-VIIRS) mean log bias : {ds:+.3f}  (x{10**ds:.2f})")
        print(f"large bins (large+very-large)             : {dl:+.3f}  (x{10**dl:.2f})")
        print(f"size-dependent gap                        : {ds - dl:+.3f} dex")
        if abs(ds - dl) > 0.1:
            print(
                "\n  -> The calibration's error DEPENDS ON FLARE SIZE. This is the\n"
                "     failure the project proposes to fix, measured on our own data.\n"
                "     It is a divergence from EOG's published formula, NOT proof that\n"
                "     EOG is wrong: there is no metered volume here to test against."
            )
        else:
            print(
                "\n  -> No meaningful size dependence. The premise that hand-fitted\n"
                "     calibration fails on small flares is NOT supported here, and\n"
                "     the project's framing needs revisiting. A measured negative\n"
                "     result is still a result."
            )

    log_metrics(
        cfg,
        stage="baseline",
        script="src/baseline.py",
        model="viirs_power_law_calibration",
        split_type="by_site",
        params={"predictors": PREDICTORS, "coef": [round(float(c), 5) for c in coef]},
        records=[
            {"metric": "median_log_bias", "size_bin": r["size_bin"], "value": r["median_log_bias"],
             "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "n": r["n"]}
            for _, r in res.iterrows()
        ],
        notes="RULE 6 baseline. Reproduces EOG volumes, not independent truth.",
    )
    print(f"\nlogged to {cfg['experiments']['log_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
