"""Temporal trend analysis: is Gulf flaring rising or falling?

A result available from data already on disk -- no VNF, no Earth Engine. It
gives the project a second independent finding and frames it against a real
international commitment (World Bank Zero Routine Flaring by 2030) rather
than only against a sensor limitation.

THREE CONFOUNDS, ALL HANDLED EXPLICITLY. A naive "sum volume by year and fit
a line" would be wrong for all three reasons below, and would look perfectly
convincing while being wrong.

1. CATALOGUE COVERAGE CHANGES.
   The global catalogue holds 7,209 rows in 2019 and 10,690 in 2024. More
   sites appearing does not mean more flaring -- it can mean better
   detection, reprocessing, or a lower effective threshold. So a rising
   TOTAL is ambiguous.
   Handled by reporting a BALANCED PANEL alongside the raw total: only sites
   observed in every year of the window. Composition is then fixed, and a
   change in the panel is a change in flaring at those sites, not a change
   in who is being counted.

2. CALIBRATION CHANGES MID-RECORD.
   Verified from the source filenames: 2012-2016 was produced with slope
   0.0298, 2017-2024 with 0.029353. A trend fitted across that boundary
   absorbs the calibration change. The whole processing chain differs
   between those eras, not only the constant.
   Handled by fitting the primary trend on 2017-2024 only, and reporting the
   earlier era separately and clearly marked.

3. THE TREND IS DRIVEN BY A FEW HUGE SITES.
   Volume is extremely skewed, so a total is a statement about the largest
   flares and says nothing about the small ones this project is about.
   Handled by reporting the trend PER SIZE BIN as well as in aggregate.

UNCERTAINTY
   Bootstrap resamples SITES, not years. Sites are the independent units; the
   years are repeated measurements on them. Resampling years would treat 8
   observations as the sample size and give absurdly wide intervals, and
   would break the time ordering the slope is fitted on.

Run:
    python run.py trends
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Config, load_config
from src.experiment import log_metrics
from src.splits import assign_size_bins

# The calibration boundary, read off the source filenames, not assumed.
CALIBRATION_ERAS = {"0.0298": (2012, 2016), "0.029353": (2017, 2024)}
PRIMARY_ERA = (2017, 2024)

# Reference palette, categorical slots 1-3, fixed order, never cycled.
PALETTE = {"Iran": "#2a78d6", "Iraq": "#eb6834", "Saudi Arabia": "#1baf7a"}
OTHER_COLOR = "#8a8985"
INK, INK2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e3e2dd"


def annual_totals(df: pd.DataFrame, by: str = "region_code") -> pd.DataFrame:
    """Annual volume, plus a count of ACTIVE sites.

    Active means volume > 0, not merely present as a row. That distinction is
    not pedantry: the wide 2012-2016 workbook carries a row for every site in
    all five years whether or not it was detected that year, while the
    per-year files list only detected sites. Counting rows would therefore
    show a flat, identical site count for 2012-2016 and a sudden drop at
    2017 -- an artefact of file layout that looks exactly like a real
    collapse in coverage.
    """
    active = df[df["volume_mcm"] > 0]
    vol = df.groupby(["year", by], as_index=False).agg(total_mcm=("volume_mcm", "sum"))
    cnt = active.groupby(["year", by], as_index=False).agg(n_sites=("site_id", "nunique"))
    return vol.merge(cnt, on=["year", by], how="left").fillna({"n_sites": 0}).sort_values(["year", by])


def balanced_panel(df: pd.DataFrame, y0: int, y1: int) -> pd.DataFrame:
    """Only sites observed in EVERY year of the window.

    Fixes catalogue composition, so a change is a change in flaring rather
    than a change in who is counted. The cost is sample size -- and the
    surviving sites skew toward large, persistent flares, which is itself
    worth stating.
    """
    w = df[(df["year"] >= y0) & (df["year"] <= y1)]
    n_years = y1 - y0 + 1
    counts = w.groupby("site_id")["year"].nunique()
    keep = counts.index[counts == n_years]
    return w[w["site_id"].isin(keep)].copy()


def _slope_pct_per_year(years: np.ndarray, totals: np.ndarray) -> float:
    """Fit log10(total) ~ year; return compound % change per year.

    Log space because flaring changes multiplicatively -- "down 5% a year" is
    the meaningful statement, not "down 400 Mm3 a year", which would mean
    something different in Iraq than in Saudi Arabia.
    """
    ok = totals > 0
    if ok.sum() < 3:
        return float("nan")
    slope = np.polyfit(years[ok], np.log10(totals[ok]), 1)[0]
    return (10.0**slope - 1.0) * 100.0


def bootstrap_trend(
    df: pd.DataFrame, cfg: Config, n_resamples: int | None = None
) -> tuple[float, float, float]:
    """Point estimate and CI for % change per year. Resamples SITES."""
    bs = cfg.evaluation["bootstrap"]
    n = int(n_resamples or bs["n_resamples"])
    level = float(bs["ci_level"])
    rng = np.random.default_rng(int(cfg.seeds["bootstrap"]))

    piv = df.pivot_table(index="site_id", columns="year", values="volume_mcm",
                         aggfunc="sum", fill_value=0.0)
    years = piv.columns.to_numpy(float)
    mat = piv.to_numpy(float)
    point = _slope_pct_per_year(years, mat.sum(axis=0))

    draws = []
    for _ in range(n):
        idx = rng.integers(0, mat.shape[0], mat.shape[0])
        draws.append(_slope_pct_per_year(years, mat[idx].sum(axis=0)))
    draws = np.array([d for d in draws if np.isfinite(d)])
    if len(draws) == 0:
        return point, float("nan"), float("nan")
    lo = (1 - level) / 2 * 100
    return point, float(np.percentile(draws, lo)), float(np.percentile(draws, 100 - lo))


def zrf_required_rate(current_year: int, target_year: int = 2030) -> float:
    """Rate needed to reach zero routine flaring by the target year.

    Zero is unreachable by any constant percentage, so this reports the rate
    that would remove 90% of current volume -- a concrete, statable yardstick
    rather than a meaningless -100%.
    """
    n = max(target_year - current_year, 1)
    return (10 ** (np.log10(0.10) / n) - 1) * 100


def make_figure(totals: pd.DataFrame, cfg: Config, name_map: dict) -> list:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(9, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    fig.patch.set_facecolor(SURFACE)
    for a in (ax, ax2):
        a.set_facecolor(SURFACE)
        a.grid(True, color=GRID, linewidth=0.6, zorder=0)
        a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(GRID)
        a.tick_params(colors=INK2, labelsize=9)

    for code, grp in totals.groupby("region_code"):
        label = name_map.get(code, code)
        colour = PALETTE.get(label, OTHER_COLOR)
        g = grp.sort_values("year")
        ax.plot(g["year"], g["total_mcm"], color=colour, linewidth=2,
                marker="o", markersize=5, label=label, zorder=3)
        ax2.plot(g["year"], g["n_sites"], color=colour, linewidth=2,
                 marker="o", markersize=4, zorder=3)

    # Mark the calibration boundary. Without it a reader reads the whole
    # series as one consistent measurement, which it is not.
    for a in (ax, ax2):
        a.axvline(2016.5, color=INK2, linestyle="--", linewidth=1, zorder=2)
    ax.text(2016.6, ax.get_ylim()[1] * 0.97, "calibration change\n0.0298 → 0.029353",
            color=INK2, fontsize=8, va="top")

    ax.set_ylabel("Flared volume (Mm³/yr)", color=INK2, fontsize=10)
    ax2.set_ylabel("Sites in\ncatalogue", color=INK2, fontsize=10)
    ax2.set_xlabel("Year", color=INK2, fontsize=10)
    ax.set_title("Gulf gas flaring, 2012–2024", color=INK, fontsize=14, loc="left", pad=26)
    ax.text(0.0, 1.015, "EOG VIIRS record  |  volume above, catalogue coverage below",
            transform=ax.transAxes, color=INK2, fontsize=9.5, va="bottom")
    leg = ax.legend(frameon=False, fontsize=9.5, labelcolor=INK2)
    leg.set_zorder(5)

    fig.text(0.5, 0.015,
             "Lower panel is the control: a rise in volume that tracks a rise in site count "
             "may be improved detection,\nnot increased flaring. Volumes before 2017 were "
             "produced with a different calibration constant.",
             ha="center", color=INK2, fontsize=8, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    out_dir = cfg.path(cfg["outputs"]["figures_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in ("png", "pdf"):
        p = out_dir / f"fig02_flaring_trend.{ext}"
        fig.savefig(p, dpi=int(cfg["outputs"]["figure_dpi"]), facecolor=SURFACE)
        written.append(p)
    plt.close(fig)
    return written


def main(config_path: str = "config.yaml") -> int:
    cfg = load_config(config_path)
    path = cfg.path(cfg["data"]["processed_dir"], "catalog.parquet")
    if not path.exists():
        print(f"missing {path} -- run `python run.py join` first")
        return 1

    df = assign_size_bins(pd.read_parquet(path), cfg)
    name_map = df.groupby("region_code")["country"].agg(
        lambda s: s.mode().iat[0] if not s.mode().empty else s.iat[0]
    ).to_dict()

    y0, y1 = PRIMARY_ERA
    print("=" * 74)
    print("GULF FLARING TREND")
    print("=" * 74)
    print(f"primary window : {y0}-{y1} (single calibration era, slope 0.029353)")
    print(f"excluded       : 2012-2016, produced with slope 0.0298 -- a trend")
    print(f"                 across that boundary absorbs a calibration change")

    totals = annual_totals(df)
    written = make_figure(totals, cfg, name_map)

    era = df[(df["year"] >= y0) & (df["year"] <= y1)]
    panel = balanced_panel(df, y0, y1)
    n_all, n_panel = era["site_id"].nunique(), panel["site_id"].nunique()

    print(f"\nsites in window          : {n_all}")
    print(f"sites in balanced panel  : {n_panel} ({100*n_panel/n_all:.1f}%) "
          f"-- present in all {y1-y0+1} years")
    print(f"panel share of volume    : {100*panel['volume_mcm'].sum()/era['volume_mcm'].sum():.1f}%")

    records = []
    print("\n" + "-" * 74)
    print("TREND, % change per year (negative = falling), 95% bootstrap CI")
    print("-" * 74)
    print(f"{'scope':<26} {'all sites':>26} {'balanced panel':>20}")
    rows = [("Gulf (all three)", None)] + [(name_map.get(c, c), c) for c in sorted(era["region_code"].unique())]
    for label, code in rows:
        a = era if code is None else era[era["region_code"] == code]
        p = panel if code is None else panel[panel["region_code"] == code]
        pa, la, ha = bootstrap_trend(a, cfg)
        pp, lp, hp = bootstrap_trend(p, cfg)
        print(f"{label:<26} {pa:+6.2f}%  [{la:+6.2f}, {ha:+6.2f}] {pp:+8.2f}%  [{lp:+5.2f}, {hp:+5.2f}]")
        records += [
            {"metric": "trend_pct_per_year_all", "size_bin": "", "value": round(pa, 3),
             "ci_lo": round(la, 3), "ci_hi": round(ha, 3), "n": a["site_id"].nunique()},
            {"metric": "trend_pct_per_year_panel", "size_bin": "", "value": round(pp, 3),
             "ci_lo": round(lp, 3), "ci_hi": round(hp, 3), "n": p["site_id"].nunique()},
        ]

    print("\n" + "-" * 74)
    print("TREND BY FLARE SIZE (balanced panel) -- RULE 2")
    print("-" * 74)
    for b in panel["size_bin"].cat.categories:
        sub = panel[panel["size_bin"] == b]
        n = sub["site_id"].nunique()
        floor = int(cfg.evaluation["confound_control"]["min_sites_per_bin"])
        if n < floor:
            print(f"  {str(b):<12} n={n:<5} insufficient (floor {floor})")
            continue
        pt, lo, hi = bootstrap_trend(sub, cfg)
        print(f"  {str(b):<12} n={n:<5} {pt:+6.2f}%  [{lo:+6.2f}, {hi:+6.2f}]")
        records.append({"metric": "trend_pct_per_year_by_size", "size_bin": str(b),
                        "value": round(pt, 3), "ci_lo": round(lo, 3),
                        "ci_hi": round(hi, 3), "n": n})

    req = zrf_required_rate(y1)
    pt, lo, hi = bootstrap_trend(panel, cfg)
    print("\n" + "-" * 74)
    print("AGAINST ZERO ROUTINE FLARING BY 2030")
    print("-" * 74)
    print(f"  observed (balanced panel, {y0}-{y1}) : {pt:+.2f}% per year [{lo:+.2f}, {hi:+.2f}]")
    print(f"  needed from {y1} to remove 90% by 2030: {req:+.2f}% per year")
    if pt > req:
        print(f"\n  -> Observed decline is SLOWER than the 2030 pathway requires.")
    else:
        print(f"\n  -> Observed decline is at or faster than the 2030 pathway.")
    print("  'Zero' is unreachable at any constant rate, so the yardstick is a")
    print("  90% reduction. State that when quoting this number.")

    log_metrics(cfg, stage="trends", script="src/trends.py",
                model="loglinear_trend", split_type="none",
                params={"window": [y0, y1], "excluded_era": "2012-2016 (slope 0.0298)"},
                records=records,
                notes="Balanced panel controls catalogue coverage; single calibration era.")
    for p in written:
        print(f"\nwrote {p}")
    print(f"logged to {cfg['experiments']['log_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
