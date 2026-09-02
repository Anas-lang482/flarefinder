"""Stage 4b -- Sentinel-2 SWIR extraction via Earth Engine.

Flames radiate strongly in short-wave infrared. Sentinel-2 carries B11
(1610 nm) and B12 (2190 nm) at 20 m, so a flare too small for VIIRS to see at
750 m can still light up a handful of S2 pixels. B8A (865 nm) is carried as a
NON-SWIR control band: a real flare should be far brighter in B12 than in
B8A, whereas sunlit bright sand is bright in all of them.

THE SANITY CHECK COMES FIRST, AND IT CAN STOP THE PROJECT
    Before extracting thousands of points, extract the largest flares in the
    catalogue and confirm they actually glow. If a 180,000 m3/h Iraqi flare
    does not show elevated B12 against its own control points, then either
    the approach does not work or something is broken -- buffer size, cloud
    handling, coordinates. Finding that out now costs an hour. Finding it out
    in week 5 costs the project.
    Run:  python run.py s2 --sanity

TWO DESIGN DECISIONS THAT MATTER MORE THAN THEY LOOK
    MAX COMPOSITE, NOT MEDIAN.
        Flares flicker. Over a month a site might be lit in three passes out
        of six. A median composite averages the lit and unlit passes together
        and destroys precisely the signal being looked for -- it would make
        an intermittent flare look like ordinary ground. The maximum keeps
        the brightest observation, which is the one that carries information.
        This is the same reasoning as the intermittency feature: the variance
        across passes IS the signal, so never reduce it away early.

    SCENE-LEVEL CLOUD FILTER, NOT PER-PIXEL SCL MASKING.
        The obvious move is to mask cloud with the SCL band. The risk is that
        SCL can label an anomalously bright hot pixel as cloud or as
        saturated-defective, which would delete the flare itself and leave a
        clean-looking null result. So scenes are filtered on
        CLOUDY_PIXEL_PERCENTAGE and pixels are left alone. Revisit only with
        evidence, and check what SCL assigns to known flare pixels first.

SATURATION -- THE METHOD HAS AN UPPER LIMIT AS WELL AS A LOWER ONE
    Measured 2026-09-01 on the very-large bin (>20,000 m3/h): 65% of B8A and
    74% of B12 observations sit at or above 10000, which is 100% reflectance.
    In 63% of observations BOTH bands are saturated at once, and when that
    happens the B12/B8A ratio is forced toward 1 -- median 0.878 for
    saturated observations against 1.070 for unsaturated ones.

    So the ratio metric COLLAPSES at the brightest flares. The detection
    curve fails at both ends for different reasons: too dim to separate below
    ~360 m3/h, saturated above ~20,000 m3/h. Measured separation (difference
    in % exceeding B12/B8A > 1.2, 95% CI):
        tiny         0 pp  [-30, +30]   no separation
        sub-VIIRS   23 pp  [ -3, +50]   marginal, the decisive bin
        small       33 pp  [ +3, +60]   separates
        medium      40 pp  [+10, +67]   separates
        large       63 pp  [+40, +83]   separates
        very-large  17 pp  [-13, +47]   fails -- saturation, not dimness

    CONSEQUENCE FOR features.py: a single ratio cannot span the range. Add
    saturation-aware features -- a saturated-pixel count, and raw B12 for
    bright targets with the ratio reserved for dim ones. Never read a
    saturated radiance as a quantitative brightness.

    All bins above used n=15, below the 20-site floor in config. Treat as a
    first read, not a result.

QUOTA
    Community tier: 150 EECU-hours, no billing account. Every extraction is
    cached to parquet and never re-pulled. Debug against the cached table,
    never against Earth Engine -- re-running extraction loops while fixing
    downstream code is how the budget disappears.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Config, load_config

BANDS = ["B8A", "B11", "B12"]


def _init(cfg: Config):
    import ee

    ee.Initialize(project=cfg["data"]["earthengine"]["project"])
    return ee


def monthly_swir(points: pd.DataFrame, year: int, cfg: Config) -> pd.DataFrame:
    """Monthly max SWIR per point for one year. One EE call per month."""
    ee = _init(cfg)
    s2cfg = cfg["data"]["sentinel2"]
    buffer_m = float(s2cfg.get("buffer_m", 100))
    max_cloud = float(s2cfg["max_cloud_pct"])

    fc = ee.FeatureCollection([
        ee.Feature(
            ee.Geometry.Point([float(r.lon), float(r.lat)]).buffer(buffer_m),
            {"point_id": str(r.point_id)},
        )
        for r in points.itertuples()
    ])

    rows = []
    for month in range(1, 13):
        start = ee.Date.fromYMD(year, month, 1)
        end = start.advance(1, "month")
        col = (
            ee.ImageCollection(s2cfg["gee_collection"])
            .filterDate(start, end)
            .filterBounds(fc.geometry())
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
            .select(BANDS)
        )
        n_scenes = col.size().getInfo()
        if n_scenes == 0:
            print(f"    {year}-{month:02d}  no scenes under {max_cloud:.0f}% cloud")
            continue

        # max, not median -- see module docstring.
        # No setOutputs: a single-output reducer over a multi-band image
        # already names its results after the bands (B8A, B11, B12).
        stats = col.max().reduceRegions(
            collection=fc,
            reducer=ee.Reducer.max(),
            scale=int(s2cfg["native_resolution_m"]),
        )
        feats = stats.getInfo()["features"]
        for f in feats:
            p = f["properties"]
            rows.append({
                "point_id": p.get("point_id"),
                "year": year,
                "month": month,
                "n_scenes": n_scenes,
                **{f"{b}_max": p.get(b) for b in BANDS},
            })
        print(f"    {year}-{month:02d}  {n_scenes:>3} scenes, {len(feats)} points")

    return pd.DataFrame(rows)


def sanity(cfg: Config, n_per_bin: int = 15) -> int:
    """At what flare size does Sentinel-2 stop separating flares from desert?

    The first version of this check took the 20 LARGEST flares and passed.
    That answered a question nobody was asking: everyone already knows huge
    flares are visible from space -- Liu et al. 2023 operationalised exactly
    that offshore. This project is about the small ones, so the check now
    samples every size bin and reports separation per bin.

    The output is a detection curve, not a verdict. It says which size bins
    can carry a claim and where the method stops working -- and a bin where
    it stops working is a real finding, not a failure.
    """
    from src.splits import assign_size_bins

    proc = cfg.path(cfg["data"]["processed_dir"])
    catalog = assign_size_bins(pd.read_parquet(proc / "catalog.parquet"), cfg)
    controls = pd.read_parquet(proc / "controls.parquet")

    year = int(cfg.years["s2_end"])
    latest = catalog[catalog["year"] == year]
    if "is_onshore" in latest.columns:
        latest = latest[latest["is_onshore"].astype(bool)]

    # Sample per bin, seeded from seeds.split -- this is sample selection,
    # not model fitting, so it must not share a seed with the model.
    rng_seed = int(cfg.seeds["split"])
    picked = []
    for b in latest["size_bin"].cat.categories:
        sub = latest[latest["size_bin"] == b]
        if len(sub) == 0:
            continue
        picked.append(sub.sample(min(n_per_bin, len(sub)), random_state=rng_seed))
    flares = pd.concat(picked, ignore_index=True)[
        ["site_id", "lat", "lon", "volume_mcm", "m3_per_h", "size_bin"]
    ]

    # Two controls per sampled flare. Controls are matched to their OWN site,
    # so each bin is compared against desert from the same neighbourhood --
    # a like-for-like comparison rather than one global control pool.
    ctl = controls[controls["paired_site_id"].isin(flares["site_id"])]
    ctl = ctl.groupby("paired_site_id", group_keys=False).head(2)
    ctl = ctl.merge(
        flares[["site_id", "size_bin"]].rename(columns={"site_id": "paired_site_id"}),
        on="paired_site_id", how="left",
    )

    pts = pd.concat([
        flares.rename(columns={"site_id": "point_id"}).assign(kind="flare"),
        ctl.rename(columns={"control_id": "point_id"})[
            ["point_id", "lat", "lon", "size_bin"]
        ].assign(kind="control"),
    ], ignore_index=True)

    print("=" * 74)
    print(f"SENTINEL-2 DETECTION CURVE -- separation by flare size ({year})")
    print("=" * 74)
    print(flares.groupby("size_bin", observed=False).agg(
        n=("site_id", "size"),
        min_m3h=("m3_per_h", "min"),
        max_m3h=("m3_per_h", "max"),
    ).round(1).to_string())
    print()
    print(f"total points: {int((pts.kind=='flare').sum())} flares "
          f"+ {int((pts.kind=='control').sum())} paired controls")

    cache = proc / f"s2_bins_{year}.parquet"
    if cache.exists():
        print()
        print(f"using cached {cache.name} (delete to re-pull from Earth Engine)")
        obs = pd.read_parquet(cache)
    else:
        print()
        obs = monthly_swir(pts, year, cfg)
        obs.to_parquet(cache, index=False)
        print(f"wrote {cache}")

    df = obs.merge(pts[["point_id", "kind", "size_bin"]], on="point_id", how="left")
    df = df[df["B12_max"].notna()].copy()
    df["b12_b8a"] = df["B12_max"] / df["B8A_max"].replace(0, np.nan)

    # Per SITE max across months, not the median: flares are intermittent
    # (median detection_freq 0.077), so most monthly composites catch the site
    # unlit and a median averages the signal away. The lit pass is the
    # informative one.
    per_site = (
        df.groupby(["point_id", "kind", "size_bin"], observed=False)["b12_b8a"]
        .max().reset_index().dropna(subset=["b12_b8a"])
    )

    thr = float(cfg["data"]["sentinel2"].get("sanity_b12_b8a_min", 1.2))
    bs = cfg.evaluation["bootstrap"]
    rng = np.random.default_rng(int(cfg.seeds["bootstrap"]))
    n_res = int(bs["n_resamples"])

    print()
    print("" + "-" * 74)
    print("SEPARATION BY SIZE BIN -- per-site max B12/B8A, and the difference")
    print(f"in % of points exceeding {thr}, with 95% bootstrap CI (RULE 3)")
    print("-" * 74)
    print(f"{'size_bin':<12}{'n_fl':>5}{'n_ct':>5}{'flare':>8}{'ctrl':>8}"
          f"{'%fl':>6}{'%ct':>6}{'diff pp':>9}{'95% CI':>18}")

    floor = int(cfg.evaluation["confound_control"]["min_sites_per_bin"])
    for b in per_site["size_bin"].cat.categories:
        sub = per_site[per_site["size_bin"] == b]
        f = sub[sub.kind == "flare"]["b12_b8a"].to_numpy()
        c = sub[sub.kind == "control"]["b12_b8a"].to_numpy()
        if len(f) == 0 or len(c) == 0:
            continue
        ef, ec = (f > thr).astype(float), (c > thr).astype(float)
        draws = np.array([
            ef[rng.integers(0, len(ef), len(ef))].mean()
            - ec[rng.integers(0, len(ec), len(ec))].mean()
            for _ in range(n_res)
        ]) * 100
        lo, hi = np.percentile(draws, 2.5), np.percentile(draws, 97.5)
        mark = "" if len(f) >= floor else "  (below the 20-site floor)"
        print(f"{str(b):<12}{len(f):>5}{len(c):>5}{np.median(f):>8.2f}{np.median(c):>8.2f}"
              f"{100*ef.mean():>6.0f}{100*ec.mean():>6.0f}"
              f"{100*(ef.mean()-ec.mean()):>9.0f}"
              f"{f'[{lo:+.0f}, {hi:+.0f}]':>18}{mark}")

    print()
    print("" + "-" * 74)
    print("HOW TO READ THIS")
    print("-" * 74)
    print("  A CI excluding zero means Sentinel-2 separates flares from their own")
    print("  desert controls in that size bin. A CI spanning zero means it does")
    print("  NOT -- and the smallest bin where it still excludes zero is this")
    print("  method's practical detection limit. That limit is a RESULT: it is")
    print("  the number defining what the project can honestly claim.")
    print("  Every n here is small, so read the intervals, not the point values.")
    return 0


def main(config_path: str = "config.yaml", sanity_mode: bool = False) -> int:
    cfg = load_config(config_path)
    if sanity_mode:
        return sanity(cfg)
    print("Full extraction not implemented yet -- run the sanity check first:")
    print("    python run.py s2 --sanity")
    return 1


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sanity_mode="--sanity" in sys.argv))
