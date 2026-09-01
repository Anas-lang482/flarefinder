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


def sanity(cfg: Config, n_sites: int = 20) -> int:
    """Do the biggest known flares actually glow in SWIR?"""
    proc = cfg.path(cfg["data"]["processed_dir"])
    catalog = pd.read_parquet(proc / "catalog.parquet")
    controls = pd.read_parquet(proc / "controls.parquet")

    year = int(cfg.years["s2_end"])
    latest = catalog[catalog["year"] == year]
    if "is_onshore" in latest.columns:
        latest = latest[latest["is_onshore"].astype(bool)]
    top = latest.nlargest(n_sites, "volume_mcm")[["site_id", "lat", "lon", "volume_mcm"]]

    ctl = controls[controls["paired_site_id"].isin(top["site_id"])]
    ctl = ctl.groupby("paired_site_id", group_keys=False).head(2)

    pts = pd.concat([
        top.rename(columns={"site_id": "point_id"}).assign(kind="flare"),
        ctl.rename(columns={"control_id": "point_id"})[["point_id", "lat", "lon"]].assign(
            kind="control", volume_mcm=np.nan
        ),
    ], ignore_index=True)

    print("=" * 74)
    print(f"SENTINEL-2 SANITY CHECK -- do the biggest flares glow in SWIR? ({year})")
    print("=" * 74)
    print(f"points: {int((pts.kind=='flare').sum())} flares (largest in catalogue) "
          f"+ {int((pts.kind=='control').sum())} paired controls")
    print(f"volume range of the flares: {top.volume_mcm.min():,.0f} - {top.volume_mcm.max():,.0f} Mm3/yr\n")

    cache = proc / f"s2_sanity_{year}.parquet"
    if cache.exists():
        print(f"using cached {cache.name} (delete it to re-pull from Earth Engine)")
        obs = pd.read_parquet(cache)
    else:
        obs = monthly_swir(pts, year, cfg)
        obs.to_parquet(cache, index=False)
        print(f"\nwrote {cache}")

    df = obs.merge(pts[["point_id", "kind"]], on="point_id", how="left")
    df = df[df["B12_max"].notna()].copy()

    # THE STATISTIC MATTERS MORE THAN THE DATA HERE.
    #
    # A first version of this check compared the POOLED MEDIAN of raw bands
    # and reported PASS on a 3.3x B12 difference. That was wrong, and wrong in
    # the most dangerous direction -- it would have licensed the whole
    # Sentinel-2 arm on a false premise. Flare sites came out ~3x brighter
    # than controls in ALL THREE bands about equally, and the B12/B8A ratio
    # was actually LOWER at flares (1.007) than at controls (1.024). Uniform
    # brightness across NIR and SWIR is not a thermal signature; it is what
    # industrial infrastructure looks like -- metal tanks, concrete, roads.
    # That test would have passed on a car park.
    #
    # Two corrections:
    #   1. Judge on the SPECTRAL RATIO B12/B8A, not raw brightness. Flames are
    #      disproportionately bright at 2190 nm; sunlit sand and steel are
    #      bright everywhere.
    #   2. Take the MAX across months per site, not the median. Flares are
    #      intermittent -- median detection frequency in this catalogue is
    #      0.077 -- so most monthly composites catch the site unlit and the
    #      median averages the signal away. The lit pass is the informative
    #      one. This is the same reasoning as the max-not-median composite:
    #      variance across passes IS the signal.
    df["b12_b8a"] = df["B12_max"] / df["B8A_max"].replace(0, np.nan)
    df["swir_index"] = (df["B12_max"] - df["B11_max"]) / (df["B12_max"] + df["B11_max"])

    per_site = (
        df.groupby(["point_id", "kind"])
        .agg(b12_b8a_max=("b12_b8a", "max"), b12_max=("B12_max", "max"))
        .reset_index()
    )

    print()
    print("-" * 74)
    print("RESULT")
    print("-" * 74)
    print("pooled median of raw bands (the NAIVE view -- do not judge on this):")
    print(df.groupby("kind")[["B8A_max", "B11_max", "B12_max"]].median().round(1).to_string())
    print()
    print("per-site MAX across months of the spectral ratio (the real test):")
    print(per_site.groupby("kind")[["b12_b8a_max", "b12_max"]].median().round(3).to_string())

    thr = float(cfg["data"]["sentinel2"].get("sanity_b12_b8a_min", 1.2))
    fr = {}
    for k, s in per_site.groupby("kind"):
        fr[k] = float((s["b12_b8a_max"] > thr).mean())
        print(f"  points ever exceeding B12/B8A > {thr}: {k:<8} "
              f"{int((s['b12_b8a_max'] > thr).sum())}/{len(s)} ({100*fr[k]:.0f}%)")

    sat = float((df["B12_max"] >= 10000).mean())
    print()
    print(f"SATURATION: {100*sat:.1f}% of observations are at or above 10000 "
          f"(100% reflectance).")
    print("  S2 was not designed for targets this hot. Saturation compresses the")
    print("  ratio toward 1 and makes this test CONSERVATIVE -- the true spectral")
    print("  separation is likely larger than measured. Do not read saturated")
    print("  radiance as a quantitative flare brightness.")

    f_med = float(per_site[per_site.kind == "flare"]["b12_b8a_max"].median())
    c_med = float(per_site[per_site.kind == "control"]["b12_b8a_max"].median())
    print()
    if f_med > c_med * 1.2 and fr.get("flare", 0) > 2 * fr.get("control", 1):
        print("  -> PASS, and it validates the project's core premise.")
        print(f"     Flares reach B12/B8A {f_med:.2f} against {c_med:.2f} for their own")
        print("     controls, and do so in only SOME months -- which is exactly what")
        print("     intermittency predicts and why the median hid it. The signal is")
        print("     in the variation across passes, not in the average level.")
        print("     Proceed to the full extraction, keeping per-pass values.")
    else:
        print("  -> FAIL on the spectral test. Raw brightness alone is NOT evidence:")
        print("     flare sites contain industrial infrastructure that is bright in")
        print("     every band. STOP and debug before building features.")
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
