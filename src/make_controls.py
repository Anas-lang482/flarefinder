"""Stage 4a -- control points.

Every detector needs negatives. Controls are locations that are NOT flares
but sit in the same desert, at the same latitudes, under the same sun and the
same atmosphere as the real sites. If the controls were drawn from anywhere
else, the model would learn "Gulf oil field vs everywhere else" -- which is
trivial, and useless.

THE THREE CONSTRAINTS, AND WHY EACH ONE EXISTS
    5-15 km from the paired site
        Close enough to share terrain, climate and land cover. Far enough
        that a 20 m Sentinel-2 pixel at the control cannot catch light from
        the flare itself. Too close and the negatives are contaminated; too
        far and the comparison stops being like-for-like.

    >= 2 km from ANY known flare, any country, ANY YEAR
        The year part matters. A site that flared in 2015 and went quiet by
        2020 is still not a clean negative -- there may be residual
        infrastructure, and it may resume. Checking only the current year
        would quietly poison the negative set.

    onshore
        Water is cold and dark in SWIR, so a sea control is trivially
        separable and would flatter the detector. The whole difficulty of
        this project is that hot desert looks like a flare; a control set
        that skips that difficulty measures nothing.

RANDOMNESS
    Seeded from seeds.split, not seeds.model. Control placement is part of
    how the data is partitioned, not part of fitting -- and reusing a seed
    across those two roles is how a subtle dependence gets introduced.

Run:
    python run.py controls
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Config, load_config

EARTH_RADIUS_M = 6_371_000.0


def _to_metres(lat: np.ndarray, lon: np.ndarray, lat0: float):
    """Local equirectangular projection. Fine at these distances."""
    return (
        EARTH_RADIUS_M * np.radians(lon) * np.cos(lat0),
        EARTH_RADIUS_M * np.radians(lat),
    )


def propose_controls(
    sites: pd.DataFrame, cfg: Config, per_site: int, oversample: int = 6
) -> pd.DataFrame:
    """Propose candidates at a random bearing and distance from each site.

    Oversampled on purpose: many candidates will be rejected for landing near
    another flare, which happens constantly in a dense field like Rumaila.
    Generating exactly `per_site` and hoping would silently under-produce
    controls for precisely the densest, most interesting areas.
    """
    cc = cfg["controls"]
    rng = np.random.default_rng(int(cfg.seeds["split"]))
    d_min, d_max = float(cc["min_distance_m"]), float(cc["max_distance_m"])

    n = len(sites) * per_site * oversample
    site_idx = np.repeat(np.arange(len(sites)), per_site * oversample)
    bearing = rng.uniform(0, 2 * np.pi, n)
    # sqrt keeps candidates uniform over the annulus AREA rather than
    # bunching them at the inner radius.
    dist = np.sqrt(rng.uniform(d_min**2, d_max**2, n))

    lat0 = np.radians(float(sites["lat"].mean()))
    slat = sites["lat"].to_numpy(float)[site_idx]
    slon = sites["lon"].to_numpy(float)[site_idx]
    dlat = np.degrees(dist * np.cos(bearing) / EARTH_RADIUS_M)
    dlon = np.degrees(dist * np.sin(bearing) / (EARTH_RADIUS_M * np.cos(lat0)))

    return pd.DataFrame({
        "paired_site_id": sites["site_id"].to_numpy()[site_idx],
        "region_code": sites["region_code"].to_numpy()[site_idx],
        "lat": slat + dlat,
        "lon": slon + dlon,
        "offset_m": dist,
    })


def drop_near_any_flare(cand: pd.DataFrame, all_sites: pd.DataFrame, min_m: float) -> pd.DataFrame:
    """Reject candidates within min_m of ANY flare, any country, any year."""
    from scipy.spatial import cKDTree

    lat0 = np.radians(float(all_sites["lat"].mean()))
    sx, sy = _to_metres(all_sites["lat"].to_numpy(float), all_sites["lon"].to_numpy(float), lat0)
    cx, cy = _to_metres(cand["lat"].to_numpy(float), cand["lon"].to_numpy(float), lat0)

    tree = cKDTree(np.column_stack([sx, sy]))
    dist, _ = tree.query(np.column_stack([cx, cy]))
    out = cand.copy()
    out["dist_to_nearest_flare_m"] = dist
    return out[dist >= min_m].copy()


def land_flags(points: pd.DataFrame, cfg: Config) -> np.ndarray:
    """True where the point is on land, from an Earth Engine land mask."""
    import ee

    ee.Initialize(project=cfg["data"]["earthengine"]["project"])
    land = ee.Image("NOAA/NGDC/ETOPO1").select("bedrock").gt(0).rename("is_land")

    keep = np.zeros(len(points), dtype=bool)
    chunk = int(cfg["controls"].get("ee_chunk_size", 2000))
    for start in range(0, len(points), chunk):
        part = points.iloc[start : start + chunk]
        fc = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]))
            for r in part.itertuples()
        ])
        vals = land.reduceRegions(fc, ee.Reducer.first(), 1000).aggregate_array("first").getInfo()
        keep[start : start + len(part)] = np.array([bool(v) for v in vals])
        print(f"    land check {start + len(part):>6}/{len(points)}  on land so far {keep.sum()}")
    return keep


def keep_onshore(cand: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Drop candidates that fall in water, using an Earth Engine land mask.

    Cheap in EECU terms: one sample of one static image per point. Done in
    chunks because a single getInfo on tens of thousands of points will time
    out rather than fail cleanly.
    """
    import ee

    ee.Initialize(project=cfg["data"]["earthengine"]["project"])

    # ETOPO1 bedrock elevation: > 0 is land. Chosen over a landcover product
    # because it is static, global, tiny to sample, and has no class-scheme
    # ambiguity at coastlines.
    land = ee.Image("NOAA/NGDC/ETOPO1").select("bedrock").gt(0).rename("is_land")

    keep = np.zeros(len(cand), dtype=bool)
    chunk = int(cfg["controls"].get("ee_chunk_size", 2000))
    for start in range(0, len(cand), chunk):
        part = cand.iloc[start : start + chunk]
        fc = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]))
            for r in part.itertuples()
        ])
        vals = land.reduceRegions(fc, ee.Reducer.first(), 1000).aggregate_array("first").getInfo()
        keep[start : start + len(part)] = np.array([bool(v) for v in vals])
        print(f"    land check {start + len(part):>6}/{len(cand)}  kept so far {keep.sum()}")

    return cand[keep].copy()


def main(config_path: str = "config.yaml") -> int:
    cfg = load_config(config_path)
    cat_path = cfg.path(cfg["data"]["processed_dir"], "catalog.parquet")
    if not cat_path.exists():
        print(f"missing {cat_path} -- run `python run.py join` first")
        return 1

    catalog = pd.read_parquet(cat_path)
    sites = catalog.drop_duplicates("site_id")[["site_id", "region_code", "lat", "lon"]].copy()
    cc = cfg["controls"]
    per_site = int(cc["per_site"])

    # Tag every SITE onshore/offshore and persist it to the catalogue.
    # config declares site_domain.include_offshore: false, but nothing was
    # ever enforcing it -- the catalogue carried offshore Gulf sites the whole
    # time. This surfaced only because control generation could not place a
    # single land point around them. Offshore sites are FLAGGED, not deleted:
    # CLAUDE.md requires them reported separately if ever included, and a
    # silent delete would make that impossible.
    if "is_onshore" not in catalog.columns:
        print()
        print("tagging sites onshore/offshore via Earth Engine land mask...")
        sites["is_onshore"] = land_flags(sites, cfg)
        catalog = catalog.merge(sites[["site_id", "is_onshore"]], on="site_id", how="left")
        catalog.to_parquet(cat_path, index=False)
        print(f"  wrote is_onshore back to {cat_path.name}")
    else:
        sites = sites.merge(
            catalog.drop_duplicates("site_id")[["site_id", "is_onshore"]], on="site_id", how="left"
        )

    n_off = int((~sites["is_onshore"].astype(bool)).sum())
    print()
    print(f"sites onshore : {len(sites) - n_off}")
    print(f"sites OFFSHORE: {n_off}  -- excluded, per site_domain.include_offshore: false")
    if not cfg["site_domain"]["include_offshore"]:
        sites = sites[sites["is_onshore"].astype(bool)].copy()

    print("=" * 74)
    print("CONTROL POINT GENERATION")
    print("=" * 74)
    print(f"flare sites            : {len(sites)}")
    print(f"controls wanted        : {per_site} per site = {len(sites) * per_site}")
    print(f"offset from paired site: {cc['min_distance_m']/1000:.0f}-{cc['max_distance_m']/1000:.0f} km")
    print(f"exclusion radius       : {cc['min_distance_to_any_flare_m']/1000:.0f} km from ANY flare, any year")

    cand = propose_controls(sites, cfg, per_site)
    print(f"\ncandidates proposed    : {len(cand)}")

    cand = drop_near_any_flare(cand, sites, float(cc["min_distance_to_any_flare_m"]))
    print(f"after flare-proximity  : {len(cand)}")

    if cc.get("require_onshore", True):
        print("\nchecking land mask via Earth Engine...")
        cand = keep_onshore(cand, cfg)
        print(f"after onshore filter   : {len(cand)}")

    # Take the requested number per site, from what survived.
    cand = cand.sample(frac=1.0, random_state=int(cfg.seeds["split"]))
    controls = cand.groupby("paired_site_id", group_keys=False).head(per_site).copy()
    controls = controls.sort_values(["paired_site_id"]).reset_index(drop=True)
    controls["control_id"] = [f"CTL-{i:06d}" for i in range(len(controls))]
    controls = controls[
        ["control_id", "paired_site_id", "region_code", "lat", "lon",
         "offset_m", "dist_to_nearest_flare_m"]
    ]

    got = controls.groupby("paired_site_id").size()
    short = int((got < per_site).sum())
    print(f"\ncontrols produced      : {len(controls)}")
    print(f"sites with a full set  : {int((got == per_site).sum())} of {len(sites)}")
    if short:
        print(f"sites SHORT of {per_site}       : {short}")
        print("  Expected in dense fields, where most of the annulus is within")
        print("  2 km of another flare. Not an error -- but if it is a large")
        print("  fraction, the negative set under-represents dense areas, which")
        print("  are exactly where false positives are most likely.")

    print("\nper region:")
    print(controls.groupby("region_code").size().to_string())
    print(f"\nnearest-flare distance: min {controls['dist_to_nearest_flare_m'].min():.0f} m, "
          f"median {controls['dist_to_nearest_flare_m'].median():.0f} m")

    out = cfg.path(cfg["data"]["processed_dir"]) / "controls.parquet"
    controls.to_parquet(out, index=False)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
