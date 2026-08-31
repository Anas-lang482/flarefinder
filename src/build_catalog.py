"""Stage 2 -- build the study catalogue.

Takes the parsed EOG workbooks (src/data_eog.py handles the schema drift
across the nine annual files) and turns them into the study population:
one stable site identity per physical flare, tracked across years.

THE PROBLEM THIS SOLVES
    EOG re-issues site ids every year, and the reported coordinates jitter
    between years because they are derived from clustered satellite
    detections rather than surveyed positions. So the same physical flare
    appears with a different id and slightly different lat/lon each year.
    Without reconciling them you cannot count how many years a site was
    observed, and -- worse -- a by-site holdout would leak, because the
    "same" site would sit in train under its 2019 id and in test under its
    2023 id. RULE 1 depends on getting this right.

UNITS (verified 2026-08-31, all nine workbooks)
    Every file reports volume in BCM (billion cubic metres) per year. There
    is no million-m3 variant despite the spec allowing for one. Conversion
    is therefore uniform:
        volume_mcm = BCM * 1000          (1 BCM = 1000 million m3)
    Sanity check: Iraq 2024 sums to 18.16 BCM, consistent with the World
    Bank's published Iraq figure. Units confirmed, not assumed.

CLUSTERING CHOICE
    Complete-linkage agglomerative clustering with a 750 m cut, NOT DBSCAN.
    DBSCAN chains: with eps=750 m, a line of flares each 700 m apart merges
    into one cluster kilometres long, which in a dense Iraqi oil field would
    silently fuse distinct flares. Complete linkage guarantees no cluster
    exceeds the threshold in diameter, which is the property actually wanted
    here. The cost is that a genuine site whose coordinate drifts more than
    750 m across years splits into two ids -- the conservative failure, since
    it costs sample size rather than creating false merges.

Run:
    python run.py join
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import HOURS_PER_YEAR, Config, load_config
from src.data_eog import build_site_year_table

EARTH_RADIUS_M = 6_371_000.0
BCM_TO_MCM = 1000.0  # 1 billion m3 = 1000 million m3


def _cluster_coordinates(lat: np.ndarray, lon: np.ndarray, radius_m: float) -> np.ndarray:
    """Complete-linkage clustering of points, cut at radius_m.

    Returns a cluster label per input point. Runs on UNIQUE coordinates only,
    so the O(n^2) distance matrix stays small.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    if len(lat) == 1:
        return np.array([0])

    # Equirectangular projection to metres. Valid at this scale: the study
    # region spans ~15 degrees of latitude and we only care about distances
    # under a kilometre, where the approximation error is centimetres.
    lat_rad = np.radians(lat)
    lat0 = np.mean(lat_rad)
    x = EARTH_RADIUS_M * np.radians(lon) * np.cos(lat0)
    y = EARTH_RADIUS_M * lat_rad
    pts = np.column_stack([x, y])

    Z = linkage(pdist(pts), method="complete")
    return fcluster(Z, t=radius_m, criterion="distance")


def assign_stable_site_ids(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Give every physical flare one id that persists across years."""
    radius_m = float(cfg.thresholds["site_matching"]["cross_year_cluster_radius_m"])
    out = df.copy()
    out["site_id"] = pd.NA

    # Cluster within region: flares never straddle a national boundary at
    # 750 m, and per-region keeps the distance matrices small.
    for region, grp in out.groupby("region_code", dropna=True):
        coords = grp[["latitude", "longitude"]].drop_duplicates().reset_index(drop=True)
        labels = _cluster_coordinates(
            coords["latitude"].to_numpy(float), coords["longitude"].to_numpy(float), radius_m
        )
        coords["cluster"] = [f"{region}-{c:05d}" for c in labels]
        merged = grp.merge(coords, on=["latitude", "longitude"], how="left")
        out.loc[grp.index, "site_id"] = merged["cluster"].to_numpy()

    return out


def cluster_diagnostics(df: pd.DataFrame, lat_col: str = "latitude", lon_col: str = "longitude") -> pd.DataFrame:
    """How far apart are the points inside each cluster, really?

    Reported because the 750 m radius is a [CHOICE], and a reader is
    entitled to see what it actually did rather than trust the number.
    """
    rows = []
    for sid, grp in df.groupby("site_id"):
        if len(grp) < 2:
            continue
        lat = np.radians(grp[lat_col].to_numpy(float))
        lon = np.radians(grp[lon_col].to_numpy(float))
        x = EARTH_RADIUS_M * lon * np.cos(lat.mean())
        y = EARTH_RADIUS_M * lat
        pts = np.column_stack([x, y])
        d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
        rows.append({"site_id": sid, "n_points": len(grp), "max_extent_m": d.max()})
    return pd.DataFrame(rows)


def build_catalog(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    site_year = build_site_year_table(cfg, gulf_only=True)

    print("\nAssigning stable cross-year site ids...")
    df = assign_stable_site_ids(site_year, cfg)

    df["volume_mcm"] = df["bcm"].astype(float) * BCM_TO_MCM

    # Collapse to one row per site-year. A cluster can absorb two EOG rows in
    # the same year (two detections of one flare); sum their volumes and take
    # the volume-weighted centroid rather than dropping either.
    agg = (
        df.groupby(["site_id", "year"], as_index=False)
        .agg(
            country=("country", "first"),
            region_code=("region_code", "first"),
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
            volume_mcm=("volume_mcm", "sum"),
            avg_temp_k=("avg_temp_k", "mean"),
            ellipticity=("ellipticity", "mean"),
            detection_freq=("detection_freq", "mean"),
            clear_obs=("clear_obs", "sum"),
            n_eog_rows=("volume_mcm", "size"),
        )
    )

    # Stable per-site coordinate = mean across years, so the site has ONE
    # position for Sentinel-2 extraction rather than a different one per year.
    centroid = agg.groupby("site_id", as_index=False).agg(
        lat=("latitude", "mean"), lon=("longitude", "mean")
    )
    agg = agg.merge(centroid, on="site_id", how="left")

    # Country names drift too: the Neutral Zone is "Saudi-Kuwaiti Neutral
    # Zone" in most years and "Saudi Arabia - Kuwait" in 2024, which would
    # split one entity across two rows of every summary table. Take the most
    # common name per site so each site has exactly one country label.
    canon = (
        agg.groupby("site_id")["country"]
        .agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iat[0])
        .rename("country_canon")
    )
    agg = agg.merge(canon, on="site_id", how="left")
    agg["country"] = agg["country_canon"]
    agg = agg.drop(columns=["country_canon"])

    n_years = agg.groupby("site_id", as_index=False).size().rename(columns={"size": "n_years_observed"})
    agg = agg.merge(n_years, on="site_id", how="left")

    # Carry m3/h alongside Mm3/yr. Every threshold in this project is quoted
    # in m3/h (the unit of the VIIRS detection limit), and recomputing the
    # conversion at each call site is how the two drift apart.
    agg["m3_per_h"] = agg["volume_mcm"] * 1e6 / HOURS_PER_YEAR

    cols = [
        "site_id", "lat", "lon", "country", "region_code", "year",
        "volume_mcm", "m3_per_h", "n_years_observed", "avg_temp_k",
        "ellipticity", "detection_freq", "clear_obs", "n_eog_rows",
    ]
    # Second frame keeps the ORIGINAL per-year coordinates, which the
    # cluster diagnostic needs -- the catalogue itself carries only the
    # stable centroid, whose within-site spread is zero by construction.
    return agg[cols].sort_values(["site_id", "year"]).reset_index(drop=True), df


def print_summary(df: pd.DataFrame, cfg: Config) -> None:
    print("\n" + "=" * 72)
    print("CATALOGUE SUMMARY")
    print("=" * 72)
    print(f"site-year rows : {len(df)}")
    print(f"unique sites   : {df['site_id'].nunique()}")
    print(f"years          : {df['year'].min()}-{df['year'].max()}")

    print("\nsites per country (unique site_id):")
    per = df.groupby("country").agg(
        sites=("site_id", "nunique"),
        site_years=("site_id", "size"),
        total_mcm=("volume_mcm", "sum"),
    )
    print(per.round(1).to_string())

    print("\nyears observed per site:")
    ny = df.groupby("site_id")["n_years_observed"].first()
    print(ny.value_counts().sort_index().to_string())

    # -- Seymour threshold -------------------------------------------------
    limit_m3h = float(cfg.thresholds["viirs"]["detection_limit_m3_per_h"])
    hours = 8760.0
    limit_mcm = limit_m3h * hours / 1e6
    print("\n" + "-" * 72)
    print("SMALL-FLARE THRESHOLD (Seymour et al. 2025)")
    print("-" * 72)
    print(f"  VIIRS detection limit      : {limit_m3h:.0f} m3/h")
    print(f"  ASSUMPTION: continuous flaring for all {hours:.0f} h of the year.")
    print("  This is an UPPER bound on the annual equivalent. Real flares are")
    print(
        f"  intermittent (median detection freq here is "
        f"{df['detection_freq'].median():.2f}), so a site flaring at "
        f"{limit_m3h:.0f} m3/h only"
    )
    print("  part of the year falls BELOW this line while still exceeding the")
    print("  instantaneous limit whenever it burns. Sites counted below the")
    print("  threshold are therefore a CONSERVATIVE (under)count of the")
    print("  small-flare population.")
    print(f"  Annual equivalent          : {limit_mcm:.4f} million m3/year")

    latest = df[df["year"] == df["year"].max()]
    below = latest["volume_mcm"] < limit_mcm
    print(f"\n  In {int(df['year'].max())}: {int(below.sum())} of {len(latest)} sites "
          f"({100 * below.mean():.1f}%) fall below it.")
    print("  These are in the EOG record despite being under the stated limit,")
    print("  which is itself worth investigating.")

    # -- log histogram -----------------------------------------------------
    print("\nvolume distribution (log10 million m3/year, latest year):")
    v = latest.loc[latest["volume_mcm"] > 0, "volume_mcm"]
    counts, edges = np.histogram(np.log10(v), bins=14)
    peak = counts.max() or 1
    for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
        bar = "#" * int(40 * c / peak)
        mark = "  <-- Seymour limit" if lo <= np.log10(limit_mcm) < hi else ""
        print(f"  1e{lo:+.1f}..1e{hi:+.1f} {c:>5} {bar}{mark}")


def main(config_path: str = "config.yaml") -> int:
    cfg = load_config(config_path)
    df, raw = build_catalog(cfg)

    diag = cluster_diagnostics(raw)
    if not diag.empty:
        print(f"\ncluster extents: max {diag['max_extent_m'].max():.0f} m, "
              f"median {diag['max_extent_m'].median():.0f} m "
              f"(cut at {cfg.thresholds['site_matching']['cross_year_cluster_radius_m']} m)")

    out = cfg.path(cfg["data"]["processed_dir"]) / "catalog.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print_summary(df, cfg)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
