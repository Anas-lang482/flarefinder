"""Stage 3 -- VIIRS Nightfire (VNF) per-detection ingest.

WHY THIS STAGE IS THE BIGGEST UPGRADE IN THE PROJECT
    The EOG annual workbooks give ONE row per site per year. VNF gives one row
    per satellite overpass, with radiant heat, source temperature and source
    area. Two things follow:

    1. It makes the RULE 6 baseline real. EOG's published calibration
       multiplies RADIANT HEAT by a fitted slope (0.029353, visible in the
       workbook filenames). Radiant heat is absent from the per-site sheets,
       so src/baseline.py could only fit a detection-frequency regression and
       has to carry a caveat that removes most of its force. VNF removes it.

    2. It turns intermittency from a single annual fraction into an actual
       TIME SERIES. Flares flicker across passes; sun-heated desert follows
       the sun. That contrast is this project's whole discriminator, and one
       number per year cannot express it.

ACCESS -- read before running
    VNF is NOT public, unlike the annual flare workbooks. Verified 2026-09-01:
    every path under /wwwdata/viirs_products/vnf/ returns a 302 to
    eogauth.mines.edu. You need a bearer token.

    You mint the token yourself -- this module never sees your password and
    must never be given it. See docs/PLAYBOOK.md Step 5, or run:
        python run.py vnf --probe
    which prints the exact command and checks whether a token is present.

    Store it in .env (gitignored) as:
        EOG_ACCESS_TOKEN=<token>

    Tokens are short-lived. If a download starts returning HTML login pages
    instead of gzip, the token expired -- mint a new one.

    NOTE: since 2025-01-10 VNF is covered by a VIIRS Nightfire Data Use
    License. Accepting it is your decision to make on the EOG site, not
    something this code does for you.

DESIGN -- built around the two constraints that matter
    CACHE EVERYTHING. Nightly files are global; re-downloading a night you
    already have is pure waste. Any file already on disk is never fetched
    again. Delete a file to force a refetch.

    FAIL LOUDLY ON A WRONG PATH. The last documented source for this data
    (s3://blackmarble-combustion) did not exist, and that was only caught
    because someone probed it. This module verifies it received gzip, not an
    HTML login page, and says which it got.

Run:
    python run.py vnf --probe              # check token + one night
    python run.py vnf                      # full configured range
"""

from __future__ import annotations

import gzip
import io
import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.config import Config, load_config

TOKEN_URL = "https://eogauth.mines.edu/realms/eog/protocol/openid-connect/token"
EARTH_RADIUS_M = 6_371_000.0

# Columns we care about, matched case-insensitively -- the VNF schema has
# drifted across versions exactly as the annual workbooks did, so nothing is
# matched by exact name.
WANTED = {
    "lat": ["lat_gmtco", "latitude", "lat"],
    "lon": ["lon_gmtco", "longitude", "lon"],
    "radiant_heat": ["rh", "radiant_heat", "rhi"],
    "temp_bb": ["temp_bb", "temperature_bb", "temp"],
    "area_bb": ["area_bb", "source_area", "area"],
    "esf_bb": ["esf_bb"],
    "cloud_mask": ["cloud_mask", "cm"],
    "datetime": ["date_ltz", "date_mscan", "datetime", "date"],
}


def token_help() -> str:
    return (
        "\nNo EOG_ACCESS_TOKEN found.\n"
        "\nMint one yourself -- do NOT give your password to this script.\n"
        "Run this in PowerShell, substituting your own EOG account details:\n"
        "\n"
        '  $body = @{ username="YOUR_EMAIL"; password="YOUR_PASSWORD";\n'
        '             client_id="eogdata_oidc";\n'
        '             client_secret="2677ce81-f1fd-44e6-9c4d-1a09c3fbb5c5";\n'
        '             grant_type="password" }\n'
        f'  (Invoke-RestMethod -Method Post -Uri "{TOKEN_URL}" -Body $body).access_token\n'
        "\n"
        "Copy the printed token into a file called .env in the project root:\n"
        "\n"
        "  EOG_ACCESS_TOKEN=eyJhbGciOi...\n"
        "\n"
        ".env is gitignored, so the token will not be committed.\n"
        "Tokens expire after a few hours -- re-mint when downloads start\n"
        "returning HTML instead of gzip.\n"
    )


def load_token() -> str | None:
    """Environment first, then .env. Never prompts, never stores a password."""
    tok = os.environ.get("EOG_ACCESS_TOKEN")
    if tok:
        return tok.strip()
    env = Path(__file__).resolve().parent.parent / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("EOG_ACCESS_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def nightly_urls(day: date, cfg: Config) -> list[str]:
    """Candidate URLs for one night.

    A list, not a single string: the VNF layout varies by version and
    satellite, and which variant exists is not documented reliably enough to
    hardcode. The first one that returns gzip wins, and the winner is
    reported so config can be pinned to it.
    """
    v = cfg["data"]["viirs"].get("vnf_version", "v30")
    base = cfg["data"]["viirs"].get(
        "vnf_base_url", "https://eogdata.mines.edu/wwwdata/viirs_products/vnf"
    )
    d = day.strftime("%Y%m%d")
    vt = v.replace(".", "")
    out = []
    for sat in ("npp", "j01", "noaa20"):
        out += [
            f"{base}/{v}/rearrange/{day:%Y}/VNF_{sat}_d{d}_noaa_{vt}.csv.gz",
            f"{base}/{v}/rearrange/VNF_{sat}_d{d}_noaa_{vt}.csv.gz",
            f"{base}/{v}/{day:%Y}/VNF_{sat}_d{d}_noaa_{vt}.csv.gz",
        ]
    return out


def _looks_like_gzip(content: bytes) -> bool:
    return len(content) > 2 and content[0] == 0x1F and content[1] == 0x8B


def fetch_night(day: date, cfg: Config, token: str, raw_dir: Path) -> Path | None:
    """Download one night, cached. Returns the local path, or None if absent.

    Distinguishes three outcomes that look alike if you only check the status
    code: real data, an expired token (HTML login page with a 200), and a
    genuinely missing night.
    """
    dest = raw_dir / f"VNF_{day:%Y%m%d}.csv.gz"
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    headers = {"Authorization": f"Bearer {token}"}
    for url in nightly_urls(day, cfg):
        try:
            r = requests.get(url, headers=headers, timeout=180)
        except Exception:
            continue
        if r.status_code != 200:
            continue
        if _looks_like_gzip(r.content):
            dest.write_bytes(r.content)
            print(f"    {day} OK  {len(r.content)/1e6:6.2f} MB  {url.rsplit('/', 1)[-1]}")
            return dest
        # 200 but not gzip: almost always the auth redirect landing page.
        if b"<html" in r.content[:400].lower():
            raise PermissionError(
                "Server returned an HTML page instead of data. The token is "
                "missing, expired, or the Data Use License has not been "
                "accepted on the EOG site." + token_help()
            )
    return None


def read_night(path: Path, bbox: tuple[float, float, float, float]) -> pd.DataFrame:
    """Read one nightly file and keep only rows inside the study bbox."""
    with gzip.open(path, "rb") as fh:
        df = pd.read_csv(io.BytesIO(fh.read()), low_memory=False)

    lower = {str(c).strip().lower(): c for c in df.columns}
    cols = {}
    for canon, aliases in WANTED.items():
        for a in aliases:
            if a in lower:
                cols[canon] = lower[a]
                break
    if "lat" not in cols or "lon" not in cols:
        raise ValueError(f"no lat/lon column in {path.name}: {list(df.columns)[:15]}")

    out = pd.DataFrame({k: df[v] for k, v in cols.items()})
    lon_min, lat_min, lon_max, lat_max = bbox
    m = (
        out["lat"].between(lat_min, lat_max)
        & out["lon"].between(lon_min, lon_max)
    )
    out = out[m].copy()
    out["source_file"] = path.name
    return out


def region_bbox(cfg: Config) -> tuple[float, float, float, float]:
    """Union bbox over the active regions."""
    boxes = [r["bbox"] for r in cfg.active_regions()]
    return (
        min(b[0] for b in boxes), min(b[1] for b in boxes),
        max(b[2] for b in boxes), max(b[3] for b in boxes),
    )


def match_to_sites(det: pd.DataFrame, catalog: pd.DataFrame, radius_m: float) -> pd.DataFrame:
    """Attach the nearest catalogue site_id within radius_m, else NA.

    Detections with no match are KEPT, not dropped: a nighttime detection with
    no catalogue entry is a candidate uncatalogued flare, which is the whole
    point of the project. Dropping them would discard the signal.
    """
    sites = catalog.drop_duplicates("site_id")[["site_id", "lat", "lon"]].reset_index(drop=True)
    if det.empty or sites.empty:
        det["site_id"] = pd.NA
        return det

    lat0 = np.radians(float(sites["lat"].mean()))
    to_x = lambda lon: EARTH_RADIUS_M * np.radians(lon) * np.cos(lat0)
    to_y = lambda lat: EARTH_RADIUS_M * np.radians(lat)

    sx, sy = to_x(sites["lon"].to_numpy(float)), to_y(sites["lat"].to_numpy(float))
    dx, dy = to_x(det["lon"].to_numpy(float)), to_y(det["lat"].to_numpy(float))

    from scipy.spatial import cKDTree

    tree = cKDTree(np.column_stack([sx, sy]))
    dist, idx = tree.query(np.column_stack([dx, dy]), distance_upper_bound=radius_m)
    hit = np.isfinite(dist)
    det = det.copy()
    det["site_id"] = pd.Series(
        np.where(hit, sites["site_id"].to_numpy()[np.clip(idx, 0, len(sites) - 1)], None),
        index=det.index, dtype="object",
    )
    det["match_dist_m"] = np.where(hit, dist, np.nan)
    return det


def main(config_path: str = "config.yaml", probe: bool = False) -> int:
    cfg = load_config(config_path)
    raw_dir = cfg.path(cfg["data"]["raw_dir"], cfg["data"]["viirs"]["raw_subdir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    token = load_token()
    print("=" * 74)
    print("VNF NIGHTTIME INGEST" + ("  (probe)" if probe else ""))
    print("=" * 74)
    if not token:
        print(token_help())
        return 1
    print(f"token       : present ({len(token)} chars)")
    print(f"cache dir   : {raw_dir}")

    bbox = region_bbox(cfg)
    print(f"study bbox  : {[round(b, 2) for b in bbox]}")

    vcfg = cfg["data"]["viirs"]
    if probe:
        days = [date.fromisoformat(vcfg.get("probe_date", "2024-01-15"))]
    else:
        start = date.fromisoformat(str(vcfg["start_date"]))
        end = date.fromisoformat(str(vcfg["end_date"]))
        days = [start + timedelta(d) for d in range((end - start).days + 1)]
    print(f"nights      : {len(days)} ({days[0]} .. {days[-1]})\n")

    frames, missing = [], 0
    for day in days:
        try:
            p = fetch_night(day, cfg, token, raw_dir)
        except PermissionError as e:
            print(f"\nFAILED: {e}")
            return 1
        if p is None:
            missing += 1
            if probe:
                print(f"    {day} NOT FOUND at any candidate URL")
            continue
        frames.append(read_night(p, bbox))

    if not frames:
        print(
            f"\nNo data retrieved ({missing} nights not found).\n"
            "The URL pattern is probably wrong for your VNF version. Log in at\n"
            "https://eogdata.mines.edu/products/vnf/ , copy the real path of one\n"
            "nightly file, and set data.viirs.vnf_base_url / vnf_version in\n"
            "config.yaml to match. Do NOT guess -- verify one real file first."
        )
        return 1

    det = pd.concat(frames, ignore_index=True)
    print(f"\ndetections in study bbox : {len(det)}  ({missing} nights unavailable)")

    cat_path = cfg.path(cfg["data"]["processed_dir"], "catalog.parquet")
    if cat_path.exists():
        catalog = pd.read_parquet(cat_path)
        radius = float(cfg.thresholds["site_matching"]["match_radius_m"])
        det = match_to_sites(det, catalog, radius)
        matched = int(det["site_id"].notna().sum())
        print(f"matched to catalogue     : {matched} ({100*matched/len(det):.1f}%) within {radius:.0f} m")
        print(f"UNMATCHED                : {len(det)-matched} -- candidate uncatalogued flares")

    if "radiant_heat" in det.columns:
        rh = pd.to_numeric(det["radiant_heat"], errors="coerce").dropna()
        if len(rh):
            print(f"\nradiant heat (MW)        : median {rh.median():.3f}  max {rh.max():.1f}")
            print("  ^ this column is what makes the RULE 6 baseline reproducible")

    out = cfg.path(cfg["data"]["processed_dir"]) / "vnf_detections.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    det.to_parquet(out, index=False)
    print(f"\nwrote {out}  rows={len(det)}")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(probe="--probe" in sys.argv))
