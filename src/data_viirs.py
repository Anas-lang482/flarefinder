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
# Verified against the real v3.0 schema on 2026-09-03, using the sample file
# and README that EOG publishes publicly on the VNF product page. Matched
# case-insensitively because the schema varies across VNF versions.
WANTED = {
    "lat": ["lat_gmtco", "latitude", "lat"],
    "lon": ["lon_gmtco", "longitude", "lon"],
    "site_key": ["id_key"],          # EOG unique IR-source id
    "datetime": ["date_ltz", "date_mscan", "datetime", "date"],
    "temp_bb": ["temp_bb"],          # source temperature, Kelvin
    "temp_bkg": ["temp_bkg"],        # Earth background temperature
    "esf_bb": ["esf_bb"],            # emission scaling factor
    "radiant_heat": ["rh"],          # RADIANT HEAT -- the reason for the licence
    "rhi": ["rhi"],                  # radiant heat intensity
    "area_bb": ["area_bb"],          # source area
    "area_pixel": ["area_pixel"],
    "methane_eq": ["methane_eq"],    # EOG computes these per detection --
    "co2_eq": ["co2_eq"],            # directly useful for the CO2e multiplier
    "cloud_mask": ["cloud_mask", "cm"],
    "qf_fit": ["qf_fit"],            # Planck-fit quality bitfield
    "qf_detect": ["qf_detect"],      # detection-method bitfield
}

# The VNF fill value. README v3.0 lists 999999 as the Fill Value for
# Lat_GMTCO, Lon_GMTCO, Temp_BB, Temp_Bkg, ESF_BB, RHI, RH and others.
# Left unconverted it is catastrophic: a median radiant heat computed over
# a column where most rows are 999999 returns 999999 and looks like a real
# number. Every numeric column is masked on read.
FILL_VALUE = 999999

# QF_Fit bits meaning the Planck fit hit a boundary, so the temperature is
# not trustworthy: 8 = Max Temp Fit, 16 = Min Temp Fit.
QF_FIT_BOUNDARY_BITS = 8 | 16


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
        # utf-8-sig, not utf-8: PowerShell redirection and Out-File often
        # write a UTF-8 BOM, which would make the first line start with
        # ﻿ and silently fail the prefix match below -- the token would
        # look absent while sitting right there in the file.
        for line in env.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip().lstrip("﻿")
            if line.startswith("#") or not line:
                continue
            if line.startswith("EOG_ACCESS_TOKEN"):
                val = line.split("=", 1)[1] if "=" in line else ""
                return val.strip().strip('"').strip("'")
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
    """Read one VNF file, mask fill values, keep rows inside the study bbox.

    Handles .csv and .csv.gz. Column matching is case-insensitive because the
    schema differs across VNF versions.
    """
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rb") as fh:
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

    # Mask the fill value BEFORE anything else touches these numbers.
    # 999999 is a real float that survives every downstream operation
    # silently: a median over a mostly-unfitted column returns 999999 and
    # reads as a plausible radiant heat. This is the single most dangerous
    # line in the module.
    numeric = [c for c in out.columns if c not in ("site_key", "datetime")]
    for c in numeric:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        out.loc[out[c] == FILL_VALUE, c] = pd.NA

    lon_min, lat_min, lon_max, lat_max = bbox
    out = out[
        out["lat"].between(lat_min, lat_max) & out["lon"].between(lon_min, lon_max)
    ].copy()

    # Flag boundary-limited Planck fits. Temperature and everything derived
    # from it (radiant heat, area) are unreliable where the fit was clipped
    # at the allowed min or max, so mark them rather than dropping them --
    # whether to exclude is an analysis decision, not a parsing one.
    if "qf_fit" in out.columns:
        qf = pd.to_numeric(out["qf_fit"], errors="coerce").fillna(0).astype("int64")
        out["fit_at_boundary"] = (qf & QF_FIT_BOUNDARY_BITS) > 0
    else:
        out["fit_at_boundary"] = False

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


def ingest_local(cfg: Config, raw_dir: Path) -> pd.DataFrame:
    """Parse VNF files already on disk.

    THE PRIMARY PATH, not a fallback. Verified 2026-09-03 on EOG's own
    registration page: since 2026-06-01 programmatic access via OpenID client
    is restricted to PAID subscribers, and the client_id/client_secret are
    issued by EOG after payment. The public client that older documentation
    named (eogdata_oidc) returns invalid_client on both auth hosts.

    An academic data licence therefore grants the DATA but not the API. Files
    downloaded through the browser while logged in are the supported route,
    and this function is what turns them into the project's table.

    Drop any VNF nightly file into data/raw/viirs/ -- .csv or .csv.gz, any
    naming -- and run `python run.py vnf`.
    """
    # Deduplicate by resolved path. Windows globbing is case-INSENSITIVE, so
    # "*.csv.gz" and "*.CSV.GZ" both match the same file and the naive
    # concatenation parsed every file twice -- which doubled the detection
    # count and would have doubled every downstream statistic while looking
    # entirely plausible.
    patterns = ("*.csv.gz", "*.csv", "*.CSV.GZ", "*.CSV")
    seen, files = set(), []
    for pat in patterns:
        for f in sorted(raw_dir.glob(pat)):
            key = str(f.resolve()).lower()
            if key not in seen:
                seen.add(key)
                files.append(f)
    files.sort()
    if not files:
        return pd.DataFrame()

    bbox = region_bbox(cfg)
    frames = []
    for f in files:
        try:
            part = read_night(f, bbox)
        except Exception as exc:
            print(f"  SKIPPED {f.name}: {type(exc).__name__}: {str(exc)[:90]}")
            continue
        frames.append(part)
        print(f"  parsed {f.name:<52} {len(part):>7} rows in study bbox")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main(config_path: str = "config.yaml", probe: bool = False) -> int:
    cfg = load_config(config_path)
    raw_dir = cfg.path(cfg["data"]["raw_dir"], cfg["data"]["viirs"]["raw_subdir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print("VNF NIGHTTIME INGEST")
    print("=" * 74)
    print(f"looking for local VNF files in : {raw_dir}")

    det = ingest_local(cfg, raw_dir)

    if det.empty:
        print()
        print("No VNF files found on disk, and no usable API route.")
        print()
        print("EOG restricted programmatic access to PAID subscribers on")
        print("2026-06-01; client credentials are issued after payment. An")
        print("academic licence covers the data, not the API, so the supported")
        print("route is a browser download:")
        print()
        print("  1. Log in at https://eogdata.mines.edu/products/vnf/")
        print("  2. Open the nightly VNF directory for the dates you want")
        print("  3. Save the .csv.gz files into:")
        print(f"       {raw_dir}")
        print("  4. Re-run: python run.py vnf")
        print()
        print("Start with ONE night and check the output before downloading")
        print("more -- the schema needs to match before bulk collection is")
        print("worth the effort.")
        return 1

    print()
    print(f"detections in study bbox : {len(det)}")

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
            print()
            print(f"radiant heat (MW)        : median {rh.median():.3f}  max {rh.max():.1f}")
            print("  ^ the column absent from the public workbooks. This is what")
            print("    makes the RULE 6 baseline reproducible.")
        else:
            print()
            print("  !! radiant_heat column found but entirely empty -- check the")
            print("     column mapping in WANTED against this file's header.")
    else:
        print()
        print("  !! NO radiant_heat column matched. Send me the file's header row;")
        print("     the VNF schema varies by version and WANTED needs remapping.")

    out = cfg.path(cfg["data"]["processed_dir"]) / "vnf_detections.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    det.to_parquet(out, index=False)
    print()
    print(f"wrote {out}  rows={len(det)}")
    return 0
