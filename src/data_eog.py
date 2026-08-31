"""Stage 1 -- EOG flare catalogue ingest.

Reads the annual EOG "Global Gas Flaring" workbooks from data/raw/eog and
normalises them into one tidy site-year parquet table, so that nothing
downstream ever touches Excel again.

CAVEAT to carry into every downstream output: EOG volumes are themselves
derived from VIIRS -- the sensor whose blind spots this project targets. EOG
is a volume reference for the sites it contains. It is NOT independent
evidence for sites it omits.

Why this module is more than pd.read_excel: the schema drifts every year.
Verified against the nine downloaded workbooks on 2026-08-31:
  - sheet names: "flares_upstream" / "flares upstream" / "flare upstream"
  - temperature: "Avg_Temp_K" / "Avg. temp., K" / "Avg temp., K" / "Avg. temp"
  - id column:   "id_key" / "id #" / "Catalog ID" / "ID 2021"
  - 2017 has no Ellipticity column at all
  - 2012-2016 ships as ONE wide sheet with BCM_2012..BCM_2016 side by side
Matching columns by regex rather than exact name is what makes all nine
files loadable by the same code.

Run:
    python run.py download
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.config import Config, bcm_per_year_to_m3_per_hour, load_config

# Canonical output schema. Everything downstream depends on these names.
CANONICAL_COLUMNS = [
    "year",
    "site_id",
    "country",
    "eog_iso_code",
    "region_code",
    "latitude",
    "longitude",
    "bcm",
    "m3_per_h",
    "avg_temp_k",
    "ellipticity",
    "detection_freq",
    "clear_obs",
    "flare_type",
    "source_file",
]

# Regex patterns, applied case-insensitively to lowercased column names.
# Order matters where several columns could match: first hit wins.
PATTERNS = {
    "country": [r"^country$"],
    "eog_iso_code": [r"^iso[_ ]?code$"],
    "latitude": [r"^latitude$", r"^lat$"],
    "longitude": [r"^longitude$", r"^lon$", r"^long$"],
    "avg_temp_k": [r"^avg[_. ]*temp"],
    "ellipticity": [r"^ellip"],
    "flare_type": [r"^type$"],
}


def _norm(name: str) -> str:
    return str(name).strip().lower()


def _find(columns: list[str], patterns: list[str]) -> str | None:
    for pat in patterns:
        for col in columns:
            if re.match(pat, _norm(col)):
                return col
    return None


def _find_year_col(columns: list[str], stem: str, year: int) -> str | None:
    """Find a column like 'BCM 2019' / 'BCM_2019', or the unsuffixed variant.

    2019 labels its columns 'Detection freq.' and 'Clear obs.' with no year,
    so the unsuffixed fallback is required, not defensive padding.
    """
    for col in columns:
        n = _norm(col)
        if re.match(rf"^{stem}", n) and str(year) in n:
            return col
    for col in columns:
        if re.match(rf"^{stem}", _norm(col)):
            return col
    return None


def _find_id_col(columns: list[str], year: int) -> str | None:
    for pat in (rf"^id[_ ]?{year}$", rf"^id[_ ]?key[_ ]?{year}$", r"^id[_ ]?key$", r"^id ?#$", r"^catalog[_ ]?id$"):
        col = _find(columns, [pat])
        if col:
            return col
    return None


def _upstream_sheet(xl: pd.ExcelFile) -> str:
    """Pick the per-site upstream sheet, never the country-aggregate sheet."""
    for s in xl.sheet_names:
        n = _norm(s)
        if "upstream" in n and "countr" not in n:
            return s
    raise ValueError(f"no upstream site sheet in {xl.sheet_names}")


def _years_in_file(columns: list[str]) -> list[int]:
    """Years this sheet actually carries, from its BCM columns."""
    years = set()
    for col in columns:
        m = re.search(r"bcm[_ ]?(\d{4})", _norm(col))
        if m:
            years.add(int(m.group(1)))
    return sorted(years)


def parse_workbook(path: Path) -> pd.DataFrame:
    """Parse one workbook into tidy site-year rows.

    Handles both the per-year files and the wide 2012-2016 file, which packs
    five years of BCM/clear-obs/detection-frequency into single rows.
    """
    xl = pd.ExcelFile(path)
    sheet = _upstream_sheet(xl)
    df = pd.read_excel(path, sheet_name=sheet)
    cols = list(df.columns)

    years = _years_in_file(cols)
    if not years:
        raise ValueError(f"no BCM year column found in {path.name}: {cols}")

    static = {key: _find(cols, pats) for key, pats in PATTERNS.items()}

    frames = []
    for year in years:
        bcm_col = _find_year_col(cols, "bcm", year)
        if bcm_col is None:
            continue
        out = pd.DataFrame(index=df.index)
        out["year"] = year

        id_col = _find_id_col(cols, year)
        out["site_id"] = df[id_col] if id_col else pd.NA

        for key, col in static.items():
            out[key] = df[col] if col else pd.NA

        out["bcm"] = pd.to_numeric(df[bcm_col], errors="coerce")

        det_col = _find_year_col(cols, "detection", year)
        clr_col = _find_year_col(cols, "clear", year)
        out["detection_freq"] = pd.to_numeric(df[det_col], errors="coerce") if det_col else pd.NA
        out["clear_obs"] = pd.to_numeric(df[clr_col], errors="coerce") if clr_col else pd.NA

        # detection_freq units are NOT consistent across workbooks. Verified
        # 2026-08-31: percent (0-100) in 2012-2016 and 2018-2021, fraction
        # (0-1) in 2017 and 2022-2024. Left mixed, the column is meaningless
        # -- and it is the intermittency signal this whole project rests on.
        # Normalise everything to a FRACTION. The test is per year-column,
        # because a value above 1.0 is impossible on the fraction scale.
        det = pd.to_numeric(out["detection_freq"], errors="coerce")
        if det.notna().any() and float(det.max()) > 1.0:
            out["detection_freq"] = det / 100.0
            out.attrs["detection_freq_rescaled"] = True

        out["source_file"] = path.name
        frames.append(out)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_site_year_table(cfg: Config, gulf_only: bool = True) -> pd.DataFrame:
    raw_dir = cfg.path(cfg["data"]["raw_dir"], cfg["data"]["eog"]["raw_subdir"])
    files = sorted(raw_dir.glob("*.xlsx"))
    if not files:
        raise FileNotFoundError(f"no EOG workbooks in {raw_dir}")

    frames = []
    for f in files:
        part = parse_workbook(f)
        if not part.empty:
            frames.append(part)
        print(f"  parsed {f.name:<62} rows={len(part):>6} years={sorted(part['year'].unique())}")

    df = pd.concat(frames, ignore_index=True)

    # Map EOG ISO codes to our region codes. SAUCROP/SAUKWTNZ -> SAU.
    mapping = cfg.eog_code_to_region()
    df["region_code"] = df["eog_iso_code"].map(mapping)

    if gulf_only:
        codes = cfg.core_eog_codes()
        before = len(df)
        df = df[df["eog_iso_code"].isin(codes)].copy()
        print(f"  filtered to active regions {sorted(codes)}: {before} -> {len(df)} rows")
        if df.empty:
            raise ValueError(
                "Region filter returned zero rows. Check eog_iso_codes in "
                "config.yaml -- EOG does not use standard ISO3 for Saudi Arabia."
            )

    df["m3_per_h"] = df["bcm"].apply(
        lambda b: bcm_per_year_to_m3_per_hour(b) if pd.notna(b) else pd.NA
    )

    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[CANONICAL_COLUMNS]

    # site_id is an int in some workbooks and a string in others, which makes
    # the column mixed-type and unwritable to parquet. Force string: these are
    # identifiers, never quantities, and nothing should ever do arithmetic on
    # one. Note EOG re-issues ids per year, so site_id is NOT stable across
    # years -- cross-year site tracking needs the id2015..id2023 crosswalk
    # columns, which is a separate job.
    df["site_id"] = df["site_id"].astype("string")

    # Drop rows with no usable volume: they cannot serve as ground truth.
    n_before = len(df)
    df = df[df["bcm"].notna()].copy()
    if n_before != len(df):
        print(f"  dropped {n_before - len(df)} rows with missing BCM")

    return df.sort_values(["year", "region_code", "site_id"]).reset_index(drop=True)


def main(config_path: str = "config.yaml") -> int:
    cfg = load_config(config_path)
    print("Parsing EOG workbooks...")
    df = build_site_year_table(cfg)

    out_dir = cfg.path(cfg["data"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eog_site_year.parquet"
    df.to_parquet(out_path, index=False)

    print(f"\nwrote {out_path}  rows={len(df)}")
    print(f"years   : {df['year'].min()}-{df['year'].max()}")
    print(f"regions : {df['region_code'].value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
