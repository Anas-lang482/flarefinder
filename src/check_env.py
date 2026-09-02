"""Stage 0 -- environment verification.

Prints PASS / FAIL / SKIP for every dependency, directory and credential the
pipeline needs, then exits non-zero if anything is broken. Run this first,
and run it again any time something behaves strangely.

Distinguishing SKIP from FAIL matters: a missing Earth Engine project id
while GEE registration is still pending is a normal state, not a broken
environment, and reporting it as a failure would train you to ignore
failures.

Standalone:
    python src/check_env.py
Via the runner:
    python run.py check-env
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# (import name, pip name, why the project needs it)
REQUIRED_PACKAGES = [
    ("numpy", "numpy", "numerics"),
    ("pandas", "pandas", "site-year tables"),
    ("pyarrow", "pyarrow", "parquet in data/processed"),
    ("openpyxl", "openpyxl", "reads the EOG .xlsx catalogue"),
    ("yaml", "pyyaml", "config.yaml loader"),
    ("geopandas", "geopandas", "site geometry"),
    ("shapely", "shapely", "geometry ops"),
    ("pyproj", "pyproj", "projections / metre distances"),
    ("ee", "earthengine-api", "Sentinel-2 SWIR extraction"),
    ("boto3", "boto3", "s3://blackmarble-combustion"),
    ("requests", "requests", "EOG downloads"),
    ("tqdm", "tqdm", "download progress"),
    ("sklearn", "scikit-learn", "splits + metrics"),
    ("lightgbm", "lightgbm", "primary model (RULE 5)"),
    ("xgboost", "xgboost", "model comparison"),
    ("shap", "shap", "feature attribution"),
    ("matplotlib", "matplotlib", "figures"),
    ("folium", "folium", "interactive map"),
    ("streamlit", "streamlit", "public app"),
]

# Nice to have, but the pipeline runs without them. A missing optional
# package is a SKIP, never a FAIL -- reporting it as a failure would train
# you to ignore the FAIL column, which is the one that matters.
# (import name, pip name, why it is optional)
OPTIONAL_PACKAGES = [
    (
        "geemap",
        "geemap",
        "GEE convenience wrapper -- unused: every Earth Engine call in src/ "
        "uses the `ee` API directly, so geemap being blocked costs nothing",
    ),
    (
        "rasterio",
        "rasterio",
        "local GeoTIFF IO -- unused: Sentinel-2 is read via Earth Engine "
        "server-side reductions that return tables, not rasters",
    ),
]

REQUIRED_DIRS = [
    "data/raw",
    "data/processed",
    "src",
    "notebooks",
    "experiments",
    "models",
    "app",
    "figures",
]

MIN_PYTHON = (3, 11)


class Report:
    """Collects results so the summary reflects what actually happened."""

    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.skipped = 0

    def record(self, status: str, item: str, detail: str = "") -> None:
        colour = {"PASS": "PASS", "FAIL": "FAIL", "SKIP": "SKIP"}[status]
        line = f"  [{colour}] {item}"
        if detail:
            line += f"  -- {detail}"
        print(line)
        if status == "PASS":
            self.passed += 1
        elif status == "FAIL":
            self.failed += 1
        else:
            self.skipped += 1


def check_python(report: Report) -> None:
    print("\nPython")
    v = sys.version_info
    actual = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) >= MIN_PYTHON:
        report.record("PASS", f"python {actual}", f"needs >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]}")
    else:
        report.record("FAIL", f"python {actual}", f"needs >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]}")

    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    if in_venv:
        report.record("PASS", "virtual environment", sys.prefix)
    else:
        # Not fatal, but installing into the system interpreter makes the
        # environment hard to reproduce, which undermines RULE 7.
        report.record("SKIP", "virtual environment", "not active -- recommended but not required")


def check_packages(report: Report) -> None:
    print("\nPackages")
    for import_name, pip_name, why in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(import_name)
        except Exception as exc:  # ImportError, but also DLL errors on Windows
            msg = f"{type(exc).__name__}: {exc}"
            # Windows Smart App Control blocks unsigned native DLLs based on a
            # cloud reputation lookup, so the SAME package can import fine one
            # run and fail the next. Say so, or this looks like a broken
            # install and someone reinstalls packages that were never broken.
            if "Application Control" in str(exc):
                msg = ("BLOCKED by Windows Smart App Control (not a broken "
                       "install; this failure is intermittent) -- " + msg)
            report.record("FAIL", f"{pip_name:<18}", msg)
            continue
        version = getattr(mod, "__version__", "unknown")
        report.record("PASS", f"{pip_name:<18}", f"{version}  ({why})")


def check_optional_packages(report: Report) -> None:
    print("\nOptional packages")
    for import_name, pip_name, why in OPTIONAL_PACKAGES:
        try:
            mod = importlib.import_module(import_name)
        except Exception as exc:
            report.record("SKIP", f"{pip_name:<18}", f"unavailable ({type(exc).__name__}) -- {why}")
            continue
        version = getattr(mod, "__version__", "unknown")
        report.record("PASS", f"{pip_name:<18}", f"{version}  (optional)")


def check_dirs(report: Report) -> None:
    print("\nRepo layout")
    for rel in REQUIRED_DIRS:
        path = REPO_ROOT / rel
        if path.is_dir():
            report.record("PASS", rel)
        else:
            report.record("FAIL", rel, "missing")


def check_config(report: Report, config_path: str) -> dict | None:
    print("\nConfig")
    path = REPO_ROOT / config_path
    if not path.is_file():
        report.record("FAIL", config_path, "missing")
        return None
    try:
        import yaml
    except ImportError:
        report.record("SKIP", config_path, "pyyaml not installed, cannot parse")
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
    except Exception as exc:
        report.record("FAIL", config_path, f"parse error: {exc}")
        return None

    report.record("PASS", config_path, "parsed")

    for key in ("regions", "years", "seeds", "thresholds", "evaluation", "models"):
        if key in cfg:
            report.record("PASS", f"config.{key}")
        else:
            report.record("FAIL", f"config.{key}", "missing section")

    core = [r for r in cfg.get("regions", []) if r.get("tier") == "core"]
    report.record(
        "PASS" if core else "FAIL",
        "config.regions core tier",
        ", ".join(r["code"] for r in core) if core else "no core regions defined",
    )

    seed = cfg.get("seeds", {}).get("global")
    report.record("PASS" if seed is not None else "FAIL", "config.seeds.global", str(seed))
    return cfg


def check_earthengine(report: Report, cfg: dict | None) -> None:
    print("\nEarth Engine")
    try:
        import ee
    except ImportError:
        report.record("FAIL", "earthengine-api", "not installed")
        return

    project = os.environ.get("EE_PROJECT")
    source = "EE_PROJECT env var"
    if not project and cfg:
        project = (cfg.get("data", {}).get("earthengine", {}) or {}).get("project")
        source = "config.yaml"

    if not project:
        report.record(
            "SKIP",
            "GEE project id",
            "not set -- fill data.earthengine.project in config.yaml once GEE approves you",
        )
        return
    report.record("PASS", "GEE project id", f"{project} (from {source})")

    try:
        ee.Initialize(project=project)
    except Exception as exc:
        report.record(
            "FAIL",
            "ee.Initialize",
            f"{type(exc).__name__} -- run `earthengine authenticate`, then retry",
        )
        return
    report.record("PASS", "ee.Initialize", "authenticated")

    # A real query, not just a handshake: proves the Sentinel-2 collection is
    # actually reachable with these credentials.
    try:
        collection_id = "COPERNICUS/S2_SR_HARMONIZED"
        if cfg:
            collection_id = cfg["data"]["sentinel2"]["gee_collection"]
        n = ee.ImageCollection(collection_id).filterDate("2024-01-01", "2024-01-08").size().getInfo()
        report.record("PASS", "S2 collection query", f"{collection_id}: {n} images in a test week")
    except Exception as exc:
        report.record("FAIL", "S2 collection query", f"{type(exc).__name__}: {exc}")


def main(config_path: str = "config.yaml") -> int:
    print("=" * 70)
    print("FlareFinder environment check")
    print("=" * 70)

    report = Report()
    check_python(report)
    check_packages(report)
    check_optional_packages(report)
    check_dirs(report)
    cfg = check_config(report, config_path)
    check_earthengine(report, cfg)

    print("\n" + "=" * 70)
    print(f"PASS {report.passed}   FAIL {report.failed}   SKIP {report.skipped}")
    if report.failed:
        print("Environment is NOT ready. Fix the FAIL items above.")
    elif report.skipped:
        print("Environment is usable, but some checks were skipped (see SKIP).")
    else:
        print("Environment is ready.")
    print("=" * 70)
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "config.yaml"))
