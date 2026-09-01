"""FlareFinder pipeline entry point.

Why a run.py and not a Makefile: this project is developed on Windows, where
`make` is not installed by default. A Python entry point runs identically in
Git Bash, PowerShell and CMD, so the commands in the write-up work for anyone
reproducing the project regardless of platform.

Usage:
    python run.py check-env
    python run.py download --stage eog
    python run.py baseline
    python run.py evaluate

Every stage takes --config (default config.yaml). RULE 7: no numeric
parameter is ever passed on the command line; it goes in the config file so
that a run is fully described by (config file, git commit).
"""

from __future__ import annotations

import argparse
import sys

# Stage order is meaningful. RULE 6: `baseline` must run and be recorded
# before `train`, because the project claims to fix a failure that has to be
# measured first.
STAGES = {
    "check-env": ("src.check_env", "Verify every dependency imports and Earth Engine authenticates."),
    "download": ("src.data_eog", "Fetch EOG catalogue, VIIRS/Black Marble passes, Sentinel-2 scenes."),
    "vnf": ("src.data_viirs", "Download VIIRS Nightfire per-detection records (needs EOG token)."),
    "join": ("src.build_catalog", "Cluster sites across years into stable ids; build catalog.parquet."),
    "features": ("src.features", "Build fusion features, including temporal intermittency."),
    "splits": ("src.splits", "Build by-site/year/region holdouts and run the confound diagnostic."),
    "figures": ("src.figures", "Render publication figures (fig01 flare map: PNG, PDF, HTML)."),
    "baseline": ("src.baseline", "RULE 6: reproduce the standard VIIRS calibration and measure its bias."),
    "train": ("src.model_volume", "Fit detection and volume models on the training split only."),
    "evaluate": ("src.evaluate", "Per-size-bin metrics with bootstrap CIs on held-out sites/years/regions."),
    "app": ("app", "Launch the Streamlit map of flares absent from the official record."),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="FlareFinder pipeline runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="stages:\n"
        + "\n".join(f"  {name:<10} {desc}" for name, (_, desc) in STAGES.items()),
    )
    parser.add_argument("stage", choices=list(STAGES), help="pipeline stage to run")
    parser.add_argument("--config", default="config.yaml", help="config file (default: config.yaml)")
    parser.add_argument("--dry-run", action="store_true", help="show what would run, change nothing")
    parser.add_argument("--probe", action="store_true", help="vnf: check token and fetch a single night")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    module, desc = STAGES[args.stage]

    print(f"[flarefinder] stage   : {args.stage}")
    print(f"[flarefinder] module  : {module}")
    print(f"[flarefinder] config  : {args.config}")
    print(f"[flarefinder] purpose : {desc}")

    if args.dry_run:
        print("[flarefinder] dry run, stopping here.")
        return 0

    if args.stage == "check-env":
        from src import check_env

        return check_env.main(args.config)

    if args.stage == "download":
        from src import data_eog

        return data_eog.main(args.config)

    if args.stage == "vnf":
        from src import data_viirs

        return data_viirs.main(args.config, probe=args.probe)

    if args.stage == "join":
        from src import build_catalog

        return build_catalog.main(args.config)

    if args.stage == "figures":
        from src import figures

        return figures.main(args.config)

    if args.stage == "baseline":
        from src import baseline

        return baseline.main(args.config)

    if args.stage == "splits":
        from src import splits

        return splits.main(args.config)

    print(
        f"\n[flarefinder] NOT IMPLEMENTED: {module} is still a stub.\n"
        f"              Implement it before this stage can run.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
