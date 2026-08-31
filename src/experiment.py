"""Stage 10 -- experiment logging.

RULE 8: every run appends its params and metrics to experiments/log.csv.
A result that was not logged did not happen.

Long format, one row per metric: a wide table cannot hold a per-size-bin
metric with a confidence interval without exploding into dozens of columns,
and RULE 2 means almost every metric here IS per-size-bin.

Each row also carries the git commit and a hash of the config, so a logged
number can be traced back to the exact code and settings that produced it.
That pair is what makes RULE 7 reproducibility real rather than aspirational.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import uuid
from datetime import datetime, timezone
from typing import Any

from src.config import Config

FIELDS = [
    "run_id", "timestamp", "stage", "script", "config_hash", "git_commit",
    "region_scope", "split_type", "split_fold", "model", "params_json",
    "metric", "size_bin", "value", "ci_lo", "ci_hi", "n_samples", "notes",
]


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _config_hash(cfg: Config) -> str:
    blob = json.dumps(cfg.raw, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def log_metrics(
    cfg: Config,
    stage: str,
    script: str,
    model: str,
    split_type: str,
    params: dict[str, Any],
    records: list[dict[str, Any]],
    split_fold: str = "",
    notes: str = "",
) -> str:
    """Append one row per metric. Returns the run_id tying them together."""
    path = cfg.path(cfg["experiments"]["log_path"])
    path.parent.mkdir(parents=True, exist_ok=True)

    run_id = uuid.uuid4().hex[:8]
    common = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stage": stage,
        "script": script,
        "config_hash": _config_hash(cfg),
        "git_commit": _git_commit(),
        "region_scope": ",".join(sorted(r["code"] for r in cfg.active_regions())),
        "split_type": split_type,
        "split_fold": split_fold,
        "model": model,
        "params_json": json.dumps(params, default=str),
        "notes": notes,
    }

    existed = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        if not existed:
            w.writeheader()
        for r in records:
            row = dict(common)
            row.update({
                "metric": r.get("metric", ""),
                "size_bin": r.get("size_bin", ""),
                "value": r.get("value", ""),
                "ci_lo": r.get("ci_lo", ""),
                "ci_hi": r.get("ci_hi", ""),
                "n_samples": r.get("n", ""),
            })
            w.writerow(row)
    return run_id
