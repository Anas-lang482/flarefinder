"""Stage 0 -- configuration loader.

Single source of truth. RULE 7 says no numeric parameter may appear in code;
this module is how code gets them instead.

Usage:
    from src.config import load_config
    cfg = load_config()
    cfg.seeds["global"]
    cfg.core_eog_codes()
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

# EOG reports annual volumes in billion cubic metres. The whole project talks
# in m3/h because that is the unit the VIIRS detection limit is quoted in
# (Seymour et al. 2025: ~360 m3/h), so conversions live here rather than
# being re-derived, differently, in five modules.
HOURS_PER_YEAR = 8760.0
BCM_TO_M3 = 1e9


@dataclass
class Config:
    raw: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    # -- convenience accessors -------------------------------------------
    @property
    def seeds(self) -> dict[str, int]:
        return self.raw["seeds"]

    @property
    def years(self) -> dict[str, int]:
        return self.raw["years"]

    @property
    def evaluation(self) -> dict[str, Any]:
        return self.raw["evaluation"]

    @property
    def thresholds(self) -> dict[str, Any]:
        return self.raw["thresholds"]

    def active_regions(self) -> list[dict[str, Any]]:
        tiers = set(self.raw.get("active_tiers", ["core"]))
        return [r for r in self.raw["regions"] if r.get("tier") in tiers]

    def core_eog_codes(self) -> set[str]:
        """EOG ISO codes for the active regions.

        Deliberately NOT the region `code` field. The EOG workbooks use
        SAUCROP and SAUKWTNZ for Saudi Arabia, so joining on "SAU" silently
        returns zero rows. Always route through here.
        """
        codes: set[str] = set()
        for r in self.active_regions():
            codes.update(r.get("eog_iso_codes", []))
        return codes

    def eog_code_to_region(self) -> dict[str, str]:
        """Map every EOG ISO code back to its region code (SAUCROP -> SAU)."""
        out: dict[str, str] = {}
        for r in self.raw["regions"]:
            for c in r.get("eog_iso_codes", []):
                out[c] = r["code"]
        return out

    def path(self, *parts: str) -> Path:
        return REPO_ROOT.joinpath(*parts)


def load_config(config_path: str | Path = "config.yaml") -> Config:
    path = Path(config_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with path.open("r", encoding="utf-8") as fh:
        return Config(yaml.safe_load(fh))


def bcm_per_year_to_m3_per_hour(bcm: float) -> float:
    return bcm * BCM_TO_M3 / HOURS_PER_YEAR
