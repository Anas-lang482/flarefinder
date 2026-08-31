"""Publication figures.

fig01: geographic distribution of catalogued flare sites.

COLOUR
    Categorical slots 1-3 of the reference palette, in fixed order, never
    cycled. A map is a scatter form, so pairs are judged all-against-all
    rather than adjacent-only, and only the first three slots clear the
    colour-vision-deficiency floors under that rule. There are four country
    labels in the catalogue, so the smallest -- the Saudi-Kuwaiti Neutral
    Zone, ~20 sites -- folds into a neutral "Other" rather than taking slot
    4, which would put yellow beside orange and fail.
    Identity is never carried by colour alone: every series is named in the
    legend, and aqua sits under 3:1 contrast on a light surface so the
    legend labels are mandatory relief, not decoration.

SIZE
    Marker area scales with LOG volume, not volume. Stated plainly because
    it matters: site volumes span roughly five orders of magnitude, and a
    linear area encoding would render the entire small-flare population --
    the actual subject of this project -- as invisible dots. The size legend
    prints real values so the reader can decode it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Config, load_config

# Reference palette, categorical slots 1-3, fixed order. Do not cycle.
PALETTE = {
    "Iran": "#2a78d6",          # slot 1, blue
    "Iraq": "#eb6834",          # slot 2, orange
    "Saudi Arabia": "#1baf7a",  # slot 3, aqua
}
OTHER_COLOR = "#8a8985"  # neutral ink, not a categorical hue
OTHER_LABEL = "Other (Neutral Zone)"

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#e3e2dd"


def _latest_per_site(df: pd.DataFrame) -> pd.DataFrame:
    """One row per site: its most recent observation."""
    idx = df.groupby("site_id")["year"].idxmax()
    out = df.loc[idx].copy()
    out["series"] = np.where(out["country"].isin(PALETTE), out["country"], OTHER_LABEL)
    return out


def _size_from_volume(
    v: np.ndarray,
    ref: np.ndarray | None = None,
    lo: float = 6.0,
    hi: float = 260.0,
) -> np.ndarray:
    """Marker area from log10 volume, floored so tiny flares stay visible.

    `ref` supplies the scale's endpoints. It matters: sizing the legend
    swatches against themselves collapses their range to a single value and
    renders three identical dots, which is how the first version of this
    figure shipped an undecodable size legend.
    """
    logv = np.log10(np.clip(v, 1e-4, None))
    base = np.log10(np.clip(ref if ref is not None else v, 1e-4, None))
    span = base.max() - base.min()
    frac = (logv - base.min()) / span if span > 0 else np.zeros_like(logv)
    return lo + np.clip(frac, 0, 1) * (hi - lo)


def fig01_static(df: pd.DataFrame, cfg: Config) -> list:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    latest = _latest_per_site(df)
    fig, ax = plt.subplots(figsize=(10, 8.5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    order = [c for c in PALETTE if c in set(latest["series"])] + (
        [OTHER_LABEL] if OTHER_LABEL in set(latest["series"]) else []
    )
    for name in order:
        sub = latest[latest["series"] == name]
        ax.scatter(
            sub["lon"], sub["lat"],
            s=_size_from_volume(sub["volume_mcm"].to_numpy(float)),
            c=PALETTE.get(name, OTHER_COLOR),
            alpha=0.55,
            linewidths=0.5,
            edgecolors=SURFACE,   # 2px-equivalent surface ring on overlap
            label=f"{name} (n={len(sub):,})",
            zorder=3,
        )

    ax.set_xlabel("Longitude (°E)", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("Latitude (°N)", color=INK_SECONDARY, fontsize=10)
    ax.set_title(
        "Catalogued gas flare sites, Gulf study region",
        color=INK_PRIMARY, fontsize=14, pad=34, loc="left",
    )
    ax.text(
        0.0, 1.012,
        f"EOG VIIRS record, most recent observation per site  |  "
        f"{latest['site_id'].nunique():,} sites  |  {int(df['year'].min())}–{int(df['year'].max())}",
        transform=ax.transAxes, color=INK_SECONDARY, fontsize=9.5, va="bottom",
    )

    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)

    # Latitude correction so the map is not horizontally stretched.
    ax.set_aspect(1.0 / np.cos(np.radians(float(latest["lat"].mean()))))

    legend1 = ax.legend(
        loc="upper left", frameon=False, fontsize=9.5, labelcolor=INK_SECONDARY,
        title="Country", title_fontsize=9.5, scatterpoints=1, markerscale=0.7,
    )
    legend1.get_title().set_color(INK_SECONDARY)
    ax.add_artist(legend1)

    # Size legend in real units, because a log-area encoding is undecodable
    # without one.
    v = latest["volume_mcm"].to_numpy(float)
    # Decade-spaced ticks beat percentiles here: the 5th percentile
    # rounds to 0.00 Mm3/yr and labels a swatch with nothing.
    ticks = [t for t in (0.1, 10.0, 1000.0) if v.min() <= t <= v.max()]
    ticks = ticks or [np.percentile(v, p) for p in (10, 50, 90)]
    handles = [
        Line2D(
            [], [], marker="o", linestyle="none",
            markersize=np.sqrt(_size_from_volume(np.array([t]), ref=v)[0]),
            markerfacecolor=INK_SECONDARY, markeredgecolor=SURFACE, alpha=0.55,
            label=f"{t:,g} Mm³/yr",
        )
        for t in ticks
    ]
    leg2 = ax.legend(
        handles=handles, loc="lower left", frameon=False, fontsize=9,
        labelcolor=INK_SECONDARY, title="Volume (log area scale)",
        title_fontsize=9, labelspacing=1.4, borderpad=1.0,
    )
    leg2.get_title().set_color(INK_SECONDARY)

    limit = float(cfg.thresholds["viirs"]["detection_limit_m3_per_h"]) * 8760 / 1e6
    n_below = int((v < limit).sum())
    fig.text(
        0.5, 0.015,
        f"Marker area scales with log volume: a linear scale would make the "
        f"small-flare population invisible. {n_below:,} of {len(v):,} sites "
        f"({100 * n_below / len(v):.0f}%) fall below the "
        f"{limit:.2f} Mm³/yr VIIRS detection limit\n(Seymour et al. 2025, "
        f"assuming continuous flaring). EOG volumes are VIIRS-derived and are "
        f"not independent evidence for sites the record omits.",
        ha="center", color=INK_SECONDARY, fontsize=8, linespacing=1.5,
    )

    fig.tight_layout(rect=(0, 0.075, 1, 1))

    out_dir = cfg.path(cfg["outputs"]["figures_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in ("png", "pdf"):  # PNG per the spec, PDF because config wants vector
        p = out_dir / f"fig01_flare_map.{ext}"
        fig.savefig(p, dpi=int(cfg["outputs"]["figure_dpi"]), facecolor=SURFACE)
        written.append(p)
    plt.close(fig)
    return written


def fig01_interactive(df: pd.DataFrame, cfg: Config):
    import folium

    latest = _latest_per_site(df)
    center = cfg["outputs"]["map_center"]
    m = folium.Map(location=center, zoom_start=int(cfg["outputs"]["map_zoom"]),
                   tiles="CartoDB positron")

    vols = latest["volume_mcm"].to_numpy(float)
    sizes = _size_from_volume(vols, ref=vols)
    for (_, r), s in zip(latest.iterrows(), sizes):
        colour = PALETTE.get(r["series"], OTHER_COLOR)
        folium.CircleMarker(
            location=[float(r["lat"]), float(r["lon"])],
            radius=float(np.sqrt(s) * 0.8),
            color=colour, weight=1, fill=True, fill_color=colour, fill_opacity=0.55,
            tooltip=(
                f"{r['site_id']}<br>{r['country']}<br>"
                f"{r['volume_mcm']:,.3f} Mm³/yr ({int(r['year'])})<br>"
                f"observed in {int(r['n_years_observed'])} year(s)"
            ),
        ).add_to(m)

    counts = latest["series"].value_counts()
    rows = "".join(
        f'<div><span style="display:inline-block;width:10px;height:10px;'
        f'border-radius:50%;background:{PALETTE.get(k, OTHER_COLOR)};'
        f'margin-right:6px"></span>{k} ({counts[k]:,})</div>'
        for k in list(PALETTE) + [OTHER_LABEL] if k in counts
    )
    m.get_root().html.add_child(folium.Element(
        f'<div style="position:fixed;bottom:24px;left:24px;z-index:9999;'
        f'background:{SURFACE};padding:10px 12px;border:1px solid {GRID};'
        f'border-radius:6px;font:12px system-ui;color:{INK_SECONDARY}">'
        f'<b style="color:{INK_PRIMARY}">Flare sites</b>{rows}'
        f'<div style="margin-top:6px">Circle area ∝ log volume</div></div>'
    ))

    out = cfg.path(cfg["outputs"]["figures_dir"]) / "fig01_flare_map.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out))
    return out


def main(config_path: str = "config.yaml") -> int:
    cfg = load_config(config_path)
    path = cfg.path(cfg["data"]["processed_dir"], "catalog.parquet")
    if not path.exists():
        print(f"missing {path} -- run `python run.py join` first")
        return 1
    df = pd.read_parquet(path)
    for p in fig01_static(df, cfg):
        print(f"wrote {p}")
    print(f"wrote {fig01_interactive(df, cfg)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
