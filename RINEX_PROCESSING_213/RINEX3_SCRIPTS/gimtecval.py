# =============================================================================
# plot_regional_vs_gim_timeseries.py
# Daily TEC time series: Regional (~213-station pipeline) vs GIM
# DOY 014-030, with storm-day period shaded
#
# Updated from the original 17-station-only version:
#   - STATIONS dict replaced with the shared station_registry.py loader
#     (same registry process_tec.py and plot_tec_map_multistation.py use),
#     so this automatically covers whichever stations are in
#     station_coords_registry.csv -- 17 or ~213, no hardcoded list.
#   - TEC_OUTPUT_DIR updated to the current GISTDA_TEC base path.
#
# Sketch this recreates:
#   - "Regional" line = your pipeline's daily-median VTEC across the
#     registered network
#   - "GIM" line       = CODE/IGS GIM VTEC interpolated to the network centroid
#   - Red shaded band  = storm days (Jan 19-21, Kp up to 8.67)
#
# Usage:
#   conda activate base
#   python plot_regional_vs_gim_timeseries.py
# =============================================================================

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# import the IONEX reader from the validation script
import sys
sys.path.insert(0, str(Path(__file__).parent))
from validate_test import read_ionex, gim_at_point, find_ionex_file, IONEX_DIR

from station_registry import load_station_registry

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR       = Path("/Users/ziraa/Documents/GISTDA/GISTDA_TEC")
TEC_OUTPUT_DIR = BASE_DIR / "TEC_output"
OUT_DIR        = TEC_OUTPUT_DIR
STATION_REGISTRY_CSV = BASE_DIR / "station_coords_registry.csv"

DOY_START = 14
DOY_END   = 30
YEAR      = 2026

# storm period to shade (Jan 19-21, 2026 — Kp up to 8.67, see Kp analysis)
STORM_DOY_START = 19
STORM_DOY_END   = 21

# Station registry (lat, lon) -- loaded from the same CSV as the rest of
# the pipeline, instead of a hardcoded 17-station dict
STATIONS = load_station_registry(STATION_REGISTRY_CSV)

# network centroid — used to sample the GIM at a single representative point
CENTROID_LAT = float(np.mean([v[0] for v in STATIONS.values()]))
CENTROID_LON = float(np.mean([v[1] for v in STATIONS.values()]))

# =============================================================================
# REGIONAL VTEC LOADER (your pipeline)
# =============================================================================

def find_vtec_csv(station: str, doy: int) -> str | None:
    base = TEC_OUTPUT_DIR / station / f"DOY{doy:03d}"
    for p in [base / f"{station}_{doy:03d}_VTEC.csv",
              base / f"VTEC_{station}_{doy:03d}.csv"]:
        if p.exists(): return str(p)
    if base.exists():
        for m in (list(base.glob(f"*VTEC*{doy:03d}*.csv"))
                  + list(base.glob(f"*{doy:03d}*VTEC*.csv"))):
            if m.exists(): return str(m)
    return None


def daily_regional_median(doy: int) -> float:
    """
    Regional VTEC for one day = median across ALL registered stations and
    ALL PRNs, using the full day (not just one hour). This gives one number
    per DOY, matching the "Regional" line in the sketch. Works the same
    whether STATIONS has 17 entries or ~213.
    """
    station_medians = []
    for station in STATIONS:
        path = find_vtec_csv(station, doy)
        if path is None:
            continue
        try:
            df   = pd.read_csv(path, header=0)
            vtec = df.values.astype(float)
            med  = np.nanmedian(vtec)
            if np.isfinite(med) and med > 0:
                station_medians.append(med)
        except Exception:
            continue
    if not station_medians:
        return np.nan
    return float(np.nanmedian(station_medians))

# =============================================================================
# GIM DAILY VALUE  (network-centroid daily median across all available epochs)
# =============================================================================

def daily_gim_median(doy: int) -> float:
    """
    GIM VTEC for one day = median across all epochs in that day's IONEX file,
    evaluated at the network centroid lat/lon.
    """
    ionex_path = find_ionex_file(doy)
    if ionex_path is None:
        return np.nan
    try:
        gim = read_ionex(str(ionex_path))
    except Exception:
        return np.nan

    vals = []
    for epoch in gim["epochs"]:
        v = gim_at_point(gim, CENTROID_LAT, CENTROID_LON, epoch)
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return np.nan
    return float(np.nanmedian(vals))

# =============================================================================
# BUILD TIME SERIES
# =============================================================================

def build_series() -> pd.DataFrame:
    rows = []
    for doy in range(DOY_START, DOY_END + 1):
        date = datetime(YEAR, 1, 1) + timedelta(days=doy - 1)
        regional = daily_regional_median(doy)
        gim      = daily_gim_median(doy)
        rows.append({"doy": doy, "date": date,
                     "regional_vtec": regional, "gim_vtec": gim})
        print(f"  DOY{doy:03d}  {date.strftime('%Y-%m-%d')}  "
              f"Regional={regional:.1f}  GIM={gim:.1f}"
              if np.isfinite(regional) and np.isfinite(gim)
              else f"  DOY{doy:03d}  {date.strftime('%Y-%m-%d')}  "
                   f"Regional={regional}  GIM={gim}")
    return pd.DataFrame(rows)

# =============================================================================
# PLOT
# =============================================================================

def plot_timeseries(df: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    storm_start = datetime(YEAR, 1, 1) + timedelta(days=STORM_DOY_START - 1)
    storm_end   = datetime(YEAR, 1, 1) + timedelta(days=STORM_DOY_END)

    # ── storm-day shaded band ────────────────────────────────────────────────
    ax.axvspan(storm_start, storm_end, color="#e74c3c", alpha=0.18, zorder=1)
    ax.text(storm_start + (storm_end - storm_start) / 2,
            ax.get_ylim()[1] if df[["regional_vtec","gim_vtec"]].max().max() else 60,
            "Storm days", rotation=0, fontsize=11, fontweight="bold",
            color="#c0392b", ha="center", va="bottom")

    # ── regional line (black, solid) ─────────────────────────────────────────
    ax.plot(df["date"], df["regional_vtec"], color="#1a1a1a", lw=2.2,
            marker="o", ms=4, label=f"Regional ({len(STATIONS)}-station pipeline)",
            zorder=4)

    # ── GIM line (pink/red, lighter) ─────────────────────────────────────────
    ax.plot(df["date"], df["gim_vtec"], color="#e87ca0", lw=1.6,
            marker="o", ms=3, alpha=0.85, label="GIM (CODE)", zorder=3)

    # annotate "GIM" near the end of its line
    valid_gim = df.dropna(subset=["gim_vtec"])
    if len(valid_gim) > 0:
        last = valid_gim.iloc[-1]
        ax.annotate("GIM", xy=(last["date"], last["gim_vtec"]),
                   xytext=(12, 10), textcoords="offset points",
                   fontsize=10, color="#e87ca0", fontweight="bold",
                   arrowprops=dict(arrowstyle="->", color="#e87ca0", lw=1))

    # annotate "Regional" near the rising part of the curve
    valid_reg = df.dropna(subset=["regional_vtec"])
    if len(valid_reg) > 3:
        mid_i = len(valid_reg) // 3
        pt = valid_reg.iloc[mid_i]
        ax.annotate("Regional", xy=(pt["date"], pt["regional_vtec"]),
                   xytext=(-10, 35), textcoords="offset points",
                   fontsize=10, color="#1a1a1a", fontweight="bold",
                   arrowprops=dict(arrowstyle="->", color="#1a1a1a", lw=1))

    # ── axes ──────────────────────────────────────────────────────────────────
    ax.set_xlabel("Day (UTC)", fontsize=11)
    ax.set_ylabel("TEC (TECU)", fontsize=11)
    ax.set_title(
        f"Regional VTEC vs GIM — DOY {DOY_START:03d}\u2013{DOY_END:03d} ({YEAR})\n"
        f"{len(STATIONS)} stations  \u00b7  Network centroid: "
        f"{CENTROID_LAT:.2f}\u00b0N, {CENTROID_LON:.2f}\u00b0E",
        fontsize=12, fontweight="bold"
    )
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_ylim(bottom=0)
    ax.grid(True, lw=0.4, alpha=0.4)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)

    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"\n  → {out_path}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*60)
    print("Regional vs GIM daily TEC time series")
    print(f"DOY {DOY_START}-{DOY_END}, {len(STATIONS)} stations, network centroid "
          f"{CENTROID_LAT:.2f}N {CENTROID_LON:.2f}E")
    print("="*60)

    df = build_series()
    df.to_csv(OUT_DIR / "regional_vs_gim_timeseries.csv", index=False)

    out_path = str(OUT_DIR / "regional_vs_gim_timeseries.png")
    plot_timeseries(df, out_path)


if __name__ == "__main__":
    main()