# =============================================================================
# plot_roti_ipp_map_213stations.py
# IPP scatter map coloured by ROTI — matching KMITL CSSRG style
# ~213-station expanded network (RINEX 2.11 + 3.04)
#
# Same visual/statistical logic as plot_roti_ipp_map_17stations.py, but
# station coordinates come from the shared station_coords_registry.csv
# (via station_registry.py) instead of a hardcoded dict, so this covers
# however many stations are actually registered.
#
# *** PERFORMANCE WARNING ***
# 5-min mode (HIGH_RES_MODE=True, the default here, per request) generates
# 288 frames/day x 17 days = 4,896 frames, each scattering up to
# 213 stations x 32 PRNs = 6,816 candidate points before filtering to
# visible/finite ones. This is substantially heavier than the 17-station
# version. Test on 1-2 days first (set DOY_START = DOY_END) before running
# the full range -- and consider whether you actually need every day at
# 5-min resolution, or whether a subset of "interesting" days (storm onset,
# the delayed DOY 021 peak, the DOY 023/025 sub-threshold-Kp days) would
# serve the same purpose in a fraction of the runtime.
#
# Reads from:
#   TEC_output/{STATION}/DOY{DOY}/{STATION}_{DOY}_ipp_lat.csv
#   TEC_output/{STATION}/DOY{DOY}/{STATION}_{DOY}_ipp_lon.csv
#   TEC_output/{STATION}/DOY{DOY}/{STATION}_{DOY}_ROTI.csv
#
# Usage:
#   conda activate base
#   python plot_roti_ipp_map_213stations.py
# =============================================================================

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
from matplotlib.cm import ScalarMappable

from station_registry import load_station_registry

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR       = Path("/Users/ziraa/Documents/GISTDA/GISTDA_TEC")
TEC_OUTPUT_DIR = BASE_DIR / "TEC_output"
GEOJSON_PATH   = Path("/Users/ziraa/Downloads/th.json")
PLOT_OUT_DIR   = TEC_OUTPUT_DIR / "ROTI_IPP_maps_213stations"
STATION_REGISTRY_CSV = BASE_DIR / "station_coords_registry.csv"

DOY_START = 14
DOY_END   = 30
YEAR      = 2026

STATIONS = load_station_registry(STATION_REGISTRY_CSV)

# =============================================================================
# COLORMAP — grey (low ROTI) → yellow → red (high ROTI), matching reference
# =============================================================================
ROTI_COLORS = [
    "#cccccc", "#cccccc", "#ffff00", "#ffaa00",
    "#ff4400", "#cc0000", "#880000",
]
ROTI_CMAP = LinearSegmentedColormap.from_list("roti_ipp", ROTI_COLORS, N=256)
ROTI_MIN  = 0.0
ROTI_MAX  = 1.0   # TECU/min — clip above this

LAT_MIN, LAT_MAX =  4.5, 22.0
LON_MIN, LON_MAX = 96.0, 107.5

# =============================================================================
# GEOJSON
# =============================================================================

def load_thai_paths(path: str) -> list:
    with open(path) as f:
        geo = json.load(f)
    polys, result = geo["features"][0]["geometry"]["coordinates"], []
    for poly in polys:
        verts, codes = [], []
        for ring in poly:
            arr = np.array(ring)
            if len(arr) < 3: continue
            verts.append(arr[0]);  codes.append(MplPath.MOVETO)
            for pt in arr[1:]:
                verts.append(pt);  codes.append(MplPath.LINETO)
            verts.append(arr[0]); codes.append(MplPath.CLOSEPOLY)
        if verts:
            result.append(MplPath(np.array(verts), np.array(codes)))
    return result

# =============================================================================
# FILE FINDER (no station-specific hardcoding needed -- generic glob
# fallback handles naming oddities across the larger, more heterogeneous
# ~213-station set)
# =============================================================================

def find_csv(station: str, doy: int, prefix: str) -> str | None:
    base = TEC_OUTPUT_DIR / station / f"DOY{doy:03d}"
    for p in [base / f"{station}_{doy:03d}_{prefix}.csv",
              base / f"{prefix}_{station}_{doy:03d}.csv"]:
        if p.exists(): return str(p)
    if base.exists():
        matches = (list(base.glob(f"*{prefix}*{doy:03d}*.csv"))
                   + list(base.glob(f"*{doy:03d}*{prefix}*.csv")))
        if matches:
            return str(matches[0])
    return None

# =============================================================================
# LOAD ONE STATION'S IPP + ROTI FOR ONE DAY
# =============================================================================

def load_station_day(station: str, doy: int) -> dict | None:
    ipp_lat_path = find_csv(station, doy, "ipp_lat")
    ipp_lon_path = find_csv(station, doy, "ipp_lon")
    roti_path    = find_csv(station, doy, "ROTI")

    if ipp_lat_path is None or ipp_lon_path is None or roti_path is None:
        return None
    try:
        ipp_lat = pd.read_csv(ipp_lat_path, header=0).values.astype(float)
        ipp_lon = pd.read_csv(ipp_lon_path, header=0).values.astype(float)
        roti    = pd.read_csv(roti_path,    header=0).values.astype(float)

        if ipp_lat.shape[1] != 32:
            return None

        n_rows  = ipp_lat.shape[0]
        cadence = max(1, 86400 // n_rows)

        min_rows = min(ipp_lat.shape[0], ipp_lon.shape[0], roti.shape[0])
        ipp_lat  = ipp_lat[:min_rows, :]
        ipp_lon  = ipp_lon[:min_rows, :]
        roti     = roti[:min_rows, :]
        roti     = np.clip(roti, 0, ROTI_MAX)

        return {"station": station, "ipp_lat": ipp_lat, "ipp_lon": ipp_lon,
               "roti": roti, "cadence": cadence}
    except Exception as e:
        print(f"  [load error] {station} DOY{doy:03d}: {e}")
        return None


def load_all_stations(doy: int) -> list:
    records = []
    for station in STATIONS:
        rec = load_station_day(station, doy)
        if rec is not None:
            records.append(rec)
    return records

# =============================================================================
# COLLECT IPP POINTS FOR ONE TIME WINDOW
# =============================================================================

def collect_ipp_points(records: list, sod_centre: int,
                       window_min: float = 5.0) -> tuple:
    window_sec = int(window_min * 60)
    sod_lo = max(0,     sod_centre - window_sec)
    sod_hi = min(86399, sod_centre + window_sec)

    all_lat, all_lon, all_roti = [], [], []

    for rec in records:
        n_rows = rec["ipp_lat"].shape[0]
        cadence = max(1, 86400 // n_rows)

        row_lo = max(0,        sod_lo // cadence)
        row_hi = min(n_rows-1, sod_hi // cadence)

        for row in range(row_lo, row_hi + 1):
            for prn in range(32):
                ilat = rec["ipp_lat"][row, prn]
                ilon = rec["ipp_lon"][row, prn]
                rval = rec["roti"][row, prn]

                if (np.isfinite(ilat) and np.isfinite(ilon)
                        and np.isfinite(rval)
                        and LAT_MIN <= ilat <= LAT_MAX
                        and LON_MIN <= ilon <= LON_MAX):
                    all_lat.append(float(ilat))
                    all_lon.append(float(ilon))
                    all_roti.append(float(rval))

    return np.array(all_lat), np.array(all_lon), np.array(all_roti)

# =============================================================================
# DRAW ONE PANEL
# =============================================================================

def draw_panel(ax, ipp_lat, ipp_lon, roti_vals,
               thai_paths, time_label: str, date_str: str, doy: int,
               n_stations: int):

    ax.set_facecolor("white")

    for path in thai_paths:
        ax.add_patch(PathPatch(path, facecolor="none",
                               edgecolor="black", lw=0.9, zorder=4))

    if len(ipp_lat) > 0:
        order = np.argsort(roti_vals)
        ax.scatter(
            ipp_lon[order], ipp_lat[order],
            c=roti_vals[order], cmap=ROTI_CMAP,
            vmin=ROTI_MIN, vmax=ROTI_MAX,
            s=8, alpha=0.65, zorder=3,   # slightly smaller/more transparent
            edgecolors="none", linewidths=0,   # given much higher point density
        )
    else:
        ax.text(0.5, 0.5, "no IPP data",
                transform=ax.transAxes, fontsize=7,
                color="#888888", ha="center", va="center")

    utc_h = int(time_label.split(":")[0]) if ":" in time_label else 0
    utc_m = int(time_label.split(":")[1]) if ":" in time_label else 0
    lt_h  = (utc_h + 7) % 24
    lt_m  = utc_m

    title = (f"Date: {date_str.replace('-','')}  DOY {doy:03d}  "
             f"ROTI(TECU/min) at  {utc_h:02d}:{utc_m:02d}:00 UTC "
             f"({lt_h:02d}:{lt_m:02d}:00 LT)  \u2014  {n_stations} stations")
    ax.set_title(title, fontsize=7, pad=4, color="black")

    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_xlabel("Longitude (Degree)", fontsize=8)
    ax.set_ylabel("Latitude (Degree)",  fontsize=8)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.tick_params(labelsize=7.5)
    ax.grid(True, lw=0.3, alpha=0.3)
    for sp in ax.spines.values():
        sp.set_edgecolor("#888888")

# =============================================================================
# MODE A — 12-panel daily summary (2-hour bins)
# =============================================================================

def plot_day_summary(doy: int, records: list, thai_paths: list):
    date     = datetime(YEAR, 1, 1) + timedelta(days=doy - 1)
    date_str = date.strftime("%Y-%m-%d")

    hour_bins = list(range(0, 24, 2))
    NCOLS, NROWS = 4, 3

    fig = plt.figure(figsize=(NCOLS * 4.0, NROWS * 4.5 + 1.5), facecolor="white")
    gs  = GridSpec(NROWS, NCOLS, figure=fig,
                   hspace=0.25, wspace=0.12,
                   left=0.06, right=0.91, top=0.93, bottom=0.07)

    axes_flat = [fig.add_subplot(gs[r, c])
                 for r in range(NROWS) for c in range(NCOLS)]

    for panel_i, h in enumerate(hour_bins):
        sod = h * 3600
        lats, lons, rvals = collect_ipp_points(records, sod_centre=sod, window_min=60.0)
        draw_panel(axes_flat[panel_i], lats, lons, rvals,
                   thai_paths, f"{h:02d}:00", date_str, doy, len(records))

        r, c = divmod(panel_i, NCOLS)
        if c > 0: axes_flat[panel_i].set_ylabel("")
        if r < NROWS-1: axes_flat[panel_i].set_xlabel("")

    cbar_ax = fig.add_axes([0.93, 0.07, 0.014, 0.85])
    cb = fig.colorbar(ScalarMappable(cmap=ROTI_CMAP, norm=Normalize(ROTI_MIN, ROTI_MAX)), cax=cbar_ax)
    cb.set_label("ROTI (TECU/min)", fontsize=9, labelpad=6)
    cb.ax.tick_params(labelsize=8)

    fig.suptitle(f"Ionospheric Irregularities by ROTI Index  —  {date_str}  ({len(records)} stations)",
                fontsize=12, fontweight="bold", y=0.975)

    PLOT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOT_OUT_DIR / f"ROTI_IPP_{date_str}.png"
    plt.savefig(out, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out.name}")

# =============================================================================
# MODE B — single-panel snapshots (5-minute, for GIF)
# =============================================================================

def plot_snapshots_5min(doy: int, records: list, thai_paths: list):
    date     = datetime(YEAR, 1, 1) + timedelta(days=doy - 1)
    date_str = date.strftime("%Y-%m-%d")

    # Folder name matches make_gif.py's expected "frames_5min_{date}" pattern
    frame_dir = PLOT_OUT_DIR / f"frames_5min_{date_str}"
    frame_dir.mkdir(parents=True, exist_ok=True)

    sod_bins = list(range(0, 86400, 300))   # 5-minute steps

    for frame_i, sod in enumerate(sod_bins):
        hh, mm = sod // 3600, (sod % 3600) // 60
        lats, lons, rvals = collect_ipp_points(records, sod_centre=sod, window_min=2.5)

        fig, ax = plt.subplots(figsize=(7, 7), facecolor="white")
        draw_panel(ax, lats, lons, rvals, thai_paths,
                  f"{hh:02d}:{mm:02d}", date_str, doy, len(records))

        cbar_ax = fig.add_axes([0.88, 0.08, 0.03, 0.82])
        cb = fig.colorbar(ScalarMappable(cmap=ROTI_CMAP, norm=Normalize(ROTI_MIN, ROTI_MAX)), cax=cbar_ax)
        cb.ax.tick_params(labelsize=8)
        cb.set_label("ROTI\n(TECU/min)", fontsize=8)

        out = frame_dir / f"frame_{frame_i:04d}_{hh:02d}h{mm:02d}m.png"
        plt.savefig(out, dpi=100, facecolor="white", bbox_inches="tight")
        plt.close(fig)

        if frame_i % 24 == 0:
            print(f"    {hh:02d}:{mm:02d} UTC  ({frame_i+1}/{len(sod_bins)})")

    print(f"  → {len(sod_bins)} frames saved in {frame_dir}")

# =============================================================================
# MAIN
# =============================================================================

HIGH_RES_MODE = True   # 5-min GIF frames, per request. Set False for
                       # 2-hour 12-panel daily summaries instead.

def main():
    print(f"Loaded {len(STATIONS)} stations from registry")
    print("Loading Thailand GeoJSON ...")
    thai_paths = load_thai_paths(str(GEOJSON_PATH))

    mode = "5-min snapshots" if HIGH_RES_MODE else "2-hour summary panels"
    print(f"Mode: {mode}")
    if HIGH_RES_MODE:
        print("  NOTE: ~213 stations x 288 frames/day is significantly "
              "heavier than the 17-station version -- test on 1-2 days "
              "first (set DOY_START = DOY_END) before running the full range.")
    print()

    for doy in range(DOY_START, DOY_END + 1):
        date = datetime(YEAR, 1, 1) + timedelta(days=doy - 1)
        print(f"DOY {doy:03d}  {date.strftime('%Y-%m-%d')}")

        records = load_all_stations(doy)
        print(f"  Stations with IPP+ROTI data: {len(records)}")

        if not records:
            print("  No data — skip")
            continue

        if HIGH_RES_MODE:
            plot_snapshots_5min(doy, records, thai_paths)
        else:
            plot_day_summary(doy, records, thai_paths)

    print(f"\nAll plots saved → {PLOT_OUT_DIR}")


if __name__ == "__main__":
    main()
