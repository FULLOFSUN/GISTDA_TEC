# =============================================================================
# plot_roti_ipp_map.py
# IPP scatter map coloured by ROTI — matching KMITL CSSRG style
#
# Each dot = one IPP position (ipp_lat, ipp_lon) at a given time,
# coloured by the ROTI value at that satellite-receiver pair.
# Grey dots = low ROTI (quiet), yellow→red = high ROTI (disturbed)
#
# Reads from:
#   TEC_output/{STATION}/DOY{DOY}/{STATION}_{DOY}_ipp_lat.csv
#   TEC_output/{STATION}/DOY{DOY}/{STATION}_{DOY}_ipp_lon.csv
#   TEC_output/{STATION}/DOY{DOY}/{STATION}_{DOY}_ROTI.csv
#
# Produces:
#   - One PNG per time step (5-min or 2-hour)
#   - OR a 12-panel daily summary
#
# Usage:
#   conda activate base
#   python plot_roti_ipp_map.py
# =============================================================================

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm, Normalize
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
from matplotlib.cm import ScalarMappable

# =============================================================================
# PATHS
# =============================================================================
TEC_OUTPUT_DIR = Path("/Users/ziraa/Downloads/GISTDATA/TEC_output")
GEOJSON_PATH   = Path("/Users/ziraa/Downloads/th.json")
PLOT_OUT_DIR   = TEC_OUTPUT_DIR / "ROTI_IPP_maps"

DOY_START = 14
DOY_END   = 30
YEAR      = 2026

# =============================================================================
# STATIONS
# =============================================================================
STATIONS = {
    "CHAN": (12.61031,  102.102411),
    "CHMA": (18.835275,  98.969956),
    "CNBR": (13.406019, 100.997652),
    "CPN":  (10.72466,   99.374356),
    "DPT":  (13.756782, 100.573200),
    "KMI":  (13.727832, 100.772429),
    "LPBR": (14.800907, 100.651246),
    "NKNY": (14.212003, 101.202211),
    "NKRM": (14.992119, 102.129470),
    "NKSW": (15.690637, 100.114112),
    "PJRK": (11.811621,  99.796348),
    "SISK": (15.116122, 104.285676),
    "SOKA": ( 7.206694, 100.596121),
    "SPBR": (14.518875, 100.130580),
    "SRTN": ( 9.132225,  99.331361),
    "UDON": (17.412732, 102.780704),
    "UTTD": (17.630094, 100.096343),
}

# =============================================================================
# COLORMAP — grey (low ROTI) → yellow → red (high ROTI), matching reference
# =============================================================================
ROTI_COLORS = [
    "#cccccc",   # 0.0  grey   (quiet)
    "#cccccc",   # 0.1  grey
    "#ffff00",   # 0.3  yellow (moderate)
    "#ffaa00",   # 0.5  amber
    "#ff4400",   # 0.7  orange-red
    "#cc0000",   # 0.9  red
    "#880000",   # 1.0  dark red (active)
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
# FILE FINDER
# =============================================================================

def find_csv(station: str, doy: int, prefix: str) -> str | None:
    base = TEC_OUTPUT_DIR / station / f"DOY{doy:03d}"
    for p in [base / f"{station}_{doy:03d}_{prefix}.csv",
              base / f"{prefix}_{station}_{doy:03d}.csv"]:
        if p.exists(): return str(p)
    if station == "CPN":
        for alt in [f"CPN1_{doy:03d}_{prefix}.csv",
                    f"{prefix}_CPN1_{doy:03d}.csv"]:
            p3 = TEC_OUTPUT_DIR/"CPN1"/f"DOY{doy:03d}"/alt
            if p3.exists(): return str(p3)
    matches = (list(base.glob(f"*{prefix}*{doy:03d}*.csv"))
               + list(base.glob(f"*{doy:03d}*{prefix}*.csv")))
    return str(matches[0]) if matches else None

# =============================================================================
# LOAD ONE STATION'S IPP + ROTI FOR ONE DAY
# =============================================================================

def load_station_day(station: str, doy: int) -> dict | None:
    ipp_lat_path = find_csv(station, doy, "ipp_lat")
    ipp_lon_path = find_csv(station, doy, "ipp_lon")
    roti_path    = find_csv(station, doy, "ROTI")

    if ipp_lat_path is None or ipp_lon_path is None or roti_path is None:
        print(f"  [missing] {station} DOY{doy:03d} — "
              f"ipp_lat={'OK' if ipp_lat_path else 'MISSING'}  "
              f"ipp_lon={'OK' if ipp_lon_path else 'MISSING'}  "
              f"ROTI={'OK' if roti_path else 'MISSING'}")
        return None
    try:
        ipp_lat = pd.read_csv(ipp_lat_path, header=0).values.astype(float)
        ipp_lon = pd.read_csv(ipp_lon_path, header=0).values.astype(float)
        roti    = pd.read_csv(roti_path,    header=0).values.astype(float)

        if ipp_lat.shape[1] != 32:
            print(f"  [skip] {station} DOY{doy:03d} unexpected cols: {ipp_lat.shape}")
            return None

        # detect cadence from row count (2880 rows = 30s, 86400 = 1s, etc.)
        n_rows  = ipp_lat.shape[0]
        cadence = max(1, 86400 // n_rows)
        print(f"  {station} DOY{doy:03d}: {n_rows} rows, cadence={cadence}s")

        # align row counts if they differ
        min_rows = min(ipp_lat.shape[0], ipp_lon.shape[0], roti.shape[0])
        ipp_lat  = ipp_lat[:min_rows, :]
        ipp_lon  = ipp_lon[:min_rows, :]
        roti     = roti[:min_rows, :]

        # clip ROTI to physical range
        roti = np.clip(roti, 0, ROTI_MAX)

        return {
            "station":  station,
            "ipp_lat":  ipp_lat,
            "ipp_lon":  ipp_lon,
            "roti":     roti,
            "cadence":  cadence,
        }
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
    """
    Returns arrays of (ipp_lat, ipp_lon, roti) for all visible IPPs
    within ±window_min of sod_centre, filtered to map extent.

    Handles any CSV cadence automatically (1s, 30s, 60s etc.) by
    detecting n_rows and computing cadence = 86400 / n_rows.
    """
    window_sec = int(window_min * 60)
    sod_lo = max(0,     sod_centre - window_sec)
    sod_hi = min(86399, sod_centre + window_sec)

    all_lat, all_lon, all_roti = [], [], []

    for rec in records:
        n_rows = rec["ipp_lat"].shape[0]

        # detect cadence: 86400 rows = 1s, 2880 rows = 30s, 1440 = 60s, etc.
        cadence = max(1, 86400 // n_rows)   # seconds per row

        # convert sod window to row indices
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
               thai_paths, time_label: str, date_str: str, doy: int):

    ax.set_facecolor("white")

    # ── Thailand border ───────────────────────────────────────────────────────
    for path in thai_paths:
        ax.add_patch(PathPatch(path, facecolor="none",
                               edgecolor="black", lw=0.9, zorder=4))

    # ── IPP scatter dots ──────────────────────────────────────────────────────
    if len(ipp_lat) > 0:
        # sort so high-ROTI dots plot on top
        order = np.argsort(roti_vals)
        ax.scatter(
            ipp_lon[order], ipp_lat[order],
            c=roti_vals[order],
            cmap=ROTI_CMAP,
            vmin=ROTI_MIN, vmax=ROTI_MAX,
            s=12, alpha=0.75, zorder=3,
            edgecolors="none",
            linewidths=0,
        )
    else:
        ax.text(0.5, 0.5, "no IPP data",
                transform=ax.transAxes, fontsize=7,
                color="#888888", ha="center", va="center")

    # ── time + date label ─────────────────────────────────────────────────────
    # local time = UTC + 7
    utc_h = int(time_label.split(":")[0]) if ":" in time_label else 0
    utc_m = int(time_label.split(":")[1]) if ":" in time_label else 0
    lt_h  = (utc_h + 7) % 24
    lt_m  = utc_m

    title = (f"Date: {date_str.replace('-','')}  DOY {doy:03d}  "
             f"ROTI(TECU/min) at  {utc_h:02d}:{utc_m:02d}:00 UTC "
             f"({lt_h:02d}:{lt_m:02d}:00 LT)")
    ax.set_title(title, fontsize=7.5, pad=4, color="black")

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

    fig = plt.figure(figsize=(NCOLS * 4.0, NROWS * 4.5 + 1.5),
                     facecolor="white")
    gs  = GridSpec(NROWS, NCOLS, figure=fig,
                   hspace=0.25, wspace=0.12,
                   left=0.06, right=0.91, top=0.93, bottom=0.07)

    axes_flat = [fig.add_subplot(gs[r, c])
                 for r in range(NROWS) for c in range(NCOLS)]

    for panel_i, h in enumerate(hour_bins):
        sod   = h * 3600
        lats, lons, rvals = collect_ipp_points(
            records, sod_centre=sod, window_min=60.0
        )
        time_label = f"{h:02d}:00"
        draw_panel(axes_flat[panel_i], lats, lons, rvals,
                   thai_paths, time_label, date_str, doy)

        r, c = divmod(panel_i, NCOLS)
        if c > 0: axes_flat[panel_i].set_ylabel("")
        if r < NROWS-1: axes_flat[panel_i].set_xlabel("")

    # shared colourbar
    cbar_ax = fig.add_axes([0.93, 0.07, 0.014, 0.85])
    cb = fig.colorbar(
        ScalarMappable(cmap=ROTI_CMAP,
                       norm=Normalize(vmin=ROTI_MIN, vmax=ROTI_MAX)),
        cax=cbar_ax
    )
    cb.set_label("ROTI (TECU/min)", fontsize=9, labelpad=6)
    cb.ax.tick_params(labelsize=8)

    fig.suptitle(
        f"Ionospheric Irregularities by ROTI Index  —  {date_str}  "
        f"({len(records)} stations)",
        fontsize=12, fontweight="bold", y=0.975
    )

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

    frame_dir = PLOT_OUT_DIR / f"frames_roti_{date_str}"
    frame_dir.mkdir(parents=True, exist_ok=True)

    sod_bins = list(range(0, 86400, 300))   # 5-minute steps

    for frame_i, sod in enumerate(sod_bins):
        hh  = sod // 3600
        mm  = (sod % 3600) // 60
        time_label = f"{hh:02d}:{mm:02d}"

        lats, lons, rvals = collect_ipp_points(
            records, sod_centre=sod, window_min=2.5
        )

        fig, ax = plt.subplots(figsize=(7, 7), facecolor="white")
        draw_panel(ax, lats, lons, rvals,
                   thai_paths, time_label, date_str, doy)

        # colourbar
        cbar_ax = fig.add_axes([0.88, 0.08, 0.03, 0.82])
        cb = fig.colorbar(
            ScalarMappable(cmap=ROTI_CMAP,
                           norm=Normalize(vmin=ROTI_MIN, vmax=ROTI_MAX)),
            cax=cbar_ax
        )
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

# Set to True for 5-minute GIF frames, False for 12-panel daily summary
HIGH_RES_MODE = True # False if it was the cadence of per 2 hour / 30s 

def main():
    print("Loading Thailand GeoJSON ...")
    thai_paths = load_thai_paths(str(GEOJSON_PATH))

    mode = "5-min snapshots" if HIGH_RES_MODE else "2-hour summary panels"
    print(f"Mode: {mode}")
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