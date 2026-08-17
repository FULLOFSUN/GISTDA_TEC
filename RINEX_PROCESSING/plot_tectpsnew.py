# =============================================================================
# plot_tec_map_tps.py
# Per-day regional VTEC map -- Thin-Plate Spline (RBF) only.
# ~213-station Thai CORS network. Run independently of the IDW/Kriging
# scripts to keep RAM usage down (one method per process).
#
# Usage:
#   python plot_tec_map_tps.py
# =============================================================================

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from scipy.interpolate import RBFInterpolator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
from matplotlib.cm import ScalarMappable

from station_registry import load_station_registry
from vtec_hours_utils import collect_hour_points

# =============================================================================
# PATHS  — edit if needed
# =============================================================================
BASE_DIR       = Path("/Users/ziraa/Documents/GISTDA/GISTDA_TEC")
TEC_OUTPUT_DIR = BASE_DIR / "TEC_output"
GEOJSON_PATH   = Path("/Users/ziraa/Downloads/GISTDA/th.json")
PLOT_OUT_DIR   = TEC_OUTPUT_DIR / "TEC_maps_comparison_new"
STATION_REGISTRY_CSV = BASE_DIR / "station_coords_registry.csv"

DOY_START = 14
DOY_END   = 30
YEAR      = 2026
HOUR_STEP = 2   # every 2h = 12 panels/day. Use 3 or 4 for a faster/lighter run.

METHOD = "tps"
METHOD_LABEL = "Thin-Plate Spline (RBF)"

STATIONS = load_station_registry(STATION_REGISTRY_CSV)

# =============================================================================
# COLORMAP
# =============================================================================
COLORS_TEC = [
    "#00008B", "#0000FF", "#00BFFF", "#00FFFF",
    "#00FF88", "#88FF00", "#FFFF00", "#FFB300",
    "#FF6600", "#FF2200", "#CC0000",
]
TEC_CMAP = LinearSegmentedColormap.from_list("tec_paper", COLORS_TEC, N=512)
VTEC_MIN, VTEC_MAX = 0, 100
TEC_NORM = BoundaryNorm(np.linspace(VTEC_MIN, VTEC_MAX, 101), TEC_CMAP.N)

LAT_MIN, LAT_MAX = 4.5, 22.0
LON_MIN, LON_MAX = 96.0, 107.5
GRID_N = 100

# =============================================================================
# GEOJSON / GRID
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
            verts.append(arr[0]); codes.append(MplPath.MOVETO)
            for pt in arr[1:]:
                verts.append(pt); codes.append(MplPath.LINETO)
            verts.append(arr[0]); codes.append(MplPath.CLOSEPOLY)
        if verts:
            result.append(MplPath(np.array(verts), np.array(codes)))
    return result


def build_grid():
    lat_g = np.linspace(LAT_MIN, LAT_MAX, GRID_N)
    lon_g = np.linspace(LON_MIN, LON_MAX, GRID_N)
    LON_G, LAT_G = np.meshgrid(lon_g, lat_g)
    grid_pts = np.column_stack([LAT_G.ravel(), LON_G.ravel()])
    return lat_g, lon_g, grid_pts, LAT_G, LON_G

# =============================================================================
# DATA LOADER
# =============================================================================

def find_vtec_csv(station: str, doy: int) -> str | None:
    base = TEC_OUTPUT_DIR / station / f"DOY{doy:03d}"
    p1 = base / f"{station}_{doy:03d}_VTEC.csv"
    if p1.exists(): return str(p1)
    matches = list(base.glob(f"*VTEC*{doy:03d}*.csv")) if base.exists() else []
    for m in matches:
        if m.exists(): return str(m)
    return None


def load_day(doy: int) -> list:
    records = []
    for station, (lat, lon) in STATIONS.items():
        path = find_vtec_csv(station, doy)
        if path is None:
            continue
        try:
            df = pd.read_csv(path, header=0)
            vtec = df.values.astype(float)
            if vtec.ndim != 2 or vtec.shape[1] != 32:
                continue
            if vtec.shape[0] < 86400:
                pad = np.full((86400 - vtec.shape[0], 32), np.nan)
                vtec = np.vstack([vtec, pad])
            records.append({"station": station, "lat": lat, "lon": lon, "vtec": vtec})
        except Exception as e:
            print(f"  [load error] {station} DOY{doy:03d}: {e}")
    return records


# =============================================================================
# INTERPOLATION
# =============================================================================

def interpolate_tps(lats, lons, vals, grid_pts, LAT_G):
    pts = np.column_stack([lats, lons])
    rbf = RBFInterpolator(pts, vals, kernel="thin_plate_spline",
                          smoothing=max(len(pts) * 1.5, 10))
    return np.clip(rbf(grid_pts).reshape(LAT_G.shape), VTEC_MIN, VTEC_MAX)

# =============================================================================
# DRAW ONE PANEL
# =============================================================================

def draw_panel(ax, lats, lons, vals, z, lat_g, lon_g, thai_paths, h_utc, n_used):
    ax.set_facecolor("white")
    if z is not None:
        ax.pcolormesh(lon_g, lat_g, z, cmap=TEC_CMAP, norm=TEC_NORM,
                      shading="gouraud", zorder=2)
        ax.scatter(lons, lats, c=vals, cmap=TEC_CMAP, norm=TEC_NORM,
                   s=14, zorder=6, edgecolors="black", linewidths=0.3, alpha=0.9)
    else:
        ax.set_facecolor(TEC_CMAP(TEC_NORM(0)))
        ax.text(0.5, 0.45, f"n={n_used}\nno data", transform=ax.transAxes,
                fontsize=5.5, color="white", ha="center", va="center")

    for path in thai_paths:
        ax.add_patch(PathPatch(path, facecolor="none", edgecolor="black",
                               lw=0.7, zorder=7))

    ax.text(0.97, 0.97, f"{h_utc:02d}H", transform=ax.transAxes,
            fontsize=9, fontweight="bold", color="white", ha="right", va="top",
            path_effects=[pe.withStroke(linewidth=2.2, foreground="black")])

    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.xaxis.set_major_locator(MultipleLocator(3))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.tick_params(labelsize=5, length=2, color="#555555", labelcolor="#333333")
    for sp in ax.spines.values():
        sp.set_edgecolor("#aaaaaa")

# =============================================================================
# BUILD ONE DAY'S FIGURE
# =============================================================================

def plot_day_map(doy, records, thai_paths, grid_info):
    lat_g, lon_g, grid_pts, LAT_G, LON_G = grid_info
    date = datetime(YEAR, 1, 1) + timedelta(days=doy - 1)
    date_str = date.strftime("%Y-%m-%d")

    hour_bins = list(range(0, 24, HOUR_STEP))
    NCOLS = 4
    NROWS = (len(hour_bins) + NCOLS - 1) // NCOLS

    fig = plt.figure(figsize=(NCOLS * 3.2, NROWS * 3.6 + 1.3), facecolor="white")
    gs = GridSpec(NROWS, NCOLS, figure=fig, hspace=0.06, wspace=0.04,
                  left=0.06, right=0.94, top=0.90, bottom=0.09)
    axes_flat = [fig.add_subplot(gs[r, c]) for r in range(NROWS) for c in range(NCOLS)]

    n_stn_max = 0
    for panel_i, h_utc in enumerate(hour_bins):
        ax = axes_flat[panel_i]
        lats, lons, vals = collect_hour_points(records, h_utc)
        n_stn_max = max(n_stn_max, len(vals))

        z = None
        if len(vals) >= 4:
            try:
                z = interpolate_tps(lats, lons, vals, grid_pts, LAT_G)
            except Exception as e:
                print(f"    [error] DOY{doy:03d} {h_utc:02d}H: {e}")

        draw_panel(ax, lats, lons, vals, z, lat_g, lon_g, thai_paths, h_utc, len(vals))

        r, c = divmod(panel_i, NCOLS)
        if c > 0: ax.set_yticklabels([])
        if r < NROWS - 1: ax.set_xticklabels([])

    for panel_i in range(len(hour_bins), NROWS * NCOLS):
        axes_flat[panel_i].axis("off")

    cbar_ax = fig.add_axes([0.06, 0.03, 0.88, 0.014])
    cb = fig.colorbar(ScalarMappable(cmap=TEC_CMAP, norm=TEC_NORM), cax=cbar_ax,
                      orientation="horizontal", ticks=np.arange(0, VTEC_MAX + 1, 10))
    cb.set_label("VTEC  (TECU)", fontsize=9, labelpad=3)
    cb.ax.tick_params(labelsize=8)

    fig.text(0.5, 0.965,
             f"Thailand VTEC Map ({METHOD_LABEL})  \u2014  {date_str}  "
             f"(up to {n_stn_max} stations)",
             ha="center", fontsize=12, fontweight="bold")

    PLOT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOT_OUT_DIR / f"VTEC_{METHOD.upper()}_{date_str}.png"
    plt.savefig(out, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out.name}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"=== {METHOD_LABEL} ===")
    print("Loading Thailand GeoJSON ...")
    thai_paths = load_thai_paths(str(GEOJSON_PATH))
    print(f"  {len(thai_paths)} polygon(s)")

    grid_info = build_grid()
    print(f"Grid: {GRID_N}x{GRID_N}  |  Hour step: {HOUR_STEP}h")

    for doy in range(DOY_START, DOY_END + 1):
        date = datetime(YEAR, 1, 1) + timedelta(days=doy - 1)
        print(f"\nDOY {doy:03d}  {date.strftime('%Y-%m-%d')}")
        records = load_day(doy)
        print(f"  Stations with data: {len(records)}")
        if not records:
            print("  [skip] no data")
            continue
        plot_day_map(doy, records, thai_paths, grid_info)

    print(f"\nDone -> {PLOT_OUT_DIR}")


if __name__ == "__main__":
    main()
