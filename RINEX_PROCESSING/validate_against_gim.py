# =============================================================================
# validate_against_gim.py
# Validate your 17-station VTEC against IGS/CODE Global Ionosphere Maps (GIM)
#
# KEY POINT: the GIM is a single GLOBAL grid per epoch — the same file
# covers all 17 stations. You only need ONE IONEX file per DOY, not 17.
# The "per station" part is simply evaluating that one global grid at
# each station's exact lat/lon.
#
# Workflow:
#   1. Download/decompress ONE IONEX file per DOY (e.g. COD0OPSFIN_*.INX)
#   2. Parse all TEC maps in the file (usually 1-hourly or 2-hourly)
#   3. For each of your 17 stations, bilinearly interpolate the GIM grid
#      to (station_lat, station_lon) at each map epoch
#   4. Compare GIM VTEC vs your pipeline's VTEC at the same station/time
#   5. Report per-station AND pooled (all-station) bias/RMSE/correlation
#
# Usage:
#   conda activate base
#   python validate_against_gim.py
# =============================================================================

import re
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import RdYlGn
from matplotlib.colors import Normalize


# PATHS ===========================================================================
TEC_OUTPUT_DIR = Path("/Users/ziraa/Downloads/GISTDATA/TEC_output")
IONEX_DIR      = Path("/Users/ziraa/Downloads/GISTDATA/IONEX")   # put .INX files here
OUT_DIR        = TEC_OUTPUT_DIR

DOY_START = 14
DOY_END   = 30
YEAR      = 2026

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

# IONEX PARSER ======================================================================

def read_ionex(path: str) -> dict:
    """
    Parse an IONEX TEC file (CODE/IGS final GIM).
    Returns dict: {
        'epochs': [datetime, ...],
        'lat': np.array (n_lat,),
        'lon': np.array (n_lon,),
        'tec': np.array (n_epochs, n_lat, n_lon)   # in TECU
    }
    """
    with open(path, encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # ── header 
    exponent = -1
    lat1 = lat2 = dlat = None
    lon1 = lon2 = dlon = None
    hdr_end = 0

    for i, l in enumerate(lines):
        if "LAT1 / LAT2 / DLAT" in l:
            lat1, lat2, dlat = [float(x) for x in l[:60].split()]
        elif "LON1 / LON2 / DLON" in l:
            lon1, lon2, dlon = [float(x) for x in l[:60].split()]
        elif "EXPONENT" in l:
            exponent = int(l[:60].split()[0])
        elif "END OF HEADER" in l:
            hdr_end = i + 1
            break

    lat_arr = np.arange(lat1, lat2 + dlat/2, dlat)
    lon_arr = np.arange(lon1, lon2 + dlon/2, dlon)
    n_lat, n_lon = len(lat_arr), len(lon_arr)

    # ─--- TEC maps
    epochs, tec_maps = [], []
    i = hdr_end
    while i < len(lines):
        if "START OF TEC MAP" in lines[i]:
            i += 1
            # epoch line
            while "EPOCH OF CURRENT MAP" not in lines[i]:
                i += 1
            parts = lines[i].split()
            yy, mo, dd, hh, mm, ss = [int(x) for x in parts[:6]]
            epochs.append(datetime(yy, mo, dd, hh, mm, ss))
            i += 1

            grid = np.full((n_lat, n_lon), np.nan)
            for lat_i in range(n_lat):
                # find "LAT/LON1/LON2/DLON/H" header line
                while "LAT/LON1/LON2/DLON/H" not in lines[i]:
                    i += 1
                i += 1
                # read data values — 16 per line, 5 chars each (I5 format)
                vals = []
                while len(vals) < n_lon:
                    row = lines[i]
                    vals += [int(row[j:j+5]) for j in range(0, len(row.rstrip("\n")), 5)
                             if row[j:j+5].strip()]
                    i += 1
                grid[lat_i, :len(vals)] = vals[:n_lon]

            tec_maps.append(grid * (10 ** exponent))  # apply exponent scaling

            while "END OF TEC MAP" not in lines[i]:
                i += 1
            i += 1
        else:
            i += 1

    return {
        "epochs": epochs,
        "lat": lat_arr,
        "lon": lon_arr,
        "tec": np.array(tec_maps),   # (n_epochs, n_lat, n_lon) in TECU
    }


def gim_at_point(gim: dict, lat: float, lon: float, target_time: datetime) -> float:
    """
    Bilinear interpolation of GIM VTEC at (lat, lon) for the nearest epoch
    in time. Returns NaN if outside grid or time range.
    """
    # find nearest epoch (or interpolate between two)
    epochs = gim["epochs"]
    if not epochs:
        return np.nan

    diffs = [abs((e - target_time).total_seconds()) for e in epochs]
    idx   = int(np.argmin(diffs))
    grid  = gim["tec"][idx]   # (n_lat, n_lon)

    lat_arr, lon_arr = gim["lat"], gim["lon"]

    # GIM lat usually decreases (87.5 -> -87.5), handle both orders
    if lat_arr[0] > lat_arr[-1]:
        lat_idx_arr = lat_arr[::-1]
        grid_lat_ordered = grid[::-1, :]
    else:
        lat_idx_arr = lat_arr
        grid_lat_ordered = grid

    if not (lat_idx_arr[0] <= lat <= lat_idx_arr[-1]):
        return np.nan
    if not (lon_arr[0] <= lon <= lon_arr[-1]):
        return np.nan

    # bilinear interpolation
    i_lat = np.searchsorted(lat_idx_arr, lat) - 1
    i_lat = np.clip(i_lat, 0, len(lat_idx_arr) - 2)
    i_lon = np.searchsorted(lon_arr, lon) - 1
    i_lon = np.clip(i_lon, 0, len(lon_arr) - 2)

    lat0, lat1_ = lat_idx_arr[i_lat], lat_idx_arr[i_lat+1]
    lon0, lon1_ = lon_arr[i_lon],     lon_arr[i_lon+1]

    fy = (lat - lat0) / (lat1_ - lat0) if lat1_ != lat0 else 0
    fx = (lon - lon0) / (lon1_ - lon0) if lon1_ != lon0 else 0

    z00 = grid_lat_ordered[i_lat,   i_lon]
    z01 = grid_lat_ordered[i_lat,   i_lon+1]
    z10 = grid_lat_ordered[i_lat+1, i_lon]
    z11 = grid_lat_ordered[i_lat+1, i_lon+1]

    z0 = z00*(1-fx) + z01*fx
    z1 = z10*(1-fx) + z11*fx
    return float(z0*(1-fy) + z1*fy)


# REGIONAL PIPELINE'S VTEC LOADER =============================================

def find_vtec_csv(station: str, doy: int) -> str | None:
    base = TEC_OUTPUT_DIR / station / f"DOY{doy:03d}"
    for p in [base / f"{station}_{doy:03d}_VTEC.csv",
              base / f"VTEC_{station}_{doy:03d}.csv"]:
        if p.exists(): return str(p)
    if station == "CPN":
        for alt in [f"CPN1_{doy:03d}_VTEC.csv", f"VTEC_CPN1_{doy:03d}.csv"]:
            p3 = TEC_OUTPUT_DIR / "CPN1" / f"DOY{doy:03d}" / alt
            if p3.exists(): return str(p3)
    for m in (list(base.glob(f"*VTEC*{doy:03d}*.csv"))
              + list(base.glob(f"*{doy:03d}*VTEC*.csv"))):
        if m.exists(): return str(m)
    return None


def get_hourly_vtec(station: str, doy: int, hour: int) -> float:
    """Median VTEC for one station at one UTC hour (±30min window)."""
    path = find_vtec_csv(station, doy)
    if path is None:
        return np.nan
    try:
        df = pd.read_csv(path, header=0)
        vtec = df.values.astype(float)
        sod_lo = max(0, (hour*3600) - 1800)
        sod_hi = min(86399, (hour*3600) + 1800)
        med = np.nanmedian(vtec[sod_lo:sod_hi, :])
        return float(med) if np.isfinite(med) else np.nan
    except Exception:
        return np.nan

# MAIN VALIDATION LOOP ==================================

def find_ionex_file(doy: int) -> Path | None:
    """Find IONEX file for a DOY in IONEX_DIR."""
    date = datetime(YEAR, 1, 1) + timedelta(days=doy - 1)
    yy   = str(YEAR)[2:]
    candidates = [
        f"codg{doy:03d}0.{yy}i",                      # CODE classic final
        f"COD0OPSFIN_{YEAR}{doy:03d}0000_01D_01H_GIM.INX",  # CODE long-format

    ]
    for c in candidates:
        p = IONEX_DIR / c
        if p.exists():
            return p
    # wildcard fallback
    matches = list(IONEX_DIR.glob(f"*{doy:03d}0*")) + list(IONEX_DIR.glob(f"*{YEAR}{doy:03d}*"))
    return matches[0] if matches else None


def run_validation() -> pd.DataFrame:
    rows = []
    for doy in range(DOY_START, DOY_END + 1):
        date = datetime(YEAR, 1, 1) + timedelta(days=doy - 1)
        ionex_path = find_ionex_file(doy)
        if ionex_path is None:
            print(f"  DOY{doy:03d}: no IONEX file found — skip")
            continue

        print(f"  DOY{doy:03d}  {date.strftime('%Y-%m-%d')}  IONEX: {ionex_path.name}")
        gim = read_ionex(str(ionex_path))

        for hour in range(0, 24, 2):
            target_time = date + timedelta(hours=hour)
            for station, (lat, lon) in STATIONS.items():
                my_vtec  = get_hourly_vtec(station, doy, hour)
                gim_vtec = gim_at_point(gim, lat, lon, target_time)
                if np.isfinite(my_vtec) and np.isfinite(gim_vtec):
                    rows.append({
                        "doy": doy, "hour": hour, "station": station,
                        "my_vtec": my_vtec, "gim_vtec": gim_vtec,
                        "diff": my_vtec - gim_vtec,
                    })

    return pd.DataFrame(rows)

# STATISTICS — per station AND pooled =============================================================

def compute_stats(df: pd.DataFrame) -> tuple:
    """Returns (per_station_stats, pooled_stats)"""
    per_station = []
    for station in STATIONS:
        sub = df[df["station"] == station]
        if len(sub) == 0:
            continue
        bias = sub["diff"].mean()
        rmse = np.sqrt((sub["diff"]**2).mean())
        corr = sub["my_vtec"].corr(sub["gim_vtec"])
        per_station.append({
            "Station": station, "N": len(sub),
            "Bias (TECU)": round(bias, 2),
            "RMSE (TECU)": round(rmse, 2),
            "Correlation": round(corr, 3) if np.isfinite(corr) else None,
        })
    per_station_df = pd.DataFrame(per_station)

    pooled_bias = df["diff"].mean()
    pooled_rmse = np.sqrt((df["diff"]**2).mean())
    pooled_corr = df["my_vtec"].corr(df["gim_vtec"])
    pooled = {
        "N_total": len(df),
        "Pooled Bias (TECU)": round(pooled_bias, 3),
        "Pooled RMSE (TECU)": round(pooled_rmse, 3),
        "Pooled Correlation": round(pooled_corr, 3),
    }
    return per_station_df, pooled

# PLOTS =============================================================================

def plot_validation(df: pd.DataFrame, per_station: pd.DataFrame,
                    pooled: dict, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("white")

    # scatter: Regional VTEC vs GIM VTEC, coloured by station
    stations = list(STATIONS.keys())
    colors = plt.cm.tab20(np.linspace(0, 1, len(stations)))
    for stn, c in zip(stations, colors):
        sub = df[df["station"] == stn]
        axes[0].scatter(sub["gim_vtec"], sub["my_vtec"], s=8, alpha=0.5,
                        color=c, label=stn)
    lims = [0, max(df["gim_vtec"].max(), df["my_vtec"].max()) * 1.05]
    axes[0].plot(lims, lims, "k--", lw=1, alpha=0.6, label="1:1 line")
    axes[0].set_xlabel("GIM VTEC (TECU)")
    axes[0].set_ylabel("Regional VTEC (TECU)")
    axes[0].set_title(f"All stations vs GIM\nPooled bias={pooled['Pooled Bias (TECU)']:.2f}  "
                      f"RMSE={pooled['Pooled RMSE (TECU)']:.2f}  r={pooled['Pooled Correlation']:.3f}")
    axes[0].legend(fontsize=6, ncol=2, loc="upper left")
    axes[0].grid(True, lw=0.3, alpha=0.5)

    # bar chart: per-station bias
    axes[1].barh(per_station["Station"], per_station["Bias (TECU)"],
                color=["#d62728" if b > 0 else "#1f77b4"
                      for b in per_station["Bias (TECU)"]])
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].set_xlabel("Bias (Regional VTEC − GIM VTEC), TECU")
    axes[1].set_title("Per-station bias vs GIM")
    axes[1].grid(True, axis="x", lw=0.3, alpha=0.5)

    plt.tight_layout()
    out = out_dir / "GIM_validation_scatter_STORM.png"
    plt.savefig(out, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*60)
    print("VTEC validation against IGS/CODE GIM — all 17 stations")
    print(f"\nLooking for IONEX files in: {IONEX_DIR}")
    print("(one global file per DOY covers all stations)\n")

    df = run_validation()
    if df.empty:
        print("\nNo validation data — check IONEX_DIR has decompressed .INX/.i files")
        return

    print(f"\nTotal comparison points: {len(df):,}")

    per_station, pooled = compute_stats(df)
    print("\nPer-station statistics:")
    print(per_station.to_string(index=False))
    print("\nPooled (all stations) statistics:")
    for k, v in pooled.items():
        print(f"  {k}: {v}")

    df.to_csv(OUT_DIR / "GIM_validation_raw_STORM.csv", index=False)
    per_station.to_csv(OUT_DIR / "GIM_validation_per_station_STORM.csv", index=False)

    plot_validation(df, per_station, pooled, OUT_DIR)

    print("\nInterpretation guide:")
    print("  |bias| < 1 TECU   -> good, pipeline well-calibrated")
    print("  |bias| > 3 TECU   -> check DCB removal, receiver bias")
    print("  RMSE < 3 TECU     -> good agreement")
    print("  RMSE > 5 TECU     -> check interpolation / elevation mask")
    print("  One station only off -> local DCB or multipath issue at that site")
    print("  ALL stations same-sign bias -> systematic pipeline issue")


if __name__ == "__main__":
    main()
