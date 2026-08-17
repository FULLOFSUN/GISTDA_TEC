# =============================================================================
# plot_combined_storm.py
# Combined plot: Solar wind (Bz, speed, density) + Kp index + Regional VTEC
# January 14-30, 2026
#
# Reads:
#   - kp_index_historical_data.rtf      (GFZ Kp/Ap/F10.7)
#   - NOAA SWPC RTSW solar wind files   (mag + plasma, CSV or JSON)
#   - TEC_output/{STATION}/DOY{xxx}/{STATION}_{xxx}_VTEC.csv  (17 stations)
#
# Usage:
#   conda activate base
#   python plot_combined_storm.py
# =============================================================================

import re, json, glob
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

# =============================================================================
# PATHS — edit if needed
# =============================================================================
BASE_DIR       = Path("/Users/ziraa/Downloads/GISTDATA")
KP_FILE        = BASE_DIR / "kp_index_historical_data.rtf"
TEC_OUTPUT_DIR = BASE_DIR / "TEC_output"
OUT_DIR        = TEC_OUTPUT_DIR

# Solar wind RTSW file — single NOAA SWPC "rtsw_plot_data_*.txt" file
# covering the full DOY range (30-min resolution, -99999 = missing)
SOLAR_WIND_FILE = BASE_DIR / "rtsw_plot_data_2026-01-08T22_57_11.txt"

DOY_START = 14
DOY_END   = 30
YEAR      = 2026

STATIONS = [
    "CHAN","CHMA","CNBR","CPN","DPT","KMI","LPBR","NKNY","NKRM","NKSW",
    "PJRK","SISK","SOKA","SPBR","SRTN","UDON","UTTD",
]
# =============================================================================
# KP INDEX PARSER  (GFZ RTF format)
# =============================================================================

def parse_kp_rtf(path: str) -> pd.DataFrame:
    rows = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n").rstrip("\\").strip()
            if not line or line.startswith("#") or line.startswith("{") \
               or line.startswith("\\") or "rtf" in line.lower():
                continue
            parts = line.split()
            if len(parts) != 28 or not re.match(r"^\d{4}$", parts[0]):
                continue
            try:
                year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
                kp_vals = [float(x) for x in parts[7:15]]
                Ap      = int(parts[23])
            except (ValueError, IndexError):
                continue
            rows.append({"year": year, "month": month, "day": day,
                         "Kp": kp_vals, "Ap": Ap})
    return pd.DataFrame(rows)


def kp_to_3hourly(df: pd.DataFrame, year: int,
                  doy_start: int, doy_end: int) -> pd.DataFrame:
    start_date = datetime(year, 1, 1) + timedelta(days=doy_start - 1)
    end_date   = datetime(year, 1, 1) + timedelta(days=doy_end - 1)

    records = []
    for _, row in df.iterrows():
        date = datetime(row["year"], row["month"], row["day"])
        if not (start_date <= date <= end_date):
            continue
        for i in range(8):
            t = date + timedelta(hours=3 * i)
            records.append({"datetime": t, "Kp": row["Kp"][i]})
    return pd.DataFrame(records).sort_values("datetime")

# =============================================================================
# SOLAR WIND PARSER  (NOAA SWPC RTSW plot-data text file)
#
# Format: header lines starting with text, then data lines:
#   Timestamp(YYYY-MM-DD HH:MM:SS) Source Bt-med Bt-min Bt-max
#   Bx-med Bx-min Bx-max By-med By-min By-max Bz-med Bz-min Bz-max
#   Phi-mean Phi-min Phi-max Theta-med Theta-min Theta-max
#   Dens-med Dens-min Dens-max Speed-med Speed-min Speed-max
#   Temp-med Temp-min Temp-max
# Missing data is marked as -99999.
# =============================================================================

def load_solar_wind(rtsw_path: Path,
                    start: datetime, end: datetime) -> pd.DataFrame:
    """
    Returns DataFrame with columns: datetime, bz_gsm, speed, density
    Parsed using pd.read_csv with skiprows=14 (matches solarwind.py approach).
    """
    if not rtsw_path.exists():
        print(f"  [warn] solar wind file not found: {rtsw_path}")
        return pd.DataFrame(columns=["datetime", "bz_gsm", "speed", "density"])

    col_names = [
        "date", "time_col", "source",
        "Bt_med",  "Bt_min",  "Bt_max",
        "Bx_med",  "Bx_min",  "Bx_max",
        "By_med",  "By_min",  "By_max",
        "Bz_med",  "Bz_min",  "Bz_max",
        "Phi_mean","Phi_min", "Phi_max",
        "Theta_med","Theta_min","Theta_max",
        "Dens_med","Dens_min","Dens_max",
        "Speed_med","Speed_min","Speed_max",
        "Temp_med","Temp_min","Temp_max",
    ]

    try:
        df = pd.read_csv(
            rtsw_path, sep=r"\s+", skiprows=14,
            header=None, names=col_names,
        )
    except Exception as e:
        print(f"  [error] failed to read {rtsw_path}: {e}")
        return pd.DataFrame(columns=["datetime", "bz_gsm", "speed", "density"])

    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time_col"],
                                    errors="coerce")
    df = df.dropna(subset=["datetime"])

    for col in ["Bz_med", "Dens_med", "Speed_med"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] < -9000, col] = np.nan

    out = df[["datetime", "Bz_med", "Speed_med", "Dens_med"]].rename(
        columns={"Bz_med": "bz_gsm", "Speed_med": "speed", "Dens_med": "density"}
    )
    out = out.sort_values("datetime")
    out = out[(out["datetime"] >= start) & (out["datetime"] <= end)]
    return out

# =============================================================================
# VTEC LOADER  (regional median across 17 stations)
# =============================================================================

def find_vtec_csv(station: str, doy: int) -> str | None:
    base = TEC_OUTPUT_DIR / station / f"DOY{doy:03d}"
    p1 = base / f"{station}_{doy:03d}_VTEC.csv"
    if p1.exists(): return str(p1)
    p2 = base / f"VTEC_{station}_{doy:03d}.csv"
    if p2.exists(): return str(p2)
    if station == "CPN":
        for alt in [f"CPN1_{doy:03d}_VTEC.csv", f"VTEC_CPN1_{doy:03d}.csv"]:
            p3 = TEC_OUTPUT_DIR / "CPN1" / f"DOY{doy:03d}" / alt
            if p3.exists(): return str(p3)
    for m in (list(base.glob(f"*VTEC*{doy:03d}*.csv"))
              + list(base.glob(f"*{doy:03d}*VTEC*.csv"))):
        if m.exists(): return str(m)
    return None


def load_regional_vtec(doy_start: int, doy_end: int, year: int) -> pd.DataFrame:
    """
    Returns a DataFrame with columns: datetime, vtec_median
    vtec_median = nanmedian across all 17 stations and all PRNs,
    at native 1-second resolution, for DOY range.
    """
    records = []
    for doy in range(doy_start, doy_end + 1):
        date = datetime(year, 1, 1) + timedelta(days=doy - 1)
        day_arrays = []
        for station in STATIONS:
            path = find_vtec_csv(station, doy)
            if path is None:
                continue
            try:
                df   = pd.read_csv(path, header=0)
                vtec = df.values.astype(float)
                if vtec.ndim != 2 or vtec.shape[1] != 32:
                    continue
                if vtec.shape[0] < 86400:
                    pad = np.full((86400 - vtec.shape[0], 32), np.nan)
                    vtec = np.vstack([vtec, pad])
                # median across PRNs per second for this station
                station_med = np.nanmedian(vtec, axis=1)   # (86400,)
                day_arrays.append(station_med)
            except Exception:
                continue

        if not day_arrays:
            continue

        # median across stations per second
        stack = np.vstack(day_arrays)               # (n_stations, 86400)
        regional_med = np.nanmedian(stack, axis=0)   # (86400,)

        for sod in range(0, 86400, 60):   # downsample to 1-minute for plotting
            t = date + timedelta(seconds=sod)
            records.append({"datetime": t, "vtec_median": regional_med[sod]})

    return pd.DataFrame(records)

# =============================================================================
# COMBINED PLOT
# =============================================================================

def plot_combined(kp_3h: pd.DataFrame, sw: pd.DataFrame,
                  vtec: pd.DataFrame, doy_start: int, doy_end: int,
                  year: int, out_path: str):

    fig, axes = plt.subplots(4, 1, figsize=(15, 11), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1, 1.3],
                                          "hspace": 0.10})
    fig.patch.set_facecolor("white")

    start = datetime(year, 1, 1) + timedelta(days=doy_start - 1)
    end   = datetime(year, 1, 1) + timedelta(days=doy_end)

    # ── Panel 1: IMF Bz ───────────────────────────────────────────────────────
    if "bz_gsm" in sw.columns and sw["bz_gsm"].notna().any():
        axes[0].plot(sw["datetime"], sw["bz_gsm"], color="#1a6faf", lw=0.8)
        axes[0].axhline(0, color="gray", lw=0.6, ls="--")
        axes[0].fill_between(sw["datetime"], sw["bz_gsm"], 0,
                             where=(sw["bz_gsm"] < 0), color="#d62728", alpha=0.3)
    axes[0].set_ylabel("IMF Bz (nT)", fontsize=10)
    axes[0].grid(True, lw=0.4, alpha=0.4)
    axes[0].set_title(
        f"Space Weather Overview — DOY {doy_start:03d}–{doy_end:03d}  "
        f"({start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')})",
        fontsize=13, fontweight="bold"
    )

    # ── Panel 2: Solar wind speed & density ─────────────────────────────────
    ax2 = axes[1]
    if "speed" in sw.columns and sw["speed"].notna().any():
        ax2.plot(sw["datetime"], sw["speed"], color="#d62728", lw=0.8, label="Speed (km/s)")
    ax2.set_ylabel("SW Speed (km/s)", fontsize=10, color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax2.grid(True, lw=0.4, alpha=0.4)

    if "density" in sw.columns and sw["density"].notna().any():
        ax2b = ax2.twinx()
        ax2b.plot(sw["datetime"], sw["density"], color="#2ca02c", lw=0.8, alpha=0.7, label="Density (n/cc)")
        ax2b.set_ylabel("Density (n/cc)", fontsize=10, color="#2ca02c")
        ax2b.tick_params(axis="y", labelcolor="#2ca02c")

    # ── Panel 3: Kp index ────────────────────────────────────────────────────
    colors = []
    for kp in kp_3h["Kp"]:
        if kp < 4:   colors.append("#2ca02c")
        elif kp < 5: colors.append("#ffbf00")
        elif kp < 6: colors.append("#ff7f0e")
        elif kp < 7: colors.append("#d62728")
        else:        colors.append("#7f0000")

    axes[2].bar(kp_3h["datetime"], kp_3h["Kp"], width=0.1,
                color=colors, align="edge", edgecolor="none")
    axes[2].axhline(5, color="orange", lw=0.8, ls="--", alpha=0.6)
    axes[2].set_ylabel("Kp index", fontsize=10)
    axes[2].set_ylim(0, 9)
    axes[2].set_yticks(range(0, 10, 2))
    axes[2].grid(True, axis="y", lw=0.4, alpha=0.4)

    legend_patches = [
        Patch(color="#2ca02c", label="Quiet"),
        Patch(color="#ffbf00", label="Active"),
        Patch(color="#ff7f0e", label="Minor storm"),
        Patch(color="#d62728", label="Moderate storm"),
        Patch(color="#7f0000", label="Severe storm"),
    ]
    axes[2].legend(handles=legend_patches, fontsize=7, loc="upper right", ncol=5)

    # ── Panel 4: Regional median VTEC ────────────────────────────────────────
    axes[3].plot(vtec["datetime"], vtec["vtec_median"],
                 color="#1a6faf", lw=1.0)
    axes[3].set_ylabel("Median VTEC\n(17 stations, TECU)", fontsize=10)
    axes[3].set_xlabel("Date (UTC)", fontsize=10)
    axes[3].set_ylim(bottom=0)
    axes[3].grid(True, lw=0.4, alpha=0.4)

    # x-axis: daily ticks across the whole range
    for ax in axes:
        ax.set_xlim(start, end)
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d"))
        ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[12]))

    plt.savefig(out_path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    start = datetime(YEAR, 1, 1) + timedelta(days=DOY_START - 1)
    end   = datetime(YEAR, 1, 1) + timedelta(days=DOY_END)

    print("Loading Kp index ...")
    kp_df = parse_kp_rtf(str(KP_FILE))
    kp_3h = kp_to_3hourly(kp_df, YEAR, DOY_START, DOY_END)
    print(f"  {len(kp_3h)} 3-hourly Kp values")

    print("Loading solar wind data ...")
    sw = load_solar_wind(SOLAR_WIND_FILE, start, end)
    print(f"  {len(sw)} solar wind records  columns={list(sw.columns)}")

    print("Loading regional VTEC (17 stations, median) ...")
    vtec = load_regional_vtec(DOY_START, DOY_END, YEAR)
    print(f"  {len(vtec)} VTEC time points")

    out_path = f"{OUT_DIR}/Combined_storm_DOY{DOY_START:03d}-{DOY_END:03d}.png"
    plot_combined(kp_3h, sw, vtec, DOY_START, DOY_END, YEAR, out_path)


if __name__ == "__main__":
    main()