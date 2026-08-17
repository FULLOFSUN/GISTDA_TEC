# =============================================================================
# plot_continuous_timeseries.py
# Continuous regional VTEC time series — DOY 014–030, January 2026
#
# Shows:
#   - Median VTEC across all 17 stations at 1-minute resolution (continuous)
#   - Storm period shaded red (DOY 019–021)
#   - Optional: individual station lines as faint background
#   - Kp index as a secondary panel below
#
# Usage:
#   conda activate base
#   python plot_continuous_timeseries.py
# =============================================================================

import re
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

# =============================================================================
# PATHS
# =============================================================================
TEC_OUTPUT_DIR = Path("/Users/ziraa/Downloads/GISTDATA/TEC_output")
KP_FILE        = Path("/Users/ziraa/Downloads/GISTDATA/kp_index_historical_data.rtf")
OUT_DIR        = TEC_OUTPUT_DIR

DOY_START = 14
DOY_END   = 30
YEAR      = 2026

# storm period
STORM_DOY_START = 19
STORM_DOY_END   = 21

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
# FILE FINDERS
# =============================================================================

def find_csv(station: str, doy: int, prefix: str) -> str | None:
    base = TEC_OUTPUT_DIR / station / f"DOY{doy:03d}"
    for p in [base / f"{station}_{doy:03d}_{prefix}.csv",
              base / f"{prefix}_{station}_{doy:03d}.csv"]:
        if p.exists(): return str(p)
    if station == "CPN":
        for alt in [f"CPN1_{doy:03d}_{prefix}.csv",
                    f"{prefix}_CPN1_{doy:03d}.csv"]:
            p3 = TEC_OUTPUT_DIR / "CPN1" / f"DOY{doy:03d}" / alt
            if p3.exists(): return str(p3)
    for m in (list(base.glob(f"*{prefix}*{doy:03d}*.csv"))
              + list(base.glob(f"*{doy:03d}*{prefix}*.csv"))):
        if m.exists(): return str(m)
    return None

# =============================================================================
# BUILD CONTINUOUS TIME SERIES
# regional median VTEC at 1-minute cadence across DOY 014-030
# =============================================================================

def build_continuous_series(subsample: int = 60) -> pd.DataFrame:
    """
    subsample: seconds between kept rows (60 = 1-minute output)
    Returns DataFrame with columns: datetime, vtec_median, vtec_std,
                                     n_stations, + one col per station
    """
    all_times  = []
    all_median = []
    all_std    = []
    all_n      = []
    station_series = {s: [] for s in STATIONS}

    for doy in range(DOY_START, DOY_END + 1):
        date = datetime(YEAR, 1, 1) + timedelta(days=doy - 1)
        print(f"  DOY{doy:03d}  {date.strftime('%Y-%m-%d')}", end=" ")

        # load all station arrays for this day
        day_arrays = {}
        for station in STATIONS:
            path = find_csv(station, doy, "VTEC")
            if path is None:
                continue
            try:
                arr = pd.read_csv(path, header=0).values.astype(float)
                if arr.ndim == 2 and arr.shape[1] == 32:
                    # station median across PRNs: shape (86400,)
                    day_arrays[station] = np.nanmedian(arr, axis=1)
            except Exception:
                continue

        print(f"({len(day_arrays)} stations)")

        # build per-second regional stats
        n_rows = 86400
        for sod in range(0, n_rows, subsample):
            t = date + timedelta(seconds=int(sod))
            station_vals = []
            for station, arr in day_arrays.items():
                if sod < len(arr) and np.isfinite(arr[sod]):
                    station_vals.append(arr[sod])
                    station_series[station].append(
                        arr[sod] if np.isfinite(arr[sod]) else np.nan
                    )
                else:
                    station_series[station].append(np.nan)

            all_times.append(t)
            if station_vals:
                all_median.append(float(np.nanmedian(station_vals)))
                all_std.append(float(np.nanstd(station_vals)))
                all_n.append(len(station_vals))
            else:
                all_median.append(np.nan)
                all_std.append(np.nan)
                all_n.append(0)

    df = pd.DataFrame({
        "datetime":    all_times,
        "vtec_median": all_median,
        "vtec_std":    all_std,
        "n_stations":  all_n,
    })
    for s in STATIONS:
        df[s] = station_series[s]

    return df

# =============================================================================
# KP LOADER (from GFZ RTF file)
# =============================================================================

def load_kp(path: str) -> pd.DataFrame:
    rows = []
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.rstrip("\n").rstrip("\\").strip()
                parts = line.split()
                if len(parts) != 28 or not re.match(r"^\d{4}$", parts[0]):
                    continue
                try:
                    yr, mo, dd = int(parts[0]), int(parts[1]), int(parts[2])
                    kp_vals    = [float(x) for x in parts[7:15]]
                    for i, kp in enumerate(kp_vals):
                        rows.append({
                            "datetime": datetime(yr, mo, dd, i*3, 0, 0),
                            "Kp": kp
                        })
                except Exception:
                    continue
    except Exception:
        pass
    return pd.DataFrame(rows)

# =============================================================================
# PLOT
# =============================================================================

def plot_timeseries(df: pd.DataFrame, kp_df: pd.DataFrame, out_path: str,
                    show_individual: bool = True):

    has_kp = kp_df is not None and len(kp_df) > 0

    fig, axes = plt.subplots(
        3 if has_kp else 2, 1,
        figsize=(16, 10 if has_kp else 7),
        sharex=True,
        gridspec_kw={"height_ratios": ([2, 1, 1] if has_kp else [2, 1]),
                     "hspace": 0.06}
    )
    fig.patch.set_facecolor("white")

    storm_start = datetime(YEAR, 1, 1) + timedelta(days=STORM_DOY_START - 1)
    storm_end   = datetime(YEAR, 1, 1) + timedelta(days=STORM_DOY_END)

    # ── storm shading ──────────────────────────────────────────────────────
    for ax in axes:
        ax.axvspan(storm_start, storm_end,
                   color="#e74c3c", alpha=0.12, zorder=1, label="_nolegend_")

    # ── Panel 1: VTEC ──────────────────────────────────────────────────────
    ax = axes[0]

    # individual station lines (faint)
    if show_individual:
        for station in STATIONS:
            if station in df.columns:
                ax.plot(df["datetime"], df[station],
                        lw=0.4, alpha=0.25, color="#2980b9", zorder=2)

    # ± 1σ shaded band
    ok = np.isfinite(df["vtec_median"]) & np.isfinite(df["vtec_std"])
    ax.fill_between(df["datetime"][ok],
                    (df["vtec_median"] - df["vtec_std"])[ok],
                    (df["vtec_median"] + df["vtec_std"])[ok],
                    color="#2980b9", alpha=0.20, zorder=3,
                    label="μ ± 1σ (station spread)")

    # regional median — the main line
    ax.plot(df["datetime"][ok], df["vtec_median"][ok],
            color="#154360", lw=1.4, zorder=4,
            label="Regional median VTEC (17 stations)")

    ax.set_ylabel("Median VTEC\n(17 stations, TECU)", fontsize=10)
    ax.set_ylim(bottom=0)
    ax.grid(True, lw=0.35, alpha=0.45)
    ax.set_title(
        f"Regional TEC — Thailand CORS Network  "
        f"(DOY {DOY_START:03d}–{DOY_END:03d}, January {YEAR})",
        fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax.text((storm_start + (storm_end - storm_start)/2),
            ax.get_ylim()[1]*0.97,
            "Storm\n(G4)", fontsize=8, fontweight="bold",
            color="#c0392b", ha="center", va="top")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    # ── Panel 2: station-spread std ────────────────────────────────────────
    ax2 = axes[1]
    ok2 = np.isfinite(df["vtec_std"])
    ax2.fill_between(df["datetime"][ok2], 0, df["vtec_std"][ok2],
                     color="#8e44ad", alpha=0.6, zorder=3)
    ax2.set_ylabel("Station spread\nσ (TECU)", fontsize=10)
    ax2.set_ylim(bottom=0)
    ax2.grid(True, lw=0.35, alpha=0.45)
    for sp in ["top", "right"]:
        ax2.spines[sp].set_visible(False)

    # ── Panel 3: Kp ───────────────────────────────────────────────────────
    if has_kp:
        ax3 = axes[2]
        kp_range = kp_df[(kp_df["datetime"] >= df["datetime"].iloc[0]) &
                          (kp_df["datetime"] <= df["datetime"].iloc[-1])]
        kp_colors = []
        for kp in kp_range["Kp"]:
            if kp < 4:   kp_colors.append("#2ecc71")
            elif kp < 5: kp_colors.append("#f39c12")
            elif kp < 6: kp_colors.append("#e67e22")
            elif kp < 7: kp_colors.append("#e74c3c")
            else:        kp_colors.append("#8b0000")

        ax3.bar(kp_range["datetime"], kp_range["Kp"],
                width=timedelta(hours=2.9), color=kp_colors,
                align="edge", edgecolor="none", zorder=3)
        ax3.axhline(5, color="#e74c3c", lw=0.8, ls="--", alpha=0.7)
        ax3.set_ylabel("Kp index", fontsize=10)
        ax3.set_ylim(0, 9)
        ax3.set_yticks([0,3,5,7,9])
        ax3.grid(True, axis="y", lw=0.35, alpha=0.45)
        ax3.set_xlabel("Date (UTC)", fontsize=10)
        for sp in ["top", "right"]:
            ax3.spines[sp].set_visible(False)
    else:
        axes[-1].set_xlabel("Date (UTC)", fontsize=10)

    # ── x-axis ─────────────────────────────────────────────────────────────
    axes[-1].xaxis.set_major_locator(mdates.DayLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%-d"))
    axes[-1].xaxis.set_minor_locator(mdates.HourLocator(byhour=[6,12,18]))

    # day labels centred
    axes[-1].set_xlim(df["datetime"].iloc[0], df["datetime"].iloc[-1])

    plt.savefig(out_path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"\n  → {out_path}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*60)
    print(f"Continuous regional VTEC — DOY {DOY_START}–{DOY_END}")
    print("="*60)

    print("\nBuilding 1-minute time series ...")
    df = build_continuous_series(subsample=60)
    print(f"  Total rows: {len(df):,}")

    # cache to CSV so you can replot quickly
    cache = OUT_DIR / "continuous_vtec_1min.csv"
    df.to_csv(cache, index=False)
    print(f"  Cached → {cache.name}")

    print("\nLoading Kp index ...")
    kp_df = load_kp(str(KP_FILE)) if KP_FILE.exists() else pd.DataFrame()
    print(f"  {len(kp_df)} Kp records")

    out_path = str(OUT_DIR / "continuous_vtec_timeseries.png")
    plot_timeseries(df, kp_df, out_path, show_individual=True)


if __name__ == "__main__":
    # Quick replot from cache (skip rebuilding the series)
    import sys
    if "--replot" in sys.argv:
        cache = OUT_DIR / "continuous_vtec_1min.csv"
        if cache.exists():
            print("Loading cached series ...")
            df = pd.read_csv(cache)
            df["datetime"] = pd.to_datetime(df["datetime"])
            kp_df = load_kp(str(KP_FILE)) if KP_FILE.exists() else pd.DataFrame()
            plot_timeseries(df, kp_df,
                            str(OUT_DIR / "continuous_vtec_timeseries.png"))
        else:
            print("No cache found — run without --replot first")
    else:
        main()