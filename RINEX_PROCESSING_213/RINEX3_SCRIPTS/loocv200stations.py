# =============================================================================
# cross_validation.py
# Leave-One-Out Cross-Validation (LOOCV) for 3 TEC mapping methods:
#   IDW, TPS/RBF, Ordinary Kriging
#
# SCHA+IRI removed from this version (dropped per request) -- this also
# removes the single heaviest part of the LOOCV loop, since SCHA required
# re-fitting a spherical-cap-harmonic design matrix on every excluded-
# station call. IDW/TPS/Kriging are notably faster without it, though
# Kriging's variogram fit is still the slowest of the remaining three,
# especially at ~213 stations vs. the original 17.
#
# Updated from the original 17-station-only version:
#   - STATIONS dict replaced with the shared station_registry.py loader
#     (same registry process_tec.py and the plotting scripts use), so this
#     now runs LOOCV across whichever stations are in
#     station_coords_registry.csv -- 17 or ~213.
#   - TEC_OUTPUT_DIR updated to the current GISTDA_TEC base path.
#
# For each station i:
#   1. Remove station i from the observation set
#   2. Interpolate using the remaining stations
#   3. Compare predicted VTEC vs observed VTEC at station i
#
# Statistics computed per method:
#   ME    : Mean Error  (bias)
#   MAE   : Mean Absolute Error
#   RMSE  : Root Mean Square Error
#   MSE   : Mean Square Error
#   RMSSE : Root Mean Square Standardised Error (geostatistical only)
#   CI95  : % of predictions within ±1.96σ of the mean error (95% confidence)
#
# Outputs:
#   - cross_validation_errors.csv   (per-station per-method errors)
#   - cross_validation_stats.csv    (summary statistics)
#   - cross_validation_table.png    (coloured summary table matching paper style)
# =============================================================================

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from scipy.interpolate import RBFInterpolator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import RdYlGn

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

# Station registry (lat, lon) -- loaded from the same CSV as the rest of
# the pipeline, instead of a hardcoded 17-station dict
STATIONS      = load_station_registry(STATION_REGISTRY_CSV)
STATION_NAMES = list(STATIONS.keys())
N_STATIONS    = len(STATION_NAMES)

# =============================================================================
# DATA LOADER
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


def load_hourly_medians(doy: int, h_utc: int,
                        window_h: float = 1.0) -> dict:
    """Load per-station VTEC median for a 2-hour window."""
    sod_lo = max(0,     int((h_utc - window_h) * 3600))
    sod_hi = min(86399, int((h_utc + window_h) * 3600))
    result = {}
    for station in STATION_NAMES:
        path = find_vtec_csv(station, doy)
        if path is None: continue
        try:
            df   = pd.read_csv(path, header=0)
            vtec = df.values.astype(float)
            if vtec.shape[0] < sod_hi: continue
            med  = np.nanmedian(vtec[sod_lo:sod_hi, :])
            if np.isfinite(med) and med > 0:
                result[station] = float(med)
        except Exception:
            continue
    return result

# =============================================================================
# INTERPOLATION METHODS (single-point prediction version for LOOCV)
# UNCHANGED from the original -- only station loading/paths were updated,
# and predict_scha() + its helpers were removed.
# =============================================================================

def predict_idw(train_lats, train_lons, train_vals,
                pred_lat, pred_lon, power=2, eps=1e-6):
    mean_lat = np.radians(np.mean(train_lats))
    dlat = pred_lat - train_lats
    dlon = (pred_lon - train_lons) * np.cos(mean_lat)
    d    = np.sqrt(dlat**2 + dlon**2)
    d    = np.maximum(d, eps)
    w    = 1.0 / d**power
    return float((w * train_vals).sum() / w.sum())


def predict_tps(train_lats, train_lons, train_vals,
                pred_lat, pred_lon):
    pts  = np.column_stack([train_lats, train_lons])
    rbf  = RBFInterpolator(pts, train_vals, kernel="thin_plate_spline",
                           smoothing=max(len(pts) * 1.5, 10))
    pred = rbf(np.array([[pred_lat, pred_lon]]))
    return float(pred[0])


def _variogram_spherical(d, sill, range_, nugget):
    range_ = max(range_, 1e-6)
    g = np.where(d <= range_,
                 nugget + (sill-nugget)*(1.5*(d/range_)-0.5*(d/range_)**3),
                 sill)
    return np.where(d <= 1e-9, 0.0, g)


def _fit_variogram(pts, vals):
    from scipy.optimize import curve_fit
    n  = len(pts)
    d_mat = np.sqrt(((pts[:,None,:]-pts[None,:,:])**2).sum(-1))
    v_mat = 0.5*(vals[:,None]-vals[None,:])**2
    iu = np.triu_indices(n,k=1)
    d_emp = d_mat[iu]; v_emp = v_mat[iu]
    sill_d  = max(np.var(vals), 1e-3)
    range_d = max(d_emp.max()*0.5, 1e-3) if len(d_emp) else 1.0
    nugget_d = sill_d*0.1
    try:
        popt,_ = curve_fit(_variogram_spherical, d_emp, v_emp,
                           p0=[sill_d,range_d,nugget_d],
                           bounds=([0,1e-3,0],[sill_d*5,d_emp.max()*3,sill_d]))
        if np.isfinite(popt).all() and popt[0]>0:
            return tuple(popt)
    except Exception: pass
    return sill_d, range_d, nugget_d


def predict_kriging(train_lats, train_lons, train_vals, pred_lat, pred_lon):
    pts  = np.column_stack([train_lats, train_lons])
    sill, range_, nugget = _fit_variogram(pts, train_vals)
    n    = len(pts)
    d_pp = np.sqrt(((pts[:,None,:]-pts[None,:,:])**2).sum(-1))
    g_pp = _variogram_spherical(d_pp, sill, range_, nugget)
    K    = np.ones((n+1,n+1)); K[:n,:n]=g_pp; K[n,n]=0
    np.fill_diagonal(K[:n,:n], 0)
    try:    K_inv = np.linalg.inv(K)
    except: K_inv = np.linalg.pinv(K)
    pred_pt  = np.array([[pred_lat, pred_lon]])
    d_pred   = np.sqrt(((pred_pt-pts)**2).sum(-1))
    g_pred   = _variogram_spherical(d_pred, sill, range_, nugget)
    rhs      = np.append(g_pred, 1.0)
    weights  = K_inv @ rhs
    z_pred   = float(weights[:n] @ train_vals)
    krig_var = float(rhs @ weights)
    return z_pred, max(krig_var, 0)

# =============================================================================
# LOOCV  — runs over all DOYs × all hour bins × all stations × all methods
# =============================================================================

def run_loocv(hour_bins=None, verbose=True):
    """
    hour_bins : list of UTC hours to evaluate, e.g. [0,2,4,...,22]
                Default: all 12 bins (0,2,...,22)
    Returns   : DataFrame with columns
                doy, hour, station, obs, pred_idw, pred_tps,
                pred_krig, krig_var

    NOTE: at ~213 stations this is slower than the original 17-station
    run, mainly due to predict_kriging()'s per-call variogram fit. Test
    with a small DOY range / fewer hour bins first (see main()).
    """
    if hour_bins is None:
        hour_bins = list(range(0, 24, 2))

    all_rows = []

    for doy in range(DOY_START, DOY_END+1):
        date = datetime(YEAR,1,1)+timedelta(days=doy-1)
        if verbose: print(f"  DOY{doy:03d}  {date.strftime('%Y-%m-%d')}")

        for h_utc in hour_bins:
            data = load_hourly_medians(doy, h_utc)
            stn_with_data = [s for s in STATION_NAMES if s in data]
            if len(stn_with_data) < 5: continue

            vals_all = np.array([data[s] for s in stn_with_data])
            lats_h   = np.array([STATIONS[s][0] for s in stn_with_data])
            lons_h   = np.array([STATIONS[s][1] for s in stn_with_data])
            n        = len(stn_with_data)

            if verbose and h_utc == hour_bins[0]:
                print(f"    {n} stations with data this hour")

            for i, stn in enumerate(stn_with_data):
                obs   = vals_all[i]
                mask  = np.ones(n, dtype=bool); mask[i] = False
                tl    = lats_h[mask]; tnl = lons_h[mask]; tv = vals_all[mask]

                row = {"doy":doy,"hour":h_utc,"station":stn,"obs":obs}

                try: row["pred_idw"]  = predict_idw(tl,tnl,tv, lats_h[i],lons_h[i])
                except: row["pred_idw"] = np.nan

                try: row["pred_tps"]  = predict_tps(tl,tnl,tv, lats_h[i],lons_h[i])
                except: row["pred_tps"] = np.nan

                try:
                    z_k, var_k = predict_kriging(tl,tnl,tv, lats_h[i],lons_h[i])
                    row["pred_krig"] = z_k; row["krig_var"] = var_k
                except: row["pred_krig"] = np.nan; row["krig_var"] = np.nan

                all_rows.append(row)

    return pd.DataFrame(all_rows)

# =============================================================================
# STATISTICS
# =============================================================================

def compute_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ME, MAE, MSE, RMSE, RMSSE, CI95 for each method."""
    methods = {
        "IDW":    "pred_idw",
        "TPS/RBF":"pred_tps",
        "Kriging":"pred_krig",
    }
    rows = []

    for method_name, col in methods.items():
        sub = df.dropna(subset=[col])
        errors = sub[col].values - sub["obs"].values
        n      = len(errors)
        me     = errors.mean()
        mae    = np.abs(errors).mean()
        mse    = (errors**2).mean()
        rmse   = np.sqrt(mse)
        std_e  = errors.std()

        # 95% confidence interval: |error| ≤ 1.96 * std
        within = np.abs(errors) <= 1.96 * std_e
        ci95_n = int(within.sum())
        ci95_p = 100.0 * ci95_n / n

        # RMSSE (geostatistical: uses kriging variance as denominator)
        if method_name == "Kriging" and "krig_var" in df.columns:
            kv_sub = df.dropna(subset=[col,"krig_var"])
            if len(kv_sub) > 0:
                std_errors = (kv_sub[col].values - kv_sub["obs"].values) / \
                             np.sqrt(np.maximum(kv_sub["krig_var"].values, 1e-6))
                rmsse = float(np.sqrt((std_errors**2).mean()))
            else:
                rmsse = np.nan
        else:
            rmsse = np.nan

        rows.append({
            "Method":  method_name,
            "N":       n,
            "ME":      round(me,   3),
            "MAE":     round(mae,  3),
            "MSE":     round(mse,  3),
            "RMSE":    round(rmse, 3),
            "RMSSE":   round(rmsse,3) if np.isfinite(rmsse) else "—",
            "CI95_N":  ci95_n,
            "CI95_%":  round(ci95_p,1),
        })

    return pd.DataFrame(rows)

# =============================================================================
# PLOT — coloured summary table  (matches paper style from screenshot)
# =============================================================================

def plot_summary_table(stats: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    fig.patch.set_facecolor("white")
    ax.axis("off")

    display_cols = ["Method","N","ME (TECU)","MAE (TECU)",
                    "RMSE (TECU)","RMSSE","CI95 N","CI95 %"]
    stats_display = stats.rename(columns={
        "ME":"ME (TECU)","MAE":"MAE (TECU)","RMSE":"RMSE (TECU)","RMSSE":"RMSSE",
        "CI95_N":"CI95 N","CI95_%":"CI95 %"
    })[display_cols]

    cell_text = [row.tolist() for _, row in stats_display.iterrows()]

    tbl = ax.table(
        cellText=cell_text,
        colLabels=display_cols,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.2, 2.0)

    col_idx = {c: i for i, c in enumerate(display_cols)}
    n_methods = len(stats)

    def rank_color(values, higher_is_better=True):
        vals_num = pd.to_numeric(pd.Series(values), errors="coerce")
        if vals_num.isna().all():
            return ["#ffffff"]*len(values)
        norm = Normalize(vmin=vals_num.min(), vmax=vals_num.max())
        cmap = RdYlGn if higher_is_better else RdYlGn.reversed()
        return [matplotlib.colors.to_hex(cmap(norm(v)))
                if np.isfinite(v) else "#eeeeee" for v in vals_num]

    ci95_colors  = rank_color(stats["CI95_%"].values,     higher_is_better=True)
    rmse_colors  = rank_color(stats["RMSE"].values,       higher_is_better=False)
    mae_colors   = rank_color(stats["MAE"].values,        higher_is_better=False)
    me_colors    = rank_color(np.abs(stats["ME"].values), higher_is_better=False)

    for row_i in range(n_methods):
        tbl[(row_i+1, col_idx["CI95 %"])].set_facecolor(ci95_colors[row_i])
        tbl[(row_i+1, col_idx["RMSE (TECU)"])].set_facecolor(rmse_colors[row_i])
        tbl[(row_i+1, col_idx["MAE (TECU)"])].set_facecolor(mae_colors[row_i])
        tbl[(row_i+1, col_idx["ME (TECU)"])].set_facecolor(me_colors[row_i])

    for col_i in range(len(display_cols)):
        tbl[(0, col_i)].set_facecolor("#2c3e50")
        tbl[(0, col_i)].set_text_props(color="white", fontweight="bold")

    for row_i in range(1, n_methods+1):
        tbl[(row_i, 0)].set_text_props(fontweight="bold")
        tbl[(row_i, 0)].set_facecolor("#f5f5f5")

    ax.set_title(
        f"Cross-Validation Statistics — LOOCV across all DOY {DOY_START:03d}\u2013{DOY_END:03d}\n"
        f"{N_STATIONS} stations  \u00b7  Colour: Green = best  ·  Red = worst  (per column)",
        fontsize=12, fontweight="bold", pad=20, color="#111111"
    )

    green_p = mpatches.Patch(color="#1a9850", label="Best")
    yel_p   = mpatches.Patch(color="#fee08b", label="Middle")
    red_p   = mpatches.Patch(color="#d73027", label="Worst")
    ax.legend(handles=[green_p,yel_p,red_p], loc="lower right",
              fontsize=9, framealpha=0.85)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*60)
    print("Leave-One-Out Cross-Validation — IDW / TPS / Kriging")
    print(f"DOY {DOY_START}–{DOY_END}, {N_STATIONS} stations")
    print("="*60)

    # *** For a first test run, uncomment these to cut runtime dramatically
    # before committing to the full DOY 14-30 / 12-hour-bin run: ***
    # global DOY_START, DOY_END
    # DOY_START, DOY_END = 19, 19
    # hour_bins = list(range(0, 24, 6))   # 4 bins instead of 12
    hour_bins = list(range(0, 24, 2))

    print("\nRunning LOOCV ...")
    df = run_loocv(hour_bins=hour_bins, verbose=True)
    print(f"\nTotal LOOCV samples: {len(df):,}")

    raw_out = str(OUT_DIR / "cross_validation_errors.csv")
    df.to_csv(raw_out, index=False)
    print(f"Raw errors saved → {raw_out}")

    stats = compute_stats(df)
    print("\nSummary Statistics:")
    print(stats.to_string(index=False))

    stats_out = str(OUT_DIR / "cross_validation_stats.csv")
    stats.to_csv(stats_out, index=False)
    print(f"\nStats saved → {stats_out}")

    table_out = str(OUT_DIR / "cross_validation_table.png")
    plot_summary_table(stats, table_out)

    print("\nDone.")


if __name__ == "__main__":
    main()