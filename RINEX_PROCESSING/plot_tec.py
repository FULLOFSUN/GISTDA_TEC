# =============================================================================
# plot_tec.py  —  Per-station daily plot
#
# NOTE: the 2-D regional map (across all stations) is now a separate step --
# see plot_tec_map_multistation.py, which reads the saved VTEC CSVs after
# process_tec.py finishes and builds the map with a real Thailand GeoJSON
# boundary. This file only needs to provide plot_daily_tec() for
# process_tec.py's per-station import.
# =============================================================================
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


# =============================================================================
# ROTI helper — works with corrected (cycle-slip-free) STECl_M_new output
# =============================================================================

def _prepare_roti(ROTI):
    """
    Collapse a (86400, 32) ROTI array into a single 1-D time series for plotting.

    The corrected tec_calculator now passes STECl_M_new into roticalculation(),
    so individual PRN traces are already free of cycle-slip spikes.  We take the
    nanmax across PRNs at each second so that real ionospheric events (which
    affect multiple PRNs simultaneously) still show up clearly — matching the
    reference plot style where a single sharp peak reaches ~1.0 TECU/min.

    A mild 30-second median filter is applied only to remove residual single-
    sample outliers; the window is kept short so genuine sharp peaks are
    preserved.
    """
    if ROTI is None:
        return None

    if ROTI.ndim == 2:
        # nanmax preserves real ionospheric peaks visible across multiple PRNs
        roti_1d = np.nanmax(ROTI, axis=1)
    else:
        roti_1d = np.array(ROTI, dtype=float)

    # 30-second median filter — short enough to keep sharp peaks intact
    from scipy.ndimage import median_filter
    valid = np.isfinite(roti_1d)
    if valid.sum() > 60:
        filled = np.where(valid, roti_1d, 0.0)
        roti_1d = median_filter(filled, size=30).astype(float)
        roti_1d[~valid] = np.nan

    return roti_1d


# =============================================================================
# PER-STATION DAILY PLOT
# =============================================================================

def plot_daily_tec(result, station, lat, lon,
                   doy, year, month, day,
                   save_path):
    """
    Two-panel plot: STEC (grey) + VTEC (colour per PRN) + red median, then ROTI.
    TEC arrays have shape (86400, 32) — one column per GPS satellite.
    """
    try:
        TEC  = result["TEC"]
        ROTI = result["ROTI"]
        vtec = TEC["vertical"]   # (86400, 32)
        stec = TEC["slant"]      # (86400, 32)
        t    = np.arange(0, 86400) / 3600

        fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                                 gridspec_kw={"hspace": 0.06})
        fig.patch.set_facecolor("white")

        COLORS = plt.cm.tab10(np.linspace(0, 1, 10))

        # -- TEC panel: STEC grey + VTEC coloured ----------------------------
        for prn_idx in range(vtec.shape[1]):
            v  = vtec[:, prn_idx]
            s  = stec[:, prn_idx]
            ok_v = np.isfinite(v)
            ok_s = np.isfinite(s)
            if ok_s.sum() > 10:
                axes[0].plot(t[ok_s], s[ok_s],
                             color="#c0c0c0", lw=0.5, alpha=0.45, zorder=2)
            if ok_v.sum() > 10:
                axes[0].plot(t[ok_v], v[ok_v],
                             color=COLORS[prn_idx % 10], lw=1.0,
                             alpha=0.75, zorder=3)

        # all-PRN median (red)
        vtec_med = np.nanmedian(vtec, axis=1)
        ok_med   = np.isfinite(vtec_med)
        if ok_med.sum() > 10:
            axes[0].plot(t[ok_med], vtec_med[ok_med],
                         color="red", lw=2.5, zorder=5, label="Median VTEC")

        # legend proxies only
        axes[0].plot([], [], color="#c0c0c0", lw=1.5, label="STEC (grey)")
        axes[0].plot([], [], color=COLORS[0],  lw=1.5, label="VTEC per PRN")
        axes[0].legend(fontsize=8, loc="upper right")
        axes[0].set_ylabel("TEC (TECU)", fontsize=10)
        axes[0].set_ylim(bottom=0)
        axes[0].set_xlim(0, 24)
        axes[0].grid(True, lw=0.4, alpha=0.5)
        axes[0].set_title(
            "TEC — " + station +
            "  (" + str(round(lat, 2)) + "N, " + str(round(lon, 2)) + "E)  " +
            str(year) + "-" + str(month).zfill(2) + "-" + str(day).zfill(2) +
            "  (DOY " + str(doy).zfill(3) + ")",
            fontsize=11, fontweight="bold"
        )

        # -- ROTI panel -------------------------------------------------------
        roti_1d = _prepare_roti(ROTI)

        if roti_1d is not None:
            ok_r = np.isfinite(roti_1d)
            axes[1].fill_between(
                t,
                0,
                np.where(ok_r, roti_1d, 0),
                color="black", linewidth=0, alpha=0.85
            )

        axes[1].set_ylabel("ROTI (TECU/min)", fontsize=10)
        axes[1].set_xlabel("UTC hour", fontsize=10)
        axes[1].set_ylim(0, 1)
        axes[1].set_xlim(0, 24)
        axes[1].grid(True, lw=0.4, alpha=0.5)
        axes[1].set_title("Rate of TEC change index (ROTI)", fontsize=10)
        axes[1].xaxis.set_major_locator(MultipleLocator(4))
        axes[1].xaxis.set_minor_locator(MultipleLocator(1))

        out = os.path.join(save_path,
                           "TEC_" + station + "_" + str(doy).zfill(3) + ".png")
        plt.tight_layout()
        plt.savefig(out, dpi=150, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        print("  -> " + os.path.basename(out))

    except Exception as e:
        import traceback
        print("  [plot_daily_tec] " + station + " DOY" + str(doy) + ": " + str(e))
        traceback.print_exc()
