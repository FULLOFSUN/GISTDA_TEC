#!/usr/bin/env python3
"""
diagnose_station_outliers.py

Scans saved VTEC CSVs across a DOY range and flags stations whose values
are spatial outliers relative to their neighbors at the same hour --
the same signature as the CHAN nighttime-bias issue found earlier
(a single station producing an extreme value that distorts the whole
regional map, e.g. the bullseye artifact near ~99.5E, 20N seen in the
DOY 024 Kriging and DOY 028 IDW maps).

Method: for each DOY x hour bin, compute the median VTEC across all
stations with data, then flag any station whose value deviates from
that median by more than N robust standard deviations (using MAD --
same statistical approach tec_calculator.py's outlinecorr() already
uses for per-station time-series outliers, just applied spatially here
instead).

A station that gets flagged ONCE might just be a real, brief local
disturbance. A station that gets flagged repeatedly across many
hours/days is much more likely a genuine station-level problem (bad
antenna, multipath, local RFI, mis-surveyed position, etc.) -- this
script ranks stations by how often they're flagged so you can tell
the difference.

Usage:
    python diagnose_station_outliers.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter

from station_registry import load_station_registry

# =============================================================================
# CONFIG
# =============================================================================
BASE_DIR       = Path("/Users/ziraa/Documents/GISTDA/GISTDA_TEC")
TEC_OUTPUT_DIR = BASE_DIR / "TEC_output"
STATION_REGISTRY_CSV = BASE_DIR / "station_coords_registry.csv"

DOY_START = 14
DOY_END   = 30
YEAR      = 2026

HOUR_BINS = list(range(0, 24, 2))
MAD_THRESHOLD = 3.5   # robust z-score threshold (3.5 is a common outlier cutoff)
MIN_FLAG_COUNT = 3    # only report stations flagged at least this many times

STATIONS = load_station_registry(STATION_REGISTRY_CSV)

# =============================================================================
# DATA LOADER (same convention as the map-plotting scripts)
# =============================================================================

def find_vtec_csv(station: str, doy: int) -> str | None:
    base = TEC_OUTPUT_DIR / station / f"DOY{doy:03d}"
    p1 = base / f"{station}_{doy:03d}_VTEC.csv"
    if p1.exists(): return str(p1)
    if base.exists():
        matches = list(base.glob(f"*VTEC*{doy:03d}*.csv"))
        for m in matches:
            if m.exists(): return str(m)
    return None


def load_day(doy: int) -> dict:
    """station -> full-day VTEC array (86400, 32)."""
    records = {}
    for station in STATIONS:
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
            records[station] = vtec
        except Exception:
            continue
    return records


def hour_medians(records: dict, h_utc: int, window_h: float = 1.0) -> dict:
    sod_lo = max(0, int((h_utc - window_h) * 3600))
    sod_hi = min(86399, int((h_utc + window_h) * 3600))
    result = {}
    for station, vtec in records.items():
        med = np.nanmedian(vtec[sod_lo:sod_hi + 1, :])
        if np.isfinite(med) and med > 0:
            result[station] = float(med)
    return result

# =============================================================================
# ROBUST OUTLIER DETECTION (MAD-based, same style as outlinecorr())
# =============================================================================

def flag_outliers(stations: list, lats: np.ndarray, lons: np.ndarray,
                  vals: np.ndarray, k_neighbors: int = 8,
                  threshold: float = MAD_THRESHOLD) -> list:
    """
    [Fixed] Compares each station against its K nearest NEIGHBORS, not the
    whole-network median. A national median comparison falsely flags real
    regional spatial gradients (e.g. an EIA trough making the whole south
    of Thailand legitimately lower than the north at local noon) as if
    every station in that region were individually broken -- confirmed by
    a real run of this script, where ~20 southern stations all got flagged
    together at the exact same DOY/hour with the same-direction deviation,
    which is a coordinated regional signal, not independent station noise.

    Returns list of (station, value, robust_z, n_neighbors_used).
    """
    n = len(stations)
    if n < k_neighbors + 2:
        return []

    pts = np.column_stack([lats, lons])
    flagged = []

    for i in range(n):
        d = np.sqrt(((pts - pts[i]) ** 2).sum(axis=1))
        d[i] = np.inf  # exclude self
        neighbor_idx = np.argsort(d)[:k_neighbors]

        neighbor_vals = vals[neighbor_idx]
        local_med = np.median(neighbor_vals)
        local_mad = np.median(np.abs(neighbor_vals - local_med))
        if local_mad < 1e-6:
            continue

        robust_z = 0.6745 * (vals[i] - local_med) / local_mad
        if abs(robust_z) > threshold:
            flagged.append((stations[i], vals[i], robust_z, k_neighbors))

    return flagged

# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"Scanning DOY {DOY_START}-{DOY_END}, hour bins {HOUR_BINS}")
    print(f"MAD threshold: {MAD_THRESHOLD}  |  {len(STATIONS)} registered stations\n")

    flag_counter = Counter()
    flag_details = {}  # station -> list of (doy, hour, value, z, median_that_hour)

    for doy in range(DOY_START, DOY_END + 1):
        date = datetime(YEAR, 1, 1) + timedelta(days=doy - 1)
        records = load_day(doy)
        if not records:
            continue

        for h_utc in HOUR_BINS:
            vals_dict = hour_medians(records, h_utc)
            if len(vals_dict) < 5:
                continue

            stns_this_hour = list(vals_dict.keys())
            lats_arr = np.array([STATIONS[s][0] for s in stns_this_hour])
            lons_arr = np.array([STATIONS[s][1] for s in stns_this_hour])
            vals_arr = np.array([vals_dict[s] for s in stns_this_hour])

            flagged = flag_outliers(stns_this_hour, lats_arr, lons_arr, vals_arr)
            if not flagged:
                continue

            for stn, v, z, k in flagged:
                # local median of this station's own neighbors, for the log line
                idx = stns_this_hour.index(stn)
                d = np.sqrt((lats_arr - lats_arr[idx])**2 + (lons_arr - lons_arr[idx])**2)
                d[idx] = np.inf
                neighbor_idx = np.argsort(d)[:k]
                local_med = float(np.median(vals_arr[neighbor_idx]))

                flag_counter[stn] += 1
                flag_details.setdefault(stn, []).append(
                    (doy, h_utc, v, z, local_med)
                )

    if not flag_counter:
        print("No outlier stations found at this threshold.")
        return

    print(f"{'Station':<8} {'Times flagged':>14}  {'Lat':>8} {'Lon':>9}   Example (DOY, hour, value, nearest-8-neighbor median)")
    print("-" * 100)
    for stn, count in flag_counter.most_common():
        if count < MIN_FLAG_COUNT:
            continue
        lat, lon = STATIONS.get(stn, (float("nan"), float("nan")))
        example = flag_details[stn][0]
        doy_ex, h_ex, v_ex, z_ex, med_ex = example
        print(f"{stn:<8} {count:>14}  {lat:>8.4f} {lon:>9.4f}   "
              f"DOY{doy_ex:03d} {h_ex:02d}H: {v_ex:.1f} TECU vs local neighbor median {med_ex:.1f} TECU "
              f"(z={z_ex:.1f})")

    print(f"\n{'='*100}")
    print(f"Stations flagged >= {MIN_FLAG_COUNT} times are strong candidates for a "
          f"station-level QC issue (bad antenna, multipath, mis-surveyed\n"
          f"position, etc.) rather than genuine short-lived ionospheric variation "
          f"-- same category as the earlier CHAN nighttime-bias finding.")
    print(f"\nStations flagged only 1-2 times were left out of this summary "
          f"(MIN_FLAG_COUNT={MIN_FLAG_COUNT}) since that's more consistent with\n"
          f"real transient disturbance than a persistent station problem -- "
          f"lower MIN_FLAG_COUNT if you want to see those too.")


if __name__ == "__main__":
    main()