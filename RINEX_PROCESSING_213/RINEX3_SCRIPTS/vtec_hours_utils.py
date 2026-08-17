# =============================================================================
# vtec_hour_utils.py
# Shared "collect one hour's station VTEC points, with outlier rejection"
# logic -- used by all the map-plotting scripts (TPS/IDW/Kriging/comparison/
# multistation) so there's one source of truth instead of five duplicated
# copies of collect_hour_points().
#
# Outlier rejection uses LOCAL nearest-neighbor comparison, not a single
# whole-network median. A national-median comparison falsely flags real
# regional spatial gradients (e.g. an EIA trough making the whole south of
# Thailand legitimately lower than the north at local noon) as if every
# station in that region were individually broken -- confirmed by a real
# diagnostic run, where ~20 southern stations all got flagged together at
# the exact same DOY/hour with the same-direction deviation, which is a
# coordinated regional signal, not independent station noise. Comparing
# each station only to its nearest K neighbors avoids this false-positive
# pattern while still catching genuinely isolated bad stations (confirmed
# separately: one station flagged as an outlier ~59% of all hours, and two
# stations with physically-impossible nighttime spikes >100 TECU).
# =============================================================================

import numpy as np

MAD_THRESHOLD = 3.5   # robust z-score cutoff; keep in sync with
                       # diagnose_station_outliers.py
K_NEIGHBORS   = 8     # how many nearest stations to compare against


def collect_hour_points(records: list, h_utc: int, window_h: float = 1.0,
                        reject_outliers: bool = True,
                        mad_threshold: float = MAD_THRESHOLD,
                        k_neighbors: int = K_NEIGHBORS):
    """
    records: list of {"station","lat","lon","vtec"} dicts (vtec shape (86400,32))
    Returns (lats, lons, vals) arrays for this hour, with LOCAL spatial
    outlier stations removed by default (compared against their nearest
    k_neighbors, not the whole network).
    """
    sod_lo = max(0, int((h_utc - window_h) * 3600))
    sod_hi = min(86399, int((h_utc + window_h) * 3600))

    stations, lats, lons, vals = [], [], [], []
    for rec in records:
        window = rec["vtec"][sod_lo:sod_hi + 1, :]
        med = np.nanmedian(window)
        if np.isfinite(med) and med > 0:
            stations.append(rec.get("station", "?"))
            lats.append(rec["lat"])
            lons.append(rec["lon"])
            vals.append(float(med))

    lats, lons, vals = np.array(lats), np.array(lons), np.array(vals)
    n = len(vals)

    if not reject_outliers or n < k_neighbors + 2:
        return lats, lons, vals

    pts = np.column_stack([lats, lons])
    keep = np.ones(n, dtype=bool)

    for i in range(n):
        d = np.sqrt(((pts - pts[i]) ** 2).sum(axis=1))
        d[i] = np.inf
        neighbor_idx = np.argsort(d)[:k_neighbors]

        neighbor_vals = vals[neighbor_idx]
        local_med = np.median(neighbor_vals)
        local_mad = np.median(np.abs(neighbor_vals - local_med))
        if local_mad < 1e-6:
            continue

        robust_z = 0.6745 * (vals[i] - local_med) / local_mad
        if abs(robust_z) > mad_threshold:
            keep[i] = False

    return lats[keep], lons[keep], vals[keep]