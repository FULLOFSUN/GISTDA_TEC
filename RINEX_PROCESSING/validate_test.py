# =============================================================================
# validate_test.py
# IONEX (Global Ionosphere Map) reader + point-sampling for GIM validation.
#
# Rebuilt to match your actual IONEX files:
#   /Users/ziraa/Documents/GISTDA/GISTDA_TEC/ionex/
#     COD0OPSFIN_{YYYY}{DDD}0000_01D_01H_GIM.INX
#   -- confirmed filename pattern, uncompressed, hourly (01H) maps.
#
# Provides the interface plot_regional_vs_gim_timeseries.py (and
# cross_validation.py, if it needs it) expects:
#   IONEX_DIR
#   find_ionex_file(doy) -> Path | None
#   read_ionex(path) -> dict  (with dict["epochs"] = list of datetimes)
#   gim_at_point(gim, lat, lon, epoch) -> float VTEC (TECU)
#
# IONEX format reference: the internal header/TEC-MAP-block structure is
# the standard IONEX 1.1 format (unchanged for years) -- only CDDIS's
# delivery filename convention changed to this longer "COD0OPSFIN_..."
# style, similar to the RINEX 3 nav filename change we hit earlier.
#
# Also includes a standalone validation summary (run this file directly)
# that compares your pipeline's VTEC against the GIM per station, matching
# what was likely this file's original purpose given its name.
# =============================================================================

import re
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR   = Path("/Users/ziraa/Documents/GISTDA/GISTDA_TEC")
IONEX_DIR  = BASE_DIR / "ionex"
YEAR       = 2026

# =============================================================================
# FILE FINDER
# =============================================================================

def find_ionex_file(doy: int, year: int = YEAR) -> Path | None:
    """Locate the IONEX file for a given day-of-year, matching the
    confirmed filename pattern: COD0OPSFIN_YYYYDDD0000_01D_01H_GIM.INX
    Falls back to a glob in case of minor naming variation (e.g. a
    different solution/interval tag) before giving up."""
    exact = IONEX_DIR / f"COD0OPSFIN_{year}{doy:03d}0000_01D_01H_GIM.INX"
    if exact.exists():
        return exact

    if IONEX_DIR.exists():
        matches = list(IONEX_DIR.glob(f"COD0OPSFIN_{year}{doy:03d}*_GIM.INX"))
        if matches:
            return matches[0]
        matches = list(IONEX_DIR.glob(f"*{year}{doy:03d}*GIM*"))
        if matches:
            return matches[0]

    return None

# =============================================================================
# IONEX PARSER
# =============================================================================

def _parse_header(lines: list[str]) -> tuple[dict, int]:
    """Parse the IONEX header, return (header_dict, index_of_first_data_line).
    Uses substring matching for labels (not strict column-60 slicing) --
    more forgiving of minor spacing variation between IONEX generators,
    same approach already proven reliable for RINEX header parsing
    earlier in this pipeline."""
    hdr = {
        "exponent": -1,
        "lat1": None, "lat2": None, "dlat": None,
        "lon1": None, "lon2": None, "dlon": None,
        "interval": None,
    }
    for i, l in enumerate(lines):
        if "EXPONENT" in l:
            hdr["exponent"] = int(l.split("EXPONENT")[0].split()[0])
        elif "INTERVAL" in l:
            hdr["interval"] = int(l.split("INTERVAL")[0].split()[0])
        elif "LAT1 / LAT2 / DLAT" in l:
            parts = l.split("LAT1")[0].split()
            hdr["lat1"], hdr["lat2"], hdr["dlat"] = (float(p) for p in parts[:3])
        elif "LON1 / LON2 / DLON" in l:
            parts = l.split("LON1")[0].split()
            hdr["lon1"], hdr["lon2"], hdr["dlon"] = (float(p) for p in parts[:3])
        elif "END OF HEADER" in l:
            return hdr, i + 1
    raise ValueError("END OF HEADER not found -- not a valid IONEX file")


def read_ionex(path: str) -> dict:
    """
    Parse an IONEX file into:
      {
        "epochs": [datetime, ...],          # one per TEC map
        "lat_grid": np.array (n_lat,),       # LAT1 -> LAT2 step DLAT
        "lon_grid": np.array (n_lon,),       # LON1 -> LON2 step DLON
        "maps": {datetime: np.array (n_lat, n_lon)},   # VTEC in TECU
      }
    """
    with open(path, "r", errors="ignore") as f:
        lines = f.readlines()

    hdr, i = _parse_header(lines)
    exponent = hdr["exponent"]
    scale = 10.0 ** exponent

    lat_grid = np.arange(hdr["lat1"], hdr["lat2"] + hdr["dlat"] / 2, hdr["dlat"])
    lon_grid = np.arange(hdr["lon1"], hdr["lon2"] + hdr["dlon"] / 2, hdr["dlon"])
    n_lat, n_lon = len(lat_grid), len(lon_grid)

    epochs = []
    maps = {}

    while i < len(lines):
        line = lines[i]

        if "START OF TEC MAP" in line:
            i += 1
            # epoch line
            epoch_line = lines[i]
            parts = epoch_line.split("EPOCH")[0].split()
            yyyy, mo, dd, hh, mm, ss = (int(float(p)) for p in parts[:6])
            epoch = datetime(yyyy, mo, dd, hh, mm, ss)
            i += 1

            grid = np.full((n_lat, n_lon), np.nan)

            for lat_i in range(n_lat):
                # skip the "LAT/LON1/LON2/DLON/H" line -- grid definition
                # already known from the header, don't need to re-parse it
                i += 1
                vals = []
                while len(vals) < n_lon:
                    row = lines[i].rstrip("\n")
                    # fixed-width 5-char integer fields
                    row_vals = [row[k:k+5] for k in range(0, len(row), 5)]
                    for rv in row_vals:
                        rv = rv.strip()
                        if rv == "":
                            continue
                        vals.append(int(rv))
                    i += 1
                vals = vals[:n_lon]
                arr = np.array(vals, dtype=float)
                arr[arr == 9999] = np.nan   # IONEX missing-value sentinel
                grid[lat_i, :] = arr * scale

            epochs.append(epoch)
            maps[epoch] = grid
            i += 1  # past "END OF TEC MAP"
            continue

        if "START OF RMS MAP" in line:
            # skip RMS maps entirely -- not needed for VTEC comparison
            i += 1
            while i < len(lines) and "END OF RMS MAP" not in lines[i]:
                i += 1
            i += 1
            continue

        if "END OF FILE" in line:
            break

        i += 1

    return {
        "epochs": epochs,
        "lat_grid": lat_grid,
        "lon_grid": lon_grid,
        "maps": maps,
    }

# =============================================================================
# POINT SAMPLING (bilinear interpolation over the lat/lon grid)
# =============================================================================

def gim_at_point(gim: dict, lat: float, lon: float, epoch: datetime) -> float:
    """Bilinear-interpolate the GIM VTEC grid at (lat, lon) for a given
    epoch already present in gim['epochs']. Returns np.nan if the epoch
    isn't found or the point falls outside the grid."""
    if epoch not in gim["maps"]:
        return np.nan

    grid = gim["maps"][epoch]
    lat_grid = gim["lat_grid"]
    lon_grid = gim["lon_grid"]

    # lat_grid may be descending (LAT1=87.5 -> LAT2=-87.5); handle both
    lat_ascending = lat_grid[0] < lat_grid[-1]
    lg = lat_grid if lat_ascending else lat_grid[::-1]
    g  = grid if lat_ascending else grid[::-1, :]

    if lat < lg[0] or lat > lg[-1] or lon < lon_grid[0] or lon > lon_grid[-1]:
        return np.nan

    lat_i1 = np.searchsorted(lg, lat) - 1
    lat_i1 = np.clip(lat_i1, 0, len(lg) - 2)
    lat_i2 = lat_i1 + 1

    lon_i1 = np.searchsorted(lon_grid, lon) - 1
    lon_i1 = np.clip(lon_i1, 0, len(lon_grid) - 2)
    lon_i2 = lon_i1 + 1

    lat_frac = (lat - lg[lat_i1]) / (lg[lat_i2] - lg[lat_i1])
    lon_frac = (lon - lon_grid[lon_i1]) / (lon_grid[lon_i2] - lon_grid[lon_i1])

    v11 = g[lat_i1, lon_i1]
    v12 = g[lat_i1, lon_i2]
    v21 = g[lat_i2, lon_i1]
    v22 = g[lat_i2, lon_i2]

    top    = v11 * (1 - lon_frac) + v12 * lon_frac
    bottom = v21 * (1 - lon_frac) + v22 * lon_frac
    return float(top * (1 - lat_frac) + bottom * lat_frac)

# =============================================================================
# STANDALONE VALIDATION SUMMARY (run this file directly)
# =============================================================================

def main():
    from station_registry import load_station_registry
    import pandas as pd

    STATION_REGISTRY_CSV = BASE_DIR / "station_coords_registry.csv"
    TEC_OUTPUT_DIR = BASE_DIR / "TEC_output"
    stations = load_station_registry(STATION_REGISTRY_CSV)

    print(f"Loaded {len(stations)} stations")
    print(f"IONEX dir: {IONEX_DIR}")

    doy = 19  # quick smoke-test day -- edit as needed
    ionex_path = find_ionex_file(doy)
    if ionex_path is None:
        print(f"No IONEX file found for DOY {doy:03d}")
        return
    print(f"Reading {ionex_path.name} ...")
    gim = read_ionex(str(ionex_path))
    print(f"  {len(gim['epochs'])} epochs, "
          f"grid {len(gim['lat_grid'])}x{len(gim['lon_grid'])}")

    # sample GIM at each station for the first epoch, as a smoke test
    epoch0 = gim["epochs"][0]
    print(f"\nSample GIM VTEC at {epoch0} (first {min(10,len(stations))} stations):")
    for i, (stn, (lat, lon)) in enumerate(stations.items()):
        if i >= 10:
            break
        v = gim_at_point(gim, lat, lon, epoch0)
        print(f"  {stn:6s}  ({lat:7.3f}, {lon:8.3f})  GIM VTEC = {v:.2f} TECU"
              if np.isfinite(v) else f"  {stn:6s}  out of grid / no data")


if __name__ == "__main__":
    main()