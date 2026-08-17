# =============================================================================
# process_tec.py  —  Multi-station, multi-day TEC pipeline
# RINEX 2.11 (17 original stations) AND RINEX 3.04 (~200+ new stations)
#
# Rebuilt from the original 17-station-only process_tec.py to:
#   - Auto-discover whatever stations exist per DOY (no hardcoded registry)
#   - Support RINEX 2.11 and 3.04 obs/nav transparently (rinex_reader.py
#     detects the version and dispatches automatically)
#   - Derive station lat/lon from the RINEX header when not otherwise known
#     (tec_calculator.py does this internally now; this script also parses
#     it separately for the plotting functions, which need lat/lon explicitly)
#
# Usage:
#   conda activate base
#   cd /Users/ziraa/Documents/GISTDA/GISTDA_TEC
#   python process_tec.py
# =============================================================================
import traceback
from pathlib import Path
from datetime import datetime, timedelta

from rinex_reader   import read_rinex211, read_nav_only, read_obs_only
from tec_calculator import calculate_tec          # 5-arg: obs,nav,doy,satb,save_path
from plot_tec       import plot_daily_tec
from utils          import xyz2lla, download_dcb
from station_registry import load_station_registry

# =============================================================================
# PATHS  — EDIT THESE to match your actual layout
# =============================================================================
BASE_DIR = Path("/Users/ziraa/Documents/GISTDA/GISTDA_TEC")
OBS_ROOT = BASE_DIR / "RINEX_by_DOY"            # DOY_019/<STATION>/<station>0190.26o
NAV_DIR  = BASE_DIR / "RINEX3_Nav_File"         # RINEX3 nav files (BRDC00IGS_R_...)
DCB_DIR  = BASE_DIR / "TEC_Calculation"         # EDIT: where DCB files are cached
OUT_DIR  = BASE_DIR / "TEC_output"

# Persistent station coordinate registry (built from a DRY_RUN pass) --
# avoids re-parsing every RINEX header on every run, and keeps a station's
# coordinates stable even on days where its obs file happens to be missing.
STATION_REGISTRY_CSV = BASE_DIR / "station_coords_registry.csv"

# =============================================================================
# RUN CONFIG
# =============================================================================
DOY_START = 20     # change to process a single day (DOY_START == DOY_END)
DOY_END   = DOY_START  # change to process a range of days (e.g. 19-21)
YEAR      = 2026
YEAR_2D   = "26"

DRY_RUN = False      # True: only discover stations + parse lat/lon per day,
                     # no TEC calculation. Useful before a big run to sanity-
                     # check station/nav coverage.

# Only process stations whose name is alphabetically >= this (e.g. "P" runs
# P, Q, R, ... Z, skipping A-O). Set to None or "" to process all stations.
STATION_FROM = "PONG"
STATION_TO = "RAYG"


# =============================================================================
# STATION DISCOVERY  (replaces the old static 17-station STATIONS dict)
# =============================================================================

def discover_stations(doy: int) -> list[str]:
    """List every station folder present for this DOY under RINEX_by_DOY."""
    doy_dir = OBS_ROOT / f"DOY_{doy:03d}"
    if not doy_dir.exists():
        return []
    return sorted(p.name for p in doy_dir.iterdir() if p.is_dir())


def find_obs_file(station: str, doy: int) -> Path | None:
    """Return path to the .o file for a station/DOY under RINEX_by_DOY."""
    folder = OBS_ROOT / f"DOY_{doy:03d}" / station
    if not folder.exists():
        return None
    p = folder / f"{station.lower()}{doy:03d}0.{YEAR_2D}o"
    return p if p.exists() else None


def find_nav_file(doy: int) -> Path | None:
    """
    Return path to the broadcast nav file for a DOY. Tries both the legacy
    RINEX 2.11 GPS-only naming (brdcDDD0.YYn) and the RINEX 3.x merged
    multi-GNSS naming (BRDC00IGS_R_YYYYDDD0000_01D_MN.rnx), since the new
    ~200-station network may be paired with either depending on how you
    downloaded it.
    """
    candidates = [
        NAV_DIR / f"brdc{doy:03d}0.{YEAR_2D}n",
        NAV_DIR / f"BRDC00IGS_R_{YEAR}{doy:03d}0000_01D_MN.rnx",
        NAV_DIR / f"BRDM00DLR_S_{YEAR}{doy:03d}0000_01D_MN.rnx",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


STATION_REGISTRY = load_station_registry(STATION_REGISTRY_CSV)


def get_station_latlon(obs_file: Path, station: str = None):
    """Return (lat, lon) for a station. Checks the persistent registry
    first (fast, stable across days); falls back to parsing the RINEX
    header's APPROX POSITION XYZ (works for RINEX 2.11 or 3.04) for any
    station not yet in the registry."""
    if station and station.upper() in STATION_REGISTRY:
        return STATION_REGISTRY[station.upper()]

    try:
        with open(obs_file, "r", errors="ignore") as f:
            for line in f:
                if "APPROX POSITION XYZ" in line:
                    parts = line.split()
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    lat, lon, _alt = xyz2lla(x, y, z)
                    return lat, lon
                if "END OF HEADER" in line:
                    break
    except Exception as e:
        print(f"  [warn] could not parse position from {obs_file.name}: {e}")
    return None


# =============================================================================
# PROCESS ONE STATION x ONE DAY
# =============================================================================

def process_one(station: str, doy: int, nav_struct: dict) -> dict | None:
    obs_file = find_obs_file(station, doy)

    if obs_file is None:
        print(f"  [skip] {station} DOY{doy:03d} — obs file not found")
        return None

    latlon = get_station_latlon(obs_file, station)
    if latlon is None:
        print(f"  [skip] {station} DOY{doy:03d} — no APPROX POSITION XYZ in header")
        return None
    lat, lon = latlon

    if DRY_RUN:
        print(f"  [dry-run] {station} DOY{doy:03d}  lat={lat:.4f} lon={lon:.4f}  "
              f"obs={obs_file.name}")
        return {
            "station": station,
            "lat":     lat,
            "lon":     lon,
            "doy":     doy,
            "obs_file": obs_file.name,
        }

    print(f"  -> {station} DOY{doy:03d}  lat={lat:.4f} lon={lon:.4f}  "
          f"obs={obs_file.name}")
    try:
        # 1. Read only the obs file -- nav_struct was already parsed once
        # for this whole DOY by the caller (see main()), avoiding a
        # ~200x redundant re-parse of the large shared nav file per station.
        obs = read_obs_only(str(obs_file))
        if obs is None:
            print(f"  [fail] {station} DOY{doy:03d} — obs read failed")
            return None

        date = datetime(obs["date"][0], obs["date"][1], obs["date"][2])
        doy_r  = date.timetuple().tm_yday
        year, month, day = date.year, date.month, date.day

        # 2. Download / read DCB corrections
        satb = download_dcb(obs, str(DCB_DIR))
        if satb is None:
            print(f"  [fail] {station} DOY{doy:03d} — DCB download/read failed")
            return None

        # 3. Calculate TEC
        out_dir = OUT_DIR / station / f"DOY{doy:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        result = calculate_tec(obs, nav_struct, doy_r, satb, str(out_dir))
        if result is None:
            return None

        # 4. Per-station daily plot
        plot_daily_tec(result, station, lat, lon,
                       doy, year, month, day, str(out_dir))

        return {
            "station": station,
            "lat":     lat,
            "lon":     lon,
            "doy":     doy,
            "year":    year,
            "month":   month,
            "day":     day,
            **result,
        }

    except Exception:
        print(f"  [error] {station} DOY{doy:03d}:\n{traceback.format_exc()}")
        return None


# =============================================================================
# MAIN LOOP
# =============================================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = datetime.now()
    print(f"=== TEC pipeline  DOY {DOY_START}-{DOY_END}  ({YEAR}) "
          f"{'[DRY RUN] ' if DRY_RUN else ''}===\n")

    for doy in range(DOY_START, DOY_END + 1):
        date = datetime(YEAR, 1, 1) + timedelta(days=doy - 1)
        stations = discover_stations(doy)
        if STATION_FROM:
            stations = [s for s in stations if s.upper() >= STATION_FROM.upper()]
        print(f"\n--- DOY {doy:03d}  {date.strftime('%Y-%m-%d')}  "
              f"({len(stations)} station(s) discovered) ---")

        day_results = []

        nav_struct = None
        if not DRY_RUN:
            nav_file = find_nav_file(doy)
            if nav_file is None:
                print(f"  [skip day] DOY{doy:03d} — nav file not found")
                continue
            print(f"  Parsing nav file once for this DOY: {nav_file.name}")
            nav_struct = read_nav_only(str(nav_file))
            if nav_struct is None or not nav_struct.get("eph"):
                print(f"  [skip day] DOY{doy:03d} — nav parse failed or empty")
                continue
            print(f"  -> {len(nav_struct['eph'])} ephemerides, "
                  f"{len(set(nav_struct['index']))} unique PRNs")

        for i, station in enumerate(stations, start=1):
            r = process_one(station, doy, nav_struct)
            if r:
                day_results.append(r)
            if i % 25 == 0:
                print(f"  ... {i}/{len(stations)} attempted, "
                      f"{len(day_results)} succeeded so far")

        if DRY_RUN:
            print(f"  DOY{doy:03d} dry-run summary: "
                  f"{len(day_results)}/{len(stations)} stations had obs+nav+position")
            continue

        # 2-D regional map generation is now a separate step -- run
        # plot_tec_map_multistation.py after this script finishes; it reads
        # the saved VTEC CSVs and builds the map with a proper Thailand
        # GeoJSON boundary. See plot_tec_map_multistation.py.

        print(f"  DOY{doy:03d} done — {len(day_results)}/{len(stations)} stations")

    print(f"\n=== Finished in {datetime.now() - t0} ===")
    print(f"Outputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()