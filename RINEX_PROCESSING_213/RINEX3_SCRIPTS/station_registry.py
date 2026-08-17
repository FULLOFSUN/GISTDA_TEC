# =============================================================================
# station_registry.py -- shared station coordinate registry loader
#
# Used by both process_tec.py (to get station lat/lon during processing)
# and plot_tec_map_multistation.py (to get station lat/lon for mapping),
# so there's one single source of truth instead of two copy-pasted
# versions that could silently drift out of sync.
# =============================================================================
from pathlib import Path


def load_station_registry(csv_path: Path) -> dict[str, tuple[float, float]]:
    """Load station -> (lat, lon) from a CSV built from a prior DRY_RUN pass
    (columns: station,lat,lon). Returns {} if the file doesn't exist yet."""
    registry = {}
    if not csv_path.exists():
        print(f"  [info] no station registry found at {csv_path} -- "
              f"will parse every RINEX header instead (slower)")
        return registry
    with open(csv_path, "r") as f:
        next(f)  # header row
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 3:
                continue
            station, lat, lon = parts
            try:
                registry[station.upper()] = (float(lat), float(lon))
            except ValueError:
                continue
    print(f"  [info] loaded {len(registry)} station coordinates from {csv_path.name}")
    return registry