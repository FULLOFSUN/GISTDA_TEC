# Thai CORS Regional TEC Mapping Pipeline — ~213-Station Network

RINEX 3.04 processing pipeline for the expanded Thai CORS network
(~213 stations), covering DOY 014–030 (14–30 January 2026), including the
19 January 2026 G4 geomagnetic storm. This is the **expanded-network**
counterpart to the 17-station pipeline.

---

## 1. Core RINEX → TEC Processing

| File | Purpose |
|---|---|
| `rinex_reader.py` | RINEX 2.11 + 3.04 obs/nav parser, version-dispatched. |
| `utils.py` | GPS constants, satellite position, DCB download (HTTPS/CODE), cycle-slip correction. |
| `tec_calculator.py` | Core STEC→VTEC conversion, ROTI, IPP computation per station. |
| `station_registry.py` | Shared loader for `station_coords_registry.csv` — single source of truth for station lat/lon across all downstream scripts. |
| `station_coords_registry.csv` | Registry of ~213 station coordinates, built from a dry-run pass over discovered stations. |
| `process_tec.py` | Multi-station, multi-day driver. Auto-discovers stations per DOY, caches nav data once per day, writes VTEC/STEC/ROTI CSVs. |
| `process_tec_fromp.py` | Resume-from-station variant of `process_tec.py` (using the `STATION_FROM`/`STATION_TO` alphabetical filter), for restarting a partial run without reprocessing already-completed stations. |
| `RINEX3_NAV_FILE/` | Broadcast navigation files (`BRDC00IGS_R_...`), one per DOY, shared across all stations for that day. |
| `TEC_Calculation/` | DCB cache directory (downloaded CODE bias files), auto-created on first run. |

---

## 2. Quality Control / Diagnostics

| File | Purpose |
|---|---|
| `diagnose_station_outliers.py` | Flags stations whose VTEC deviates from their **nearest spatial neighbours** (not the whole-network median) at a given hour — catches genuine station-level faults while preserving real regional gradients (e.g. the EIA trough) that a naive network-wide comparison would misflag. |
| `diagnose_vtec_gap.py` | Checks saved VTEC output for sparse/missing PRN coverage per hour, to distinguish genuine data gaps from processing bugs. |
| `vtec_hours_utils.py` | Shared `collect_hour_points()` with local-neighbour outlier rejection, used by the mapping scripts below. **Note**: this file is named `vtec_hours_utils.py` (plural "hours") — the version built and tested in this project is `vtec_hour_utils.py` (singular). If any script imports from the singular name and can't find it, that mismatch is the likely cause — confirm this file's exact import name matches what `plot_tec218*.py` actually imports. |

---

## 3. Regional VTEC Mapping (~213-station network)

| File | Method |
|---|---|
| `plot_tec218tps.py` | Thin-Plate Spline (RBF) regional interpolation, ~213-station registry version. |
| `plot_tec218idw.py` | Inverse Distance Weighting, ~213-station registry version. |
| `plot_tec218krig.py` | Ordinary Kriging (pykrige-based), ~213-station registry version. |
| `plot_tec.py` | Per-station daily VTEC/ROTI time-series plot (station-count-agnostic). |
| `TEC_maps_comparison_new/` | Output directory for generated regional map PNGs. |

---

## 4. Cross-Validation

| File | Purpose |
|---|---|
| `loocv200stations.py` | Leave-One-Out Cross-Validation across the ~213-station network — the expanded-network counterpart to the 17-station `cross_validation.py`. |
| `cross_validation_errors.csv` | Per-prediction LOOCV errors (all methods). |
| `cross_validation_stats.csv` | Summary statistics (ME/MAE/RMSE/CI95%) per method. |
| `cross_validation_table.png` | Rendered summary table. |

> **Check before citing these results**: this project has already hit one
> case where a LOOCV re-run under a different script/data configuration
> produced materially different numbers than expected (N=3465 vs. N=3463,
> with the "best method" conclusion reversing between runs). Before using
> `cross_validation_stats.csv` from this ~213-station run in any report,
> confirm it was generated from the *current, corrected* pipeline version,
> not a stale run.

---

## 5. External Validation (GIM / Space Weather)

| File | Purpose |
|---|---|
| `validate_test.py` | IONEX (GIM) file reader and bilinear point-sampling, built and synthetic-test-verified against the real `COD0OPSFIN_..._GIM.INX` format. |
| `gimtecval.py` | Combines `validate_test.py`'s IONEX reading with the regional pipeline's VTEC to produce the regional-vs-GIM comparison — the ~213-station counterpart to `plot_regional_vs_gim_timeseries.py`. |
| `regional_vs_..._timeseries.csv` / `.png` | Output of the GIM comparison: regional network-median VTEC vs. CODE GIM at the network centroid. |
| `kp_index_historical_data.rtf` | Raw GFZ Potsdam Kp/Ap/F10.7 source file (definitive index), parsed elsewhere in this project via `parse_gfz_kp_index.py`. |
| `rtsw_plot_data_..._22_57_11.txt` | NOAA SWPC Real-Time Solar Wind (RTSW) data — I have no record of building a script that consumes this file in this project; confirm its origin/purpose before including it in any pipeline documentation. |

---

*There might be one or two missing pieces from the scripts and outputs, if anything, please contact me for any information or inquiries*
