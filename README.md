# GISTDA_TEC
# Appendix: Processing Pipeline Scripts

This appendix documents the scripts used to process raw RINEX observation data
through to validated regional VTEC maps, for the 17-station Thai CORS network
analysed in this study. Scripts are grouped and stored in RINEX_PROCESSING; each entry
gives its purpose, key inputs/outputs, and notable methods or parameters
rather than full source code, for brevity.

---

## 1. Core RINEX Processing

| Script | Purpose | Key inputs | Key outputs |
|---|---|---|---|
| `rinex_reader.py` | Parses RINEX 2.11 observation and navigation files: header/epoch parsing, obs/nav data extraction. | `.o` (obs) and `.n` (nav) RINEX files | Parsed obs/nav data structures |
| `utils.py` | Shared low-level utilities: GPS constants, satellite position (`satpos_xyz_sbias`), differential code bias (DCB) download and application, cycle-slip correction. | Nav data, DCB source (AIUB/CODE) | Corrected pseudorange/phase data, satellite positions |
| `tec_calculator.py` | Core TEC computation: geometry-free STEC derivation, DCB correction, thin-shell STEC→VTEC conversion, IPP position, ROTI. | Parsed obs/nav, DCB values | Per-epoch STEC, VTEC, ROTI, IPP lat/lon |
| `process_tec.py` | Multi-station, multi-day driver: discovers available stations/days, calls `tec_calculator.py` per station-day, saves output. | RINEX file tree (`RINEX_by_DOY/`), station list | `TEC_output/{STATION}/DOY{DOY}/{STATION}_{DOY}_VTEC.csv` (VTEC, STEC, ROTI, IPP) |

**Key parameters** (see Methodology Table for full list): ionospheric shell height *h* = 350 km, elevation mask ≥15°, Earth radius *R*<sub>E</sub> = 6371 km.

---

## 2. Per-Station Visualisation

| Script | Purpose | Key inputs | Key outputs |
|---|---|---|---|
| `plot_tec.py` | Generates the daily VTEC/ROTI time-series plot for a single station (e.g. the Chon Buri case study figures). | One station's saved VTEC CSV | Per-station daily VTEC/ROTI PNG |

---

## 3. Regional VTEC Mapping (17-station network)

| Script | Method | Key inputs | Key outputs |
|---|---|---|---|
| `plot_tec_map_multistation_17stations.py` | Thin-Plate Spline (RBF) regional interpolation | All 17 stations' VTEC CSVs, Thailand boundary GeoJSON | Per-day, 12-panel (2-hour bin) regional VTEC map PNG |
| `plot_tec_map_idw_17stations.py` | Inverse Distance Weighting (power = 2) | Same as above | Same as above |
| `plot_tec_map_kriging_17stations.py` | Ordinary Kriging, spherical variogram (auto-fit per panel, from scratch — no `pykrige` dependency) | Same as above | Same as above |

**Shared parameters**: 2-hour temporal bins, ±60 min aggregation window, 100×100 spatial grid (~0.175°×0.115° cell size), extent 4.5–22.0°N / 96.0–107.5°E.

**Note**: `plot_tec_map_kriging_17stations.py` includes a station-level QC check (`is_station_valid`) excluding any station whose nighttime VTEC exceeds a physical threshold (15 TECU) — this is where the CHAN nighttime-bias case was first caught.

---

## 4. Cross-Validation

| Script | Purpose | Key inputs | Key outputs |
|---|---|---|---|
| `cross_validation.py` | Leave-One-Out Cross-Validation (LOOCV) for IDW, TPS/RBF, and Ordinary Kriging: holds out each station in turn, predicts from the remaining 16, compares against observed. | All 17 stations' VTEC CSVs | `cross_validation_errors.csv` (per-prediction errors), `cross_validation_stats.csv` (ME/MAE/RMSE/CI95% per method) |
| `plot_loocv_scatter.py` | Visualises LOOCV as predicted-vs-observed scatter (3 panels) + residual boxplot. | `cross_validation_errors.csv` | `LOOCV_scatter.png` |
| `plot_loocv_table.py` | Renders LOOCV summary statistics (including correlation *r*) as a formatted table. | `cross_validation_errors.csv` | `loocv_summary_table.png`, `loocv_summary_stats.csv` |

**Note**: an SCHA (Spherical Cap Harmonic Analysis) + IRI-background method was implemented and tested during development but excluded from the final cross-validation comparison (see Future Work).

---

## 5. Validation Against Reference Products

| Script | Purpose | Key inputs | Key outputs |
|---|---|---|---|
| `validate_test.py` | IONEX file reader and bilinear point-sampling for CODE GIM comparison. | GFZ/CDDIS IONEX files (`COD0OPSFIN_..._GIM.INX`) | Parsed GIM grid, per-point VTEC lookup |
| `plot_regional_vs_gim_timeseries.py` | Compares daily regional network-median VTEC against CODE GIM VTEC at the network centroid. | 17 stations' VTEC CSVs, IONEX files | `regional_vs_gim_timeseries.png`, `.csv` |

*(Note: confirm this is your 17-station-specific version — an earlier registry-based ~213-station variant of this script also exists from this project; make sure the one referenced here matches your report's scope.)*

---

## 6. Space Weather Context Data

| Script | Purpose | Key inputs | Key outputs |
|---|---|---|---|
| `parse_gfz_kp_index.py` | Parses the raw GFZ Potsdam Kp/Ap/F10.7 data file directly (handles both plain-text and RTF-exported source formats). | GFZ definitive Kp index file | Filtered Kp/Ap/F10.7 CSV for the study period |
| `plot_kp_f107_ap_summary.py` | Renders the Kp/F10.7/Ap daily summary table with G-scale classification. | Parsed Kp/Ap/F10.7 CSV | `kp_f107_ap_summary.png` |
| `plot_kp_3hourly_table.py` | Full 3-hourly (not daily-max) Kp index table. | Verified 3-hourly Kp values | `kp_3hourly_table.png`, `.csv` |
| `plot_kp_barchart.py` | 3-hourly Kp bar chart (3-band colour convention: green/orange/dark-red), with storm-window highlight. | Same as above | `kp_barchart_3hourly.png` |
| `generate_storm_summary_table.py` | Combines storm chronology (Kp/G-scale) with peak VTEC per day and % enhancement vs. quiet baseline. | Kp CSV, 17 stations' VTEC CSVs | `storm_chronology_summary.png`, `.csv` |

---

## 7. Pipeline Data Flow (summary)

```
RINEX obs (.o) + nav (.n)
        │
        ▼
  rinex_reader.py  ──►  utils.py (DCB, satpos, cycle-slip)
        │
        ▼
  tec_calculator.py  (STEC → VTEC, ROTI, IPP)
        │
        ▼
  process_tec.py  (multi-station/day driver)
        │
        ▼
  TEC_output/{STATION}/DOY{DOY}/*.csv
        │
        ├──► plot_tec.py                          (per-station time series)
        ├──► plot_tec_map_*_17stations.py          (regional maps, 3 methods)
        ├──► cross_validation.py → plot_loocv_*.py (LOOCV validation)
        └──► plot_regional_vs_gim_timeseries.py    (external validation)
```

---

*Full source code for all scripts listed above is available at [repository link, if applicable] / on request.*
