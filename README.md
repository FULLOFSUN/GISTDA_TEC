# GISTDA_TEC 17 STATIONS PROJECT
Processing Pipeline Scripts for 17 stations with RINEX2 

This README.md covers the scripts used to process raw RINEX observation data
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

**Key parameters** 
ionospheric shell height *h* = 350 km, elevation mask ≥15°, Earth radius *R*<sub>E</sub> = 6371 km.

---

## 2. Per-Station Visualisation

| Script | Purpose | Key inputs | Key outputs |
|---|---|---|---|
| `plot_tec.py` | Generates the daily VTEC/ROTI time-series plot for a single station (e.g. the Chon Buri case study figures). | One station's saved VTEC CSV | Per-station daily VTEC/ROTI PNG |

---

## 3. Regional VTEC Mapping (17-station network)

| Script | Method | Key inputs | Key outputs |
|---|---|---|---|
| `plot_tec.py` | Basic reference of regional interpolation | All 17 stations' VTEC CSVs, Thailand boundary GeoJSON | Per-day, 12-panel (2-hour bin) regional VTEC map PNG |
| `plot_tectpsnew.py` | Thin-Plate Spline (RBF) regional interpolation | All 17 stations' VTEC CSVs, Thailand boundary GeoJSON | Per-day, 12-panel (2-hour bin) regional VTEC map PNG |
| `plot_tecidwnew.py` | Inverse Distance Weighting (power = 2) | Same as above | Same as above |
| `plot_teckrigingnew.py` | Ordinary Kriging, spherical variogram (auto-fit per panel, from scratch — no `pykrige` dependency) | Same as above | Same as above |
| `plot_ipp.py` | ROTI IPP plot | Same as above | Same as above |
| `gif_result.py` | Plot all of the results per methods as gif | Same as above | 14-30 combined in GIF |

**Shared parameters**: 2-hour temporal bins, ±60 min aggregation window, 100×100 spatial grid (~0.175°×0.115° cell size), extent 4.5–22.0°N / 96.0–107.5°E.

**Note**: `plot_teckrigingnew.py` includes a station-level QC check (`is_station_valid`) excluding any station whose nighttime VTEC exceeds a physical threshold (15 TECU) — this is where the CHAN nighttime-bias case was first caught.

`vtec_hours_utils` and `diagnose_stations_outliers` includes the process to diagnose or checking the quality of each results on each stations  

---

## 4. Cross-Validation

| Script | Purpose | Key inputs | Key outputs |
|---|---|---|---|
| `plot_loocv_scatter.py` | Leave-One-Out Cross-Validation (LOOCV) for IDW, TPS/RBF, and Ordinary Kriging: holds out each station in turn, predicts from the remaining 16, compares against observed. | All 17 stations' VTEC CSVs | `cross_validation_errors.csv` (per-prediction errors), `cross_validation_stats.csv` (ME/MAE/RMSE/CI95% per method) |
| `plot_loocv_scatter.py` | Visualises LOOCV as predicted-vs-observed scatter (3 panels) + residual boxplot. | `cross_validation_errors.csv` | `cross_validation_table.png` |

---

## 5. Validation Against Reference Products

| Script | Purpose | Key inputs | Key outputs |
|---|---|---|---|
| `validate_test.py` | IONEX file reader and bilinear point-sampling for CODE GIM comparison. | IONEX files (`COD0OPSFIN_..._GIM.INX`) | Parsed GIM grid, per-point VTEC lookup |
| `validate_against_gim.py` or `gimtecval.py` | Compares daily regional network-median VTEC against CODE GIM VTEC at the network centroid. | 17 stations' VTEC CSVs, IONEX files | `regional_vs_gim_timeseries.png`, `.csv` |


---

## 6. Space Weather Context Data

| Script | Purpose | Key inputs | Key outputs |
|---|---|---|---|
| `sw_kp_tec_jan.py` | Parses the raw GFZ Potsdam Kp/Ap/F10.7 data file directly (handles both plain-text and RTF-exported source formats). | GFZ definitive Kp index file | Filtered Kp/Ap/F10.7 CSV for the study period |
| `kp_f107_Ap_index.csv` | Kp Index from 2026 after parsing from the full GFZ rtf format

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
        ├──► plot_tec(method)new.py               (regional maps, 3 methods)
        ├──► validate_test.py → gimtecval.py      (LOOCV validation)
        └──► validate_against_gim.py              (external validation)
```

---

*Full source code for all scripts listed above is available at* **RINEX_PROCESSING**
