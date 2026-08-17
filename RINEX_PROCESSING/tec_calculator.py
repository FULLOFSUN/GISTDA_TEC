"""
tec_calculator.py — TEC calculation, generalized from 17-station RINEX 2.11
Thai CORS network to ~200+ station RINEX 2.11 / 3.04 network.

Signature: calculate_tec(obs, nav, doy, satb, save_path)
  obs       : dict from rinex_reader.read_rinex211()
  nav       : dict from rinex_reader.read_rinex211()
  doy       : int, day of year
  satb      : dict from utils.download_dcb()  {'P1C1':..., 'P1P2':...}
  save_path : str, output folder

Returns: {"TEC": ..., "DCB": ..., "ROTI": ..., "prm": ...}  or None on failure

Fixes applied in this version:
  [Fix 1] ROTI computed from cycle-slip-corrected STECl_M_new (not raw STECl)
  [Fix 2] *** Critical *** IPP lat/lon calculation used `coords[0]`/`coords[1]`
          directly, which is None for any station not in the hardcoded
          17-station STATION_COORDS dict (i.e. every one of the new ~200
          stations). This crashed identically on every PRN with
          "'NoneType' object is not subscriptable". Now derives receiver
          lat/lon from `refpos` (always defined, either from the coord
          override or the RINEX header ECEF) via xyz2lla, computed once
          per station instead of recomputed per-PRN inside the loop.
  [Misc]  Parquet column names converted to str (prevents int-column error)
"""

import numpy as np
import os
import math
import pandas as pd
from utils import (
    gpscons, xyz2lla, satpos_xyz_sbias, calelevation,
    windowshiftTEC, outlinecorr, rcv_bias_ma,
    roticalculation, CycleSlipCorrection_v2,
    download_dcb
)

# STATION LAT/LON — optional precise-coordinate OVERRIDE for the original
# 17 known Thai CORS stations. Any station not listed here (i.e. all of the
# new ~200 stations) automatically falls back to the RINEX header's
# APPROX POSITION XYZ, converted via xyz2lla. This dict is no longer load-
# bearing for correctness -- it only gives slightly more precise coordinates
# for stations where we happen to have them.
STATION_COORDS = {
    "CHAN": (12.61031,   102.102411),
    "CHMA": (18.835275,  98.969956),
    "CNBR": (13.406019, 100.997652),
    "CPN1": (10.72466,   99.374356),
    "CPN":  (10.72466,   99.374356),
    "DPT9": (13.756782, 100.573200),
    "DPT":  (13.756782, 100.573200),
    "KMI6": (13.727832, 100.772429),
    "KMI":  (13.727832, 100.772429),
    "LPBR": (14.800907, 100.651246),
    "NKNY": (14.212003, 101.202211),
    "NKRM": (14.992119, 102.129470),
    "NKSW": (15.690637, 100.114112),
    "PJRK": (11.811621,  99.796348),
    "SISK": (15.116122, 104.285676),
    "SOKA": ( 7.206694, 100.596121),
    "SPBR": (14.518875, 100.130580),
    "SRTN": ( 9.132225,  99.331361),
    "UDON": (17.412732, 102.780704),
    "UTTD": (17.630094, 100.096343),
}


def _lla_to_ecef(lat_deg, lon_deg, alt_m=10.0):
    """Geodetic lat/lon/alt -> ECEF XYZ (WGS-84)."""
    a  = 6378137.0
    e2 = 6.694379990141317e-3
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    N = a / math.sqrt(1 - e2 * math.sin(lat) ** 2)
    return [
        (N + alt_m) * math.cos(lat) * math.cos(lon),
        (N + alt_m) * math.cos(lat) * math.sin(lon),
        (N * (1 - e2) + alt_m) * math.sin(lat),
    ]


def _nan_matrix(n_time, n_sat):
    """Return a (n_time, n_sat) array filled with NaN."""
    return np.full((n_time, n_sat), np.nan, dtype=float)


def _save(base_path, data):
    """Save as CSV (always) and parquet (if pyarrow available)."""
    df = pd.DataFrame(data)
    df.to_csv(base_path + ".csv", index=False)
    try:
        df.columns = df.columns.astype(str)
        df.to_parquet(base_path + ".parquet", index=False)
    except Exception:
        pass


def calculate_tec(obs, nav, doy, satb, save_path):
    """
    Main TEC calculation. Works for both the original 17-station RINEX 2.11
    network and the expanded ~200+ station RINEX 2.11/3.04 network, since
    rinex_reader.py now version-dispatches obs/nav parsing and this function
    no longer depends on a hardcoded per-station coordinate lookup.
    """
    try:
        # === 1. Constants ====================================================
        const        = gpscons()
        c, f1, f2, A = const['c'], const['f1'], const['f2'], const['A']
        Re, h        = const['Re'], const['h']
        lam1, lam2   = c / f1, c / f2
        k            = (f1**2 * f2**2) / (40.3 * (f1**2 - f2**2))

        # === 2. Receiver ECEF + lat/lon (station-coord override OR RINEX header)
        station = obs.get('station', 'UNKN')
        coords  = (STATION_COORDS.get(station)
                   or STATION_COORDS.get(station[:4])
                   or STATION_COORDS.get(station.upper())
                   or STATION_COORDS.get(station.upper()[:4]))
        if coords:
            refpos = _lla_to_ecef(coords[0], coords[1])
        else:
            refpos = obs.get('rcvpos', [0, 0, 0])
            print("  [warn] " + station + " not in coord registry -- using RINEX ECEF")

        # [Fix 2] Always derive receiver lat/lon from refpos (works whether
        # refpos came from the override dict or the RINEX header), computed
        # ONCE per station rather than per-PRN. This removes the dependency
        # on `coords`, which is None for every non-registry station.
        rcv_lat_deg, rcv_lon_deg, _rcv_alt = xyz2lla(refpos[0], refpos[1], refpos[2])
        rcv_lat_r = np.radians(rcv_lat_deg)
        rcv_lon_r = np.radians(rcv_lon_deg)

        # === 3. Preallocate ==================================================
        n_time = 86400
        n_sat  = 32

        TEC = {
            'vertical':    _nan_matrix(n_time, n_sat),
            'slant':       _nan_matrix(n_time, n_sat),
            'withrcvbias': _nan_matrix(n_time, n_sat),
            'withbias':    _nan_matrix(n_time, n_sat),
        }
        prm = {
            'elevation': _nan_matrix(n_time, n_sat),
            'azimuth':   _nan_matrix(n_time, n_sat),
            'IPP_lat':   _nan_matrix(n_time, n_sat),
            'IPP_long':  _nan_matrix(n_time, n_sat),
            'Times':     _nan_matrix(n_time, n_sat),
        }
        DCB   = {'sat': np.zeros(n_sat), 'rcv': 0.0}
        STECp = _nan_matrix(n_time, n_sat)
        STECl = _nan_matrix(n_time, n_sat)

        obs_data = np.array(
            [[v if v is not None else np.nan for v in row]
             for row in obs['data']], dtype=float
        )
        obs_types = obs['type']

        # Build observable index
        type_idx = {}
        for t in ['C1', 'P1', 'P2', 'L1', 'L2']:
            for j, ot in enumerate(obs_types):
                if ot == t:
                    type_idx[t] = j
                    break
        if 'C1' not in type_idx and 'P1' in type_idx:
            type_idx['C1'] = type_idx['P1']

        if 'C1' not in type_idx or 'P2' not in type_idx:
            print("[calculate_tec] Missing C1/P2 in obs types: " + str(obs_types))
            return None

        # === 4. Per-PRN loop =================================================
        eph_arr   = np.array(nav['eph'], dtype=float)
        nav_index = list(nav['index'])
        if eph_arr.ndim != 2:
            print("[calculate_tec] nav eph is not 2D -- check nav file")
            return None

        obs_epoch = np.array(obs['epoch'])
        obs_index = np.array(obs['index'])
        time_offset = (obs['date'][3] * 3600
                       + obs['date'][4] * 60
                       + obs['date'][5])

        for PRN in range(1, n_sat + 1):
            sat_idx = np.where(obs_index == PRN)[0]
            if not sat_idx.size:
                continue

            Time_raw = obs_epoch[sat_idx] + time_offset
            valid_t  = (Time_raw >= 0) & (Time_raw < n_time)
            if valid_t.sum() == 0:
                continue

            Time     = Time_raw[valid_t].astype(int)
            sat_idx2 = sat_idx[valid_t]

            try:
                C1 = obs_data[sat_idx2, type_idx['C1']]
                P2 = obs_data[sat_idx2, type_idx['P2']]

                if 'L1' in type_idx:
                    L1 = lam1 * obs_data[sat_idx2, type_idx['L1']]
                else:
                    L1 = np.full(len(sat_idx2), np.nan)

                if 'L2' in type_idx:
                    L2 = lam2 * obs_data[sat_idx2, type_idx['L2']]
                else:
                    L2 = np.full(len(sat_idx2), np.nan)

                satpos, _ = satpos_xyz_sbias(
                    Time, PRN, eph_arr.tolist(), nav_index, C1)
                elev, azim = calelevation(satpos, refpos)

                # ── IPP lat/lon using thin-shell at h=350km ──────────────────
                # rcv_lat_r / rcv_lon_r computed once above (station-level),
                # NOT from `coords` (which is None for non-registry stations)
                azim_r = np.radians(azim)
                elev_r = np.radians(elev)

                # Earth central angle psi between receiver and IPP
                psi = np.pi/2 - elev_r - np.arcsin(Re * np.cos(elev_r) / (Re + h))

                # IPP geodetic latitude
                ipp_lat_r = np.arcsin(np.sin(rcv_lat_r) * np.cos(psi)
                       + np.cos(rcv_lat_r) * np.sin(psi) * np.cos(azim_r))

                # IPP geodetic longitude
                ipp_lon_r = rcv_lon_r + np.arcsin(
                     np.sin(psi) * np.sin(azim_r) / np.cos(ipp_lat_r))

                prm['IPP_lat'][Time,  PRN - 1] = np.degrees(ipp_lat_r)
                prm['IPP_long'][Time, PRN - 1] = np.degrees(ipp_lon_r)
                # ── END IPP ───────────────────────────────────────────────────

                prm['elevation'][Time, PRN - 1] = elev
                prm['azimuth'][Time, PRN - 1]   = azim
                prm['Times'][Time, PRN - 1]      = Time + 1

                valid_p = np.isfinite(C1) & np.isfinite(P2) & (C1 != 0) & (P2 != 0)
                if valid_p.sum() > 0:
                    STECp[Time[valid_p], PRN - 1] = k * (P2[valid_p] - C1[valid_p])

                valid_l = np.isfinite(L1) & np.isfinite(L2) & (L1 != 0) & (L2 != 0)
                if valid_l.sum() > 0:
                    STECl[Time[valid_l], PRN - 1] = k * (L1[valid_l] - L2[valid_l])

            except Exception as err:
                print('  PRN#' + str(PRN) + ' error: ' + str(err))
                continue

        # === 5. Normalise & elevation mask ===================================
        STECp /= 1e16
        STECl /= 1e16

        mask = np.copy(prm['elevation'])
        mask[mask < 15]       = np.nan
        mask[~np.isnan(mask)] = 1

        STECp            = mask * STECp
        STECl            = mask * STECl
        prm['elevation'] = mask * prm['elevation']
        prm['azimuth']   = mask * prm['azimuth']
        prm['Times']     = mask * prm['Times']

        # === 6. Cycle Slip Correction ========================================
        print("  Starting Cycle Slip Correction")
        STECl_M_new = CycleSlipCorrection_v2(
            STECl, prm['elevation'], prm['Times'], np.arange(1, n_sat + 1))

        STECl_M_new = STECl_M_new * mask
        # === 7. TEC Window Shift =============================================
        print("  Window Shift Adjustment")
        TEC['withbias'] = windowshiftTEC(STECl_M_new, STECp)

        # === 8. Satellite DCB ================================================
        try:
            Bias_sat = ((satb['P1C1'] - satb['P1P2'])
                        * (c * (f1**2 * f2**2) / (A * (f1**2 - f2**2) * 1e16)))
            DCB['sat'] = Bias_sat[:n_sat]
        except Exception as e:
            print("  DCB warning (" + str(e) + ") -- using zeros")
            DCB['sat'] = np.zeros(n_sat)

        TEC_adj   = TEC['withbias'] - DCB['sat']
        TEC_clean = outlinecorr(TEC_adj)
        TEC['withrcvbias'] = TEC_clean

        for PRN in range(n_sat):
            if np.sum(~np.isnan(TEC_clean[84900:86400, PRN])) < 1500:
                TEC['withrcvbias'][84900:86400, PRN] = np.nan

        # === 9. Receiver Bias ================================================
        slant_factor = np.sqrt(
            1 - (Re * np.cos(np.radians(prm['elevation'])) / (Re + h))**2)
        rcv_bias = rcv_bias_ma(-30, 30, 0.1, TEC_clean, slant_factor)
        DCB['rcv'] = rcv_bias * 1e-9 * (
            c * (f1**2 * f2**2) / (A * (f1**2 - f2**2) * 1e16))

        # === 10. Final TEC ===================================================
        TEC['slant']    = TEC_clean - DCB['rcv']
        TEC['vertical'] = TEC['slant'] * slant_factor

        slant_min = np.nanmin(TEC['slant'])
        if np.isfinite(slant_min) and slant_min < 0:
            TEC['slant']    -= slant_min
            TEC['vertical'] -= slant_min

        # === 11. ROTI — cycle-slip-corrected STECl [Fix 1] ===================
        ROTI = roticalculation(STECl_M_new, np.arange(1, n_sat + 1))

        # === 12. Save ========================================================
        os.makedirs(save_path, exist_ok=True)
        stem = os.path.join(save_path, str(station) + "_" + str(doy).zfill(3))
        _save(stem + "_VTEC",    TEC['vertical'])
        _save(stem + "_STEC",    TEC['slant'])
        _save(stem + "_ROTI",    ROTI)
        _save(stem + "_ele",     prm['elevation'])
        _save(stem + "_ipp_lat", prm['IPP_lat'])
        _save(stem + "_ipp_lon", prm['IPP_long'])

        print(" TEC done -- " + station + "  VTEC max " + str(round(np.nanmax(TEC['vertical']), 1)) + " TECU")
        return {"TEC": TEC, "DCB": DCB, "ROTI": ROTI, "prm": prm}

    except Exception as e:
        import traceback
        print("[calculate_tec] " + str(e))
        traceback.print_exc()
        return None
