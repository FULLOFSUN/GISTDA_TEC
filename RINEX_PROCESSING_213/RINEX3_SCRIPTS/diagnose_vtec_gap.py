#!/usr/bin/env python3
"""
Check raw RINEX obs data availability per hour, independent of the TEC
calculation pipeline entirely -- tells us whether sparse VTEC output is
caused by sparse raw observations (receiver-side) or something breaking
downstream in elevation/position calculation.

Usage:
    python diagnose_raw_obs.py
"""
import numpy as np
from rinex_reader import read_rinex211

# EDIT to point at the actual obs/nav files
OBS_FILE = "/Users/ziraa/Documents/GISTDA/GISTDA_TEC/RINEX_by_DOY/DOY_014/AMKO/amko0140.26o"
NAV_FILE = "/Users/ziraa/Documents/GISTDA/GISTDA_TEC/RINEX3_Nav_File/BRDC00IGS_R_20260140000_01D_MN.rnx"

obs, nav, doy, year, month, day, stn = read_rinex211(OBS_FILE, NAV_FILE)

print(f"Station: {stn}")
print(f"Total epoch-satellite rows in obs['data']: {len(obs['data'])}")
print(f"nav ephemerides: {len(nav['eph'])}  |  unique nav PRNs: {sorted(set(nav['index']))}")

obs_data = np.array([[v if v is not None else np.nan for v in row] for row in obs['data']], dtype=float)
obs_types = obs['type']
obs_epoch = np.array(obs['epoch'])
obs_index = np.array(obs['index'])

type_idx = {}
for t in ['C1', 'P1', 'P2', 'L1', 'L2']:
    for j, ot in enumerate(obs_types):
        if ot == t:
            type_idx[t] = j
            break
if 'C1' not in type_idx and 'P1' in type_idx:
    type_idx['C1'] = type_idx['P1']

print(f"Observable columns found: {type_idx}")
print(f"obs['type'] full list: {obs_types}")

time_offset = obs['date'][3]*3600 + obs['date'][4]*60 + obs['date'][5]

print(f"\n{'Hour':>4}  {'#PRNs w/ raw C1+P2+L1+L2':>26}  {'Which PRNs':<40}")
for h in range(24):
    sod0, sod1 = h*3600, (h+1)*3600
    prns_this_hour = set()
    for prn in range(1, 33):
        sat_idx = np.where(obs_index == prn)[0]
        if sat_idx.size == 0:
            continue
        t_raw = obs_epoch[sat_idx] + time_offset
        in_window = (t_raw >= sod0) & (t_raw < sod1)
        if not in_window.any():
            continue
        rows = sat_idx[in_window]
        try:
            c1 = obs_data[rows, type_idx['C1']]
            p2 = obs_data[rows, type_idx['P2']]
            l1 = obs_data[rows, type_idx['L1']]
            l2 = obs_data[rows, type_idx['L2']]
        except KeyError:
            continue
        valid = (np.isfinite(c1) & np.isfinite(p2) & np.isfinite(l1) & np.isfinite(l2)
                 & (c1 != 0) & (p2 != 0) & (l1 != 0) & (l2 != 0))
        if valid.any():
            prns_this_hour.add(prn)
    print(f"{h:>4}  {len(prns_this_hour):>26}  {sorted(prns_this_hour)}")