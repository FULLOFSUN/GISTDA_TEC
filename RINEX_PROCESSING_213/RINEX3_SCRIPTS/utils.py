"""
Utility functions for TEC calculation (Python version).

Changes in this version:
  [Fix] download_dcb() switched from ftplib (FTP) to HTTPS, since AIUB
        decommissioned their anonymous FTP server in June 2026. See:
        https://www.aiub.unibe.ch/download
        *** The exact HTTPS path below (CODE_HTTPS_BASE) is a best guess
        based on their documented directory structure -- confirm it against
        the actual working path (e.g. via their full_listing.csv) before
        relying on it, since the migration returned a NoSuchKey error for
        at least one guessed path. ***
  [Fix] The `lg` "is this a recent month" check was comparing month to
        itself (always True). Now compares the requested obs month against
        today's actual date.
"""
import os
import requests
import gzip
import shutil
import numpy as np
from datetime import datetime, timedelta
import json

def reffromIPPindex(station_name, ppp_file='./TEC_Calculation/PPPindex.txt'):
    """
    Read station name from PPP file and return reference position (PPP solution).
    Args:
        station_name (str): station name (case-sensitive, e.g. 'KMIT')
        ppp_file (str): path to PPPindex.txt
    Returns:
        refpos (np.ndarray): reference position [X, Y, Z] or None if not found
    """
    try:
        with open(ppp_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        # Look for station_name in every line
        for line in lines:
            if station_name in line:
                # Split the name and value '='
                parts = line.split('=')
                if len(parts) == 2:
                    # Extract only the value after = and convert it to a list of floats
                    refpos_str = parts[1].strip().replace('[', '').replace('];', '')
                    refpos = [refpos_str.split()[0],refpos_str.split()[1],refpos_str.split()[2]]
                    return refpos
        print('No PPP reference or wrong station')
        return None
    except Exception as e:
        print(f'Error reading PPPindex.txt: {e}')
        return None

def gpscons():
    """Return GPS constants (migrated from MATLAB gpscons.m)."""
    # Constants from GPS system
    return {
        'c': 299792458,         # speed of light (m/s)
        'f1': 1575.42e6,        # L1 frequency (Hz)
        'f2': 1227.60e6,        # L2 frequency (Hz)
        'A': 40.3,              # TECU constant
        'Re': 6371e3,           # Earth radius (m)
        'h': 350e3,             # Ionospheric height (m)
        'elev_mask': 15         # Elevation mask (deg)
    }

def xyz2lla(x, y, z):
    """Convert ECEF XYZ to latitude, longitude, altitude (WGS-84)."""
    # WGS-84 ellipsoid parameters
    a = 6378137.0
    e = 8.1819190842622e-2
    b = np.sqrt(a**2 * (1 - e**2))
    ep = np.sqrt((a**2 - b**2) / b**2)
    p = np.sqrt(x**2 + y**2)
    th = np.arctan2(a * z, b * p)
    lon = np.arctan2(y, x)
    lat = np.arctan2(z + ep**2 * b * np.sin(th)**3, p - e**2 * a * np.cos(th)**3)
    N = a / np.sqrt(1 - e**2 * np.sin(lat)**2)
    alt = p / np.cos(lat) - N
    lat = np.degrees(lat) 
    lon = np.degrees(lon)
    return lat, lon, alt

def calelevation(satpos, xyz):
    """
    Calculate elevation and azimuth angles from satellite and user ECEF positions.
    satpos: shape (3, N)  (satellite positions, columns are satellites)
    xyz: shape (3,)       (user ECEF position)
    Returns:
        elev: array of elevation angles (deg)
        azi: array of azimuth angles (deg)
    """

    # Helper: degrees trig
    sind = lambda x: np.sin(np.deg2rad(x))
    cosd = lambda x: np.cos(np.deg2rad(x))
    atan2d = lambda y, x: np.rad2deg(np.arctan2(y, x))

    # Convert ECEF to LLA
    satpos = np.array(satpos, dtype=float)
    xyz = np.array(xyz, dtype=float)
  
    user_lla = xyz2lla(xyz[0], xyz[1], xyz[2])  # (lat, lon, h)
    xyz_lla = xyz2lla(satpos[0, :], satpos[1, :], satpos[2, :])  # (lat, lon, h), each is array
    Lat = user_lla[0]
    Lon = user_lla[1]
    Lat_s = xyz_lla[0]
    Lon_s = xyz_lla[1]
    # Rotation matrix
    R = np.array([
        [-sind(Lon),                cosd(Lon),                0],
        [-sind(Lat)*cosd(Lon), -sind(Lat)*sind(Lon),  cosd(Lat)],
        [cosd(Lat)*cosd(Lon),  cosd(Lat)*sind(Lon),   sind(Lat)]
    ])

    # Vector from user to satellite(s)
    Rs = np.vstack((satpos[0, :] - xyz[0],
                    satpos[1, :] - xyz[1],
                    satpos[2, :] - xyz[2]))
    
    # Transform to local ENU
    RL = R @ Rs  # shape (3, N)
    RL = RL.T    # shape (N, 3)

    Xl = RL[:, 0]
    Yl = RL[:, 1]
    Zl = RL[:, 2]

    elev = np.rad2deg(np.arctan2(Zl, np.sqrt(Xl**2 + Yl**2)))
   
    azi = atan2d(Yl, Xl)
    return elev, azi


def juliandate(dt):
    # dt: datetime object
    # Returns Julian date
    a = (14 - dt.month)//12
    y = dt.year + 4800 - a
    m = dt.month + 12*a - 3
    JDN = dt.day + ((153*m + 2)//5) + 365*y + y//4 - y//100 + y//400 - 32045
    JD = JDN + (dt.hour - 12)/24 + dt.minute/1440 + dt.second/86400
    return JD

def satpos_xyz_sbias(Time, PRN, eph, index, ps):
    """
    Calculate GPS satellite position and clock bias.
    Inputs:
        Time  : array-like, seconds of day
        PRN   : int, PRN number
        eph   : numpy array, ephemeris data
        index : array-like, navigation index
        ps    : array-like, pseudorange
    Outputs:
        satpos   : numpy array, satellite positions (3 x N)
        satclock : numpy array, satellite clock bias (N,)

    [Perf fix] Vectorized over Time -- the original looped per-epoch in
    pure Python (plus a per-epoch while-loop for the Kepler equation),
    which is extremely slow across ~200+ stations x 32 PRNs. This computes
    all epochs at once with NumPy. The dynamic `while dE > 1e-12`
    convergence check became a fixed 10 Newton-Raphson iterations (GPS
    eccentricities are ~0.01-0.02, so this converges far beyond 1e-12
    within 3-4 iterations -- 10 is a safe margin, not a behavior change).

    [Fix] If a PRN has no ephemeris for this day (Sat is empty -- e.g. the
    satellite was unhealthy/off that day), this now returns NaN arrays
    instead of crashing with a cryptic IndexError ("index 0 is out of
    bounds for axis 0 with size 0"). The calling per-PRN loop in
    tec_calculator.py already treats all-NaN results as "no data for this
    PRN", so this is the correct behavior, not a suppressed error.
    """
    eph = np.array(eph)
    index = np.array(index)
    ps = np.array(ps, dtype=float)
    Time = np.array(Time, dtype=float)
    n_epochs = len(Time)

    # Find the satellite ephemeris rows for this PRN
    Sat = np.where(index == PRN)[0]
    if Sat.size == 0:
        return (np.full((3, n_epochs), np.nan),
                np.full(n_epochs, np.nan))

    # Orbit Parameters
    a       = eph[Sat,16]**2      # Semi-major axis (m)
    e       = eph[Sat,14]         # Eccentricity
    w0      = eph[Sat,23]         # Argument of perigee (rad)
    W0      = eph[Sat,19]         # Right ascension of ascending node (rad)
    Wdot    = eph[Sat,24]         # Rate of right ascension (rad/sec)
    i0      = eph[Sat,21]         # Inclination (rad)
    idot    = eph[Sat,25]         # Rate of inclination (rad/sec)
    M0      = eph[Sat,12]         # Mean anomaly (rad)
    delta_n = eph[Sat,11]         # Mean motion rate (rad/sec)

    # Correction coefficients
    Cuc     = eph[Sat,13]
    Cus     = eph[Sat,15]
    Crc     = eph[Sat,22]
    Crs     = eph[Sat,10]
    Cic     = eph[Sat,18]
    Cis     = eph[Sat,20]

    # Time
    Toe     = eph[Sat,17]
    y       = eph[Sat,0]
    m       = eph[Sat,1]
    d       = eph[Sat,2]

    # Clock
    T0_bias = eph[Sat,6]
    T0_drift= eph[Sat,7]
    T0_drate= eph[Sat,8]
    Tgd     = eph[Sat,31]

    # Constants
    GM = 3.986004418e14
    We = 7.2921151467e-5
    c = 299792458

    # Julian date (from the first ephemeris row's date -- same as original)
    Y = int(2000 + y[0])
    MO = int(m[0])
    D = int(d[0])
    YMD = datetime(Y, MO, D)
    JD = juliandate(YMD)

    # GPS Week and DOW
    GPSW = int((JD - 2444244.5)//7)
    DOW = ((JD - 2444244.5)/7 - GPSW)

    # SOW: second of GPS week, vectorized over all epochs at once
    SOW = DOW * (7 * 24 * 60 * 60) + Time   # shape (n_epochs,)

    # Find closest Toe per epoch, vectorized (broadcast against all
    # ephemeris rows for this PRN, then argmin along that axis)
    diff = np.abs(SOW[:, None] - Toe[None, :])      # (n_epochs, n_eph_rows)
    col  = np.argmin(diff, axis=1)                  # (n_epochs,)

    a_c, e_c   = a[col], e[col]
    w0_c, W0_c = w0[col], W0[col]
    Wdot_c, i0_c, idot_c = Wdot[col], i0[col], idot[col]
    M0_c, delta_n_c      = M0[col], delta_n[col]
    Cuc_c, Cus_c, Crc_c, Crs_c = Cuc[col], Cus[col], Crc[col], Crs[col]
    Cic_c, Cis_c, Toe_c        = Cic[col], Cis[col], Toe[col]
    T0_bias_c, T0_drift_c, T0_drate_c = T0_bias[col], T0_drift[col], T0_drate[col]
    Tgd_c = Tgd[col]

    tr  = ps / c
    TOS = SOW - tr
    Tk  = TOS - Toe_c

    n0  = np.sqrt(GM / a_c**3)
    MA_ = M0_c + (n0 + delta_n_c) * Tk

    # Vectorized Newton-Raphson for eccentric anomaly (fixed iterations --
    # see docstring note on why this is equivalent to the original's
    # dynamic convergence check for GPS-range eccentricities)
    EA = MA_.copy()
    for _ in range(10):
        EA = MA_ + e_c * np.sin(EA)

    TA = np.arctan2(np.sqrt(1 - e_c**2) * np.sin(EA), np.cos(EA) - e_c)

    W = W0_c + (Wdot_c - We) * Tk - (We * Toe_c)
    w = w0_c + Cuc_c*np.cos(2*(w0_c+TA)) + Cus_c*np.sin(2*(w0_c+TA))
    r = a_c*(1 - e_c*np.cos(EA)) + Crc_c*np.cos(2*(w0_c+TA)) + Crs_c*np.sin(2*(w0_c+TA))
    i = i0_c + idot_c*Tk + Cic_c*np.cos(2*(w0_c+TA)) + Cis_c*np.sin(2*(w0_c+TA))

    x_in = r * np.cos(TA)
    y_in = r * np.sin(TA)
    # z_in is always 0 in the original, so the third rotation-matrix column
    # never contributes -- only R[:,0] and R[:,1] are needed.

    cosW, sinW = np.cos(W), np.sin(W)
    cosw, sinw = np.cos(w), np.sin(w)
    cosi, sini = np.cos(i), np.sin(i)

    R00 = cosW*cosw - sinW*sinw*cosi
    R01 = -cosW*sinw - sinW*cosw*cosi
    R10 = sinW*cosw + cosW*sinw*cosi
    R11 = -sinW*sinw + cosW*cosw*cosi
    R20 = sinw*sini
    R21 = cosw*sini

    satpos = np.vstack([
        R00*x_in + R01*y_in,
        R10*x_in + R11*y_in,
        R20*x_in + R21*y_in,
    ])

    # Clock error computation
    F = -4.442807633e-10
    r_c  = F * e_c * np.sqrt(a_c) * np.sin(EA)
    t_sv = T0_bias_c + T0_drift_c*Tk + T0_drate_c*(Tk**2) + r_c
    satclock = t_sv - Tgd_c

    return satpos, satclock

def windowshiftTEC(STECl, STECp):
    """
    Window Shift TEC adjustment (migrated from MATLAB windowshiftTEC.m)
    Args:
        STECl: np.ndarray, STEC from carrier phase (time, sat)
        STECp: np.ndarray, STEC from code range (time, sat)
    Returns:
        STEC_adj: np.ndarray, adjusted STEC (time, sat)
    """
    STECl = np.array(STECl)
    STECp = np.array(STECp)
    STEC_adj = np.full(STECl.shape, np.nan)
    for LP in range(STECl.shape[1]):
        val = np.where(~np.isnan(STECl[:, LP]))[0]
        zz = np.where(np.diff(val) >= 10800)[0]  # 3hr*3600
     
        if zz.size == 0:
            diff_mean = np.nanmean(STECp[:, LP] - STECl[:, LP])
            STEC_adj[:, LP] = STECl[:, LP] + diff_mean
            continue
        wind = []
        for wd in range(len(zz)):
            if val[zz[wd]] <= 82800:
                wind.append(val[zz[wd]] + 3600)
        wind.append(len(STECl))
        for SW in range(len(wind)):
            if SW == 0:
                idx0 = 0
            else:
                idx0 = wind[SW-1]
            idx1 = wind[SW]
            diff_mean = np.nanmean(STECp[idx0:idx1, LP] - STECl[idx0:idx1, LP])
            STEC_adj[idx0:idx1, LP] = STECl[idx0:idx1, LP] + diff_mean
    return STEC_adj

def outlinecorr(STEC):

    try: 

        """
        Remove outlier of STEC (migrated from MATLAB outlinecorr.m)
        Args:
            STEC: np.ndarray (time, sat)
        Returns:
            STEC_rm_ol: np.ndarray (time, sat)
        """

        STEC = np.array(STEC)
        STECo = STEC
        STEC_m = np.nanmedian(STECo, axis=1)
        bound = 10  # TECu
  
        
        for pp in range(STEC.shape[1]):
            val = np.where(~np.isnan(STEC[:, pp]))[0]
            z = np.where(np.diff(val) >= 3600)[0]  # 1hr
            if z.size == 0:
                if np.abs(np.nanmean(STEC[:, pp] - STEC_m)) > bound:
                    STEC[:, pp] = np.nan
                continue
            z = np.append(z, len(val)-1)
            for y in range(len(z)):
                if y == 0:
                    idx0 = val[0]
                else:
                    idx0 = val[z[y-1]+1]
                idx1 = val[z[y]]
                countdif = np.abs(STEC[idx0:idx1+1, pp] - STEC_m[idx0:idx1+1]) > bound
                r_length = np.sum(~np.isnan(STEC_m[idx0:idx1+1]))
                if np.sum(countdif) > r_length/2:
                    STEC[idx0:idx1+1, pp] = np.nan
        return STEC

    except Exception as e:
        print(f'MFK error outlinecorr',e)

def rcv_bias_ma(sp, ep, step, STEC_removesatbias, slant_factor):
    """
    Estimate receiver bias (G. Ma and T. Maruyama, 2003) migrated from MATLAB rcv_bias_ma.m
    Args:
        sp: float, minimum delay
        ep: float, maximum delay
        step: float, step size
        STEC_removesatbias: np.ndarray, STEC without satellite DCB
        slant_factor: np.ndarray, slant factor
    Returns:
        rcv_bias_ns: float, receiver DCB in nanoseconds
    """
    STEC_removesatbias = STEC_removesatbias[::30, :]
    slant_factor = slant_factor[::30, :]
    br = np.arange(sp, ep+step, step)
    flac = True
    f1 = 1575.42e6
    f2 = 1227.60e6
    c = 299792458
    A = 40.3
    for loop in range(6):
        if not flac:
            br = np.arange(rcv_bias_ns-(step/(10**(loop-1))), rcv_bias_ns+(step/(10**(loop-1)))+step/(10**loop), step/(10**loop))
        std_vtec = np.zeros(len(br))
        for bri, b in enumerate(br):
            Br = b * (c * (f1**2 * f2**2 / (A * (f1**2 - f2**2) * 1e16))) * 1e-9
            STEC_adj_allnobias = STEC_removesatbias - Br
            VTEC_no_sat_bias1 = STEC_adj_allnobias * slant_factor
            VTEC_std_during = np.nanstd(VTEC_no_sat_bias1, axis=1)
            std_vtec[bri] = np.nansum(VTEC_std_during)
        min_idx = np.nanargmin(std_vtec)
        rcv_bias_ns = br[min_idx]
        flac = False
    return rcv_bias_ns

def roticalculation(STECl, PRNall):
    """
    Calculate Rate Of TEC Index (ROTI) migrated from MATLAB roticalculation.m
    Args:
        STECl: np.ndarray (time, sat)
        PRNall: list or array of PRN numbers (1-based)
    Returns:
        ROTI_sec: np.ndarray (time, sat)
    """
    STECl = np.array(STECl)
    ROTI_sec = np.full(STECl.shape, np.nan ,dtype=float)
    for PRN in PRNall:
        STEC_sec = STECl[:, PRN-1]  # PRN is 1-based
        ST = np.where(~np.isnan(STEC_sec))[0]
        flag = np.where((np.diff(ST) > 1) & (np.diff(ST) <= 60))[0]
        if flag.size > 0:
            for d in flag:
                x = [0, len(STEC_sec[ST[d]:ST[d+1]])-1]
                v = [STEC_sec[ST[d]], STEC_sec[ST[d+1]]]
                xq = np.arange(len(STEC_sec[ST[d]:ST[d+1]]))
                STEC_M = np.interp(xq, x, v)
                STEC_sec[ST[d]:ST[d+1]] = STEC_M
        for T in range(299, len(STEC_sec)):
            if np.isnan(STEC_sec[T]):
                ROTI_sec[T, PRN-1] = np.nan
            else:
                STEC_min = STEC_sec[T-299:T+1:60]
                ROT = np.diff(STEC_min)
                ROT = np.append(ROT, np.nan)
                ROTI_sec[T, PRN-1] = np.nanstd(ROT)
    return ROTI_sec


def CycleSlipCorrection_v2(stecl, elevation, time, prn_all):
    """
    Corrects cycle slip and missing data in STECl data based on elevation and time.
    
    Parameters:
        stecl (np.ndarray): 2D array of STEC values (shape: epochs x PRNs)
        elevation (np.ndarray): 2D array of elevation angles (same shape as stecl)
        time (np.ndarray): 2D array of time (same shape as stecl)
        prn_all (list or array): list of PRN indices to process
    
    Returns:
        np.ndarray: Corrected STECl array (same shape as input)
    """
    stec_new = np.full_like(stecl, np.nan ,dtype=float) 
    for prn in prn_all:
        time_col = time[:, prn-1]
        valid_idx = ~np.isnan(time_col)
        index = np.where(valid_idx)[0]
        
        if len(index) == 0:
            continue
        
        T = time[index, prn-1]
        stec = stecl[index, prn-1].copy()
        stec_ref = stecl[index, prn-1]
        elev = elevation[index, prn-1]
        
        # Calculate gap threshold based on std of diff(STEC)
        gap_bound = np.nanstd(np.diff(stec))
        gap_bound = max(1, min(gap_bound, 2))  # clamp between 1 and 2 TECu
        
        gap_time = 3 * 60 * 60  # 3 hours in seconds
        dif_time = np.diff(T)
        time_gap_idx = np.where(dif_time >= gap_time)[0]

        if len(time_gap_idx) == 0:
            # 1. No large time gaps
            slip1 = np.where(np.isnan(stec))[0]
            for i in slip1:
                try:
                    stec[i] = stec[i - 1]
                except IndexError:
                    stec[i] = stec[i + 1]
            
            diff_stec = np.abs(np.diff(stec))
            slip2 = np.where(diff_stec >= gap_bound)[0]
            slip2 = np.unique(np.concatenate([slip2, [len(stec) - 1]]))
            
            for i in range(len(slip2) - 1):
                b1 = stec[:slip2[i]+1]
                b2 = stec[slip2[i]+1:slip2[i+1]+1]
                if len(b2) > 0:
                    tec_part = b1 - (b1[-1] - b2[0])
                    stec[:slip2[i+1]+1] = np.concatenate([tec_part, b2])
        
        else:
            # 2. With large time gaps
            tg = np.concatenate([[0], time_gap_idx + 1, [len(stec)]])
            for i in range(len(tg) - 1):
                stec_g = stec[tg[i]:tg[i+1]].copy()
                slip1_g = np.where(np.isnan(stec_g))[0]
                for j in slip1_g:
                    try:
                        stec_g[j] = stec_g[j - 1]
                    except IndexError:
                        stec_g[j] = stec_g[j + 1]

                diff_stec_g = np.abs(np.diff(stec_g))
                slip2_g = np.where(diff_stec_g >= gap_bound)[0]
                slip2_g = np.unique(np.concatenate([slip2_g, [len(stec_g) - 1]]))

                for k in range(len(slip2_g) - 1):
                    b1 = stec_g[:slip2_g[k]+1]
                    b2 = stec_g[slip2_g[k]+1:slip2_g[k+1]+1]
                    if len(b2) > 0:
                        tec_part = b1 - (b1[-1] - b2[0])
                        stec_g[:slip2_g[k+1]+1] = np.concatenate([tec_part, b2])

                stec[tg[i]:tg[i+1]] = stec_g
        
        # Levelling to highest elevation
        ref_idx = np.nanargmax(elev)
        stec += (stec_ref[ref_idx] - stec[ref_idx])

        stec_new[index, prn-1] = stec

    return stec_new

def checkfileRN(r_o_name, rinex_path):
    """
    Check input the observation file (migrated from MATLAB checkfileRN.m)
    Args:
        r_o_name: str, observation file name
        rinex_path: str, RINEX file directory
    Raises:
        FileNotFoundError or ValueError with informative message
    """
    import os
    if not r_o_name:
        raise ValueError('Please define/change the observation file')
    full_path = os.path.join(rinex_path, r_o_name)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f'Please copy Observation file to {rinex_path}')
    # Optionally, check for exact filename match (case-sensitive)
    if r_o_name not in os.listdir(rinex_path):
        raise FileNotFoundError(f'Observation file {r_o_name} not found in {rinex_path}')
    return True


# =============================================================================
# DCB DOWNLOAD -- HTTPS (AIUB's FTP server was decommissioned June 2026)
# =============================================================================

# *** CONFIRMED *** against the actual bucket listing (full_listing.csv from
# the direct S3 endpoint, checked 2026-07-24):
#   - Current/sliding files live under BSWUSER52/ORB/, NOT under CODE/
#     e.g. BSWUSER52/ORB/P1C1.DCB (last updated 2026-07-23)
#   - Archived monthly files stay under CODE/yyyy/ with the plain naming
#     this code already expected: CODE/2026/P1C12601.DCB.Z (confirmed
#     present). A "_RINEX" variant also exists alongside it
#     (P1C12601_RINEX.DCB.Z) but is a different/larger product, not this
#     one -- an earlier guess based on a single 1991 sample assumed the
#     "_RINEX" suffix was the only/correct variant; it isn't, for 2026.
CODE_HTTPS_BASE = "https://zhw-b.s3.cloud.switch.ch/aiub/"
CODE_CURRENT_DIR = "BSWUSER52/ORB/"   # current sliding P1C1.DCB / P1P2.DCB
CODE_ARCHIVE_DIR = "CODE/"            # archived CODE/{year}/P1C1yymm.DCB.Z (plain, confirmed against 2026 listing)

def ReadRowDCB(filename):
    """Count the number of rows in a DCB file (migrated from MATLAB ReadRowDCB.m)."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    if not lines:
        raise ValueError(f"DCB file is empty -- download likely failed or "
                          f"returned an error page instead of real content: {filename}")
    return len(lines)

def ReadDCB(filename, number_row):
    """Parse DCB file and return DCB values (migrated from MATLAB ReadDCB.m)."""
    dcb = np.full(32, np.nan)
    num = []
    fid2 = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for i in range(7, number_row+1):  # MATLAB 8~number_row, Python 7~number_row-1
            line = lines[i]
            try:
                prn = int(line[1:3])
                value = float(line[29:35])
                num.append(prn)
                fid2.append(value)
            except Exception:
                continue
    for idx, prn in enumerate(num):
        if 1 <= prn <= 32:
            dcb[prn-1] = fid2[idx]
    return dcb

def https_download_file(url, dest, timeout=30):
    """Download a file over HTTPS (replaces the old ftplib-based
    ftp_download_file -- AIUB decommissioned their FTP server in June 2026)."""
    r = requests.get(url, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    with open(dest, 'wb') as f:
        f.write(r.content)

def download_dcb(obs, save_path):
    """
    Download and read DCB files from CODE's product server over HTTPS.
    Implements dlsat.m logic, updated from FTP to HTTPS.
    Returns satellite bias data structure.
    """
    year = str(obs['date'][0])
    month = f"{obs['date'][1]:02d}"
    P1C1_filename = f"P1C1{year[2:]}{month}.DCB"
    P1P2_filename = f"P1P2{year[2:]}{month}.DCB"
    dcb_dir = os.path.join(save_path, 'DCB')
    os.makedirs(dcb_dir, exist_ok=True)

    # [Fix] This used to compare int(month) to itself (always True).
    # Now actually checks whether the requested month is the current or
    # previous calendar month relative to today -- i.e. whether CODE's
    # *sliding* generic P1C1.DCB/P1P2.DCB files (which only cover the last
    # ~2 months) are likely to still contain this month's data, vs. needing
    # the permanently-archived dated file instead.
    requested = datetime(int(year), int(month), 1)
    now = datetime.now()
    months_diff = (now.year - requested.year) * 12 + (now.month - requested.month)
    lg = months_diff in (0, 1)

    if (not os.path.exists(os.path.join(dcb_dir, P1C1_filename)) and not os.path.exists(os.path.join(dcb_dir, P1P2_filename))) and lg:
        P1C1_filename = 'P1C1.DCB'
        P1P2_filename = 'P1P2.DCB'

    def unzip_file(src, dest):
        """Decompress a UNIX .Z (LZW `compress`) file. NOT gzip -- different
        magic bytes (\\x1f\\x9d vs gzip's \\x1f\\x8b) and algorithm. Python's
        gzip module cannot read these; shell out to the system `gzip -d`
        instead, which auto-detects and handles .Z despite its name."""
        import subprocess
        result = subprocess.run(["gzip", "-dc", str(src)], capture_output=True)
        if result.returncode != 0 or not result.stdout:
            # fall back to `uncompress` if gzip -d didn't handle it
            result = subprocess.run(["uncompress", "-c", str(src)], capture_output=True)
        if not result.stdout:
            raise RuntimeError(f"Could not decompress {src}: {result.stderr.decode(errors='ignore')}")
        with open(dest, 'wb') as f_out:
            f_out.write(result.stdout)

    P1C1_path = os.path.join(dcb_dir, P1C1_filename)
    P1P2_path = os.path.join(dcb_dir, P1P2_filename)

    if len(P1C1_filename) == 8 and len(P1P2_filename) == 8:
        # Download the sliding generic .DCB files -- these live under
        # BSWUSER52/ORB/, confirmed against the actual bucket listing.
        for fname, path in [(P1C1_filename, P1C1_path), (P1P2_filename, P1P2_path)]:
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                https_download_file(CODE_HTTPS_BASE + CODE_CURRENT_DIR + fname, path)
        P1C1Row_fileDCB = ReadRowDCB(P1C1_path)
        P1C1_DCB = ReadDCB(P1C1_path, P1C1Row_fileDCB-2)
        P1P2_DCB = ReadDCB(P1P2_path, P1C1Row_fileDCB-2)

    elif (os.path.exists(P1C1_path) and os.path.exists(P1P2_path)
          and os.path.getsize(P1C1_path) > 0 and os.path.getsize(P1P2_path) > 0):
        # Already cached locally
        P1C1Row_fileDCB = ReadRowDCB(P1C1_path)
        P1C1_DCB = ReadDCB(P1C1_path, P1C1Row_fileDCB-2)
        P1P2_DCB = ReadDCB(P1P2_path, P1C1Row_fileDCB-2)

    else:
        # Dated/archived files, .Z-compressed, under CODE/{year}/ -- but
        # confirmed against the actual 2026 bucket listing: the PLAIN
        # filename (no suffix) is what exists for recent years --
        # e.g. CODE/2026/P1C12601.DCB.Z, CODE/2026/P1P22601.DCB.Z.
        # (An earlier guess based on a 1991 sample assumed an inserted
        # "_RINEX" suffix -- that variant does exist too, but as a
        # different/larger product, not the one matching this code's
        # expected DCB format. The plain naming is correct for 2026.)
        P1C1_z, P1P2_z = f"{P1C1_filename}.Z", f"{P1P2_filename}.Z"
        P1C1_z_path = os.path.join(dcb_dir, P1C1_z)
        P1P2_z_path = os.path.join(dcb_dir, P1P2_z)
        for fname, zpath in [(P1C1_z, P1C1_z_path), (P1P2_z, P1P2_z_path)]:
            if not os.path.exists(zpath[:-2]) or os.path.getsize(zpath[:-2]) == 0:
                https_download_file(f"{CODE_HTTPS_BASE}{CODE_ARCHIVE_DIR}{year}/{fname}", zpath)
                unzip_file(zpath, zpath[:-2])
        P1C1Row_fileDCB = ReadRowDCB(P1C1_z_path[:-2])
        P1C1_DCB = ReadDCB(P1C1_z_path[:-2], P1C1Row_fileDCB-2)
        P1P2_DCB = ReadDCB(P1P2_z_path[:-2], P1C1Row_fileDCB-2)

    satb_P1C1 = P1C1_DCB[:32] * 1e-9
    satb_P1P2 = P1P2_DCB[:32] * 1e-9
    return {'P1C1': satb_P1C1, 'P1P2': satb_P1P2}


# DEBUG Function

def convert_ndarray_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_ndarray_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray_to_list(i) for i in obj]
    else:
        return obj