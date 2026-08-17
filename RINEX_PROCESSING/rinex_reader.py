# =============================================================================
# rinex_reader.py  — RINEX 2.11 AND 3.x parser, native, no georinex dependency
# Handles: Thai CORS obs files, brdc nav files with Fortran D-exponent notation
# GPS-only, robust column parsing
# =============================================================================
import math
import numpy as np
from pathlib import Path
from datetime import datetime

N_EPH_COLS = 37   # fixed row width: 9 header + 28 orbit params


def _d2f(s: str) -> float:
    """Parse Fortran D-exponent float string, e.g. '0.344178D-03'."""
    return float(s.strip().replace('D', 'E').replace('d', 'e'))


def _detect_rinex_version(path: str) -> float:
    """Peek at the RINEX VERSION / TYPE line to decide which parser to use."""
    with open(path) as f:
        first_line = f.readline()
    try:
        return float(first_line[0:9])
    except Exception:
        return 2.11  # fall back to legacy behaviour if unreadable


def read_nav_only(nav_file: str):
    """Parse just the nav file (version-dispatched), independent of any
    obs file. Use this to parse a shared nav file ONCE per DOY and reuse
    it across every station that day, instead of re-parsing it per
    station via read_rinex211 (which re-reads both obs AND nav together).
    For a ~200-station day, this avoids re-parsing a large merged
    multi-GNSS nav file 200+ times instead of once."""
    try:
        nav_version = _detect_rinex_version(nav_file)
        if nav_version >= 3.0:
            return _parse_nav_304(nav_file)
        else:
            return _parse_nav_211(nav_file)
    except Exception as e:
        import traceback
        print(f"[read_nav_only] Error: {e}")
        traceback.print_exc()
        return None


def read_obs_only(obs_file: str, subsample: int = 30):
    """Parse just the obs file (version-dispatched), independent of nav.
    Pair with a cached read_nav_only() result to avoid re-parsing the
    shared nav file per station."""
    try:
        version = _detect_rinex_version(obs_file)
        if version >= 3.0:
            return _parse_obs_304(obs_file, subsample)
        else:
            return _parse_obs_211(obs_file, subsample)
    except Exception as e:
        import traceback
        print(f"[read_obs_only] Error: {e}")
        traceback.print_exc()
        return None


def read_rinex211(obs_file: str, nav_file: str,
                  subsample: int = 30) -> tuple:
    """Kept the original name so existing pipeline scripts don't need to
    change their imports. Internally dispatches to the 2.11 or 3.x obs
    parser based on the file's declared RINEX VERSION / TYPE."""
    try:
        version = _detect_rinex_version(obs_file)
        if version >= 3.0:
            obs_struct = _parse_obs_304(obs_file, subsample)
        else:
            obs_struct = _parse_obs_211(obs_file, subsample)

        nav_version = _detect_rinex_version(nav_file)
        if nav_version >= 3.0:
            nav_struct = _parse_nav_304(nav_file)
        else:
            nav_struct = _parse_nav_211(nav_file)
        date = datetime(obs_struct["date"][0],
                        obs_struct["date"][1],
                        obs_struct["date"][2])
        doy = date.timetuple().tm_yday
        return (obs_struct, nav_struct, doy,
                date.year, date.month, date.day,
                obs_struct["station"])
    except Exception as e:
        import traceback
        print(f"[read_rinex211] Error: {e}")
        traceback.print_exc()
        return None, None, None, None, None, None, None


# =============================================================================
# OBS PARSER — RINEX 2.11 (unchanged from original)
# =============================================================================

def _parse_obs_211(path: str, subsample: int = 30) -> dict:
    with open(path) as f:
        lines = f.readlines()

    hdr       = 0
    obs_types = []
    rx_ecef   = [0.0, 0.0, 0.0]
    station   = Path(path).stem[:4].upper()

    n_obs_expected = 0
    for i, l in enumerate(lines):
        if "APPROX POSITION XYZ" in l:
            try:
                rx_ecef = [float(l[0:14]), float(l[14:28]), float(l[28:42])]
            except Exception:
                pass
        if "# / TYPES OF OBSERV" in l:
            parts = l[:60].split()
            try:
                n = int(parts[0])
                n_obs_expected = n          # total obs types declared
                obs_types += parts[1:]      # accumulate from this line
            except Exception:
                # continuation line — no leading count
                obs_types += [p for p in parts if p]
        if "END OF HEADER" in l:
            hdr = i + 1
            break
    # trim to declared count (guards against parsing noise)
    if n_obs_expected > 0:
        obs_types = obs_types[:n_obs_expected]

    mat_lab_filter = {"C1", "L1", "P2", "L2", "P1"}
    n_obs = len(obs_types)
    lps   = math.ceil(n_obs / 5)

    epoch_list, index_list, data_list = [], [], []
    i      = hdr
    ep_num = 0
    date0  = None

    while i < len(lines):
        el = lines[i]
        if len(el) < 32:
            i += 1; continue
        try:
            yy   = int(el[1:3]); mo = int(el[4:6]); dd = int(el[7:9])
            hh   = int(el[10:12]); mm = int(el[13:15]); ss = float(el[15:26])
            n_sv = int(el[29:32])
        except Exception:
            i += 1; continue

        if date0 is None:
            date0 = [2000 + yy, mo, dd, hh, mm, int(ss)]

        sod = hh * 3600 + mm * 60 + int(ss)

        prns = [el[32:68][j * 3: j * 3 + 3].strip()
                for j in range(min(n_sv, 12))]
        i += 1
        while len(prns) < n_sv:
            ov    = lines[i][32:68]
            prns += [ov[j * 3: j * 3 + 3].strip()
                     for j in range(min(n_sv - len(prns), 12))]
            i += 1

        keep = (ep_num % subsample == 0)

        for sv in prns:
            sv_num = _parse_gps_prn(sv)
            if sv_num is None:
                for _ in range(lps): i += 1
                continue

            raw = ""
            for _ in range(lps):
                raw += (lines[i] if i < len(lines)
                        else "").rstrip("\n").ljust(80)[:80]
                i += 1

            if not keep:
                continue

            vals: dict[str, float | None] = {}
            for k, nm in enumerate(obs_types):
                col = (k % 5) * 16
                ro  = (k // 5) * 80
                chunk = raw[ro + col: ro + col + 14]
                try:
                    vals[nm] = float(chunk)
                except Exception:
                    vals[nm] = None

            row = [vals.get(t) for t in obs_types]
            epoch_list.append(sod)
            index_list.append(sv_num)
            data_list.append(row)

        ep_num += 1

    if date0 is None:
        raise ValueError(f"No epochs parsed from: {path}")

    obs_type_out = [t if t in mat_lab_filter else "" for t in obs_types]
    return {
        "date"   : date0,
        "epoch"  : epoch_list,
        "type"   : obs_type_out,
        "station": station,
        "data"   : data_list,
        "index"  : index_list,
        "rcvpos" : rx_ecef,
    }


# backwards-compat alias in case anything imports `_parse_obs` directly
_parse_obs = _parse_obs_211


# =============================================================================
# OBS PARSER — RINEX 3.x, GPS-only, mapped down to legacy C1/L1/P2/L2/P1 names
# =============================================================================

# Priority-ordered candidate RINEX3 codes for each legacy 2.11-style name.
# First match found in the file's declared GPS obs types wins.
_LEGACY_CODE_PRIORITY = {
    "C1": ["C1C", "C1P", "C1W", "C1X"],
    "L1": ["L1C", "L1P", "L1W", "L1X"],
    "P2": ["C2W", "C2P", "C2X", "C2C", "C2L", "C2S"],
    "L2": ["L2W", "L2P", "L2X", "L2C", "L2L", "L2S"],
    "P1": ["C1W", "C1P"],
}


def _map_rinex3_codes_to_legacy(raw_codes: list[str]) -> dict[int, str]:
    """Return {column_index: legacy_name} for whichever RINEX3 codes match
    the priority lists above. Each legacy name is assigned at most once."""
    assigned: dict[int, str] = {}
    used_names = set()
    for legacy_name, candidates in _LEGACY_CODE_PRIORITY.items():
        if legacy_name in used_names:
            continue
        for cand in candidates:
            if cand in raw_codes:
                idx = raw_codes.index(cand)
                if idx not in assigned:
                    assigned[idx] = legacy_name
                    used_names.add(legacy_name)
                    break
    return assigned


def _parse_obs_304(path: str, subsample: int = 30) -> dict:
    with open(path) as f:
        lines = f.readlines()

    hdr        = 0
    rx_ecef    = [0.0, 0.0, 0.0]
    station    = Path(path).stem[:4].upper()
    sys_obs    = {}          # sys_char -> list of obs codes (accumulated)
    sys_counts = {}          # sys_char -> declared count
    interval   = None
    current_sys = None

    for i, l in enumerate(lines):
        if "APPROX POSITION XYZ" in l:
            try:
                parts = l[:60].split()
                rx_ecef = [float(parts[0]), float(parts[1]), float(parts[2])]
            except Exception:
                pass

        if "SYS / # / OBS TYPES" in l:
            tokens = l[:60].split()
            if tokens and len(tokens[0]) == 1 and tokens[0].isalpha():
                sys_char = tokens[0]
                try:
                    count = int(tokens[1])
                except Exception:
                    count = 0
                sys_obs[sys_char]    = tokens[2:]
                sys_counts[sys_char] = count
                current_sys = sys_char
            else:
                # continuation line for the current system
                if current_sys:
                    sys_obs[current_sys] += tokens

        if "INTERVAL" in l:
            try:
                interval = float(l[:60].split()[0])
            except Exception:
                pass

        if "END OF HEADER" in l:
            hdr = i + 1
            break

    # trim each system's obs list to its declared count
    for sys_char, count in sys_counts.items():
        if count > 0:
            sys_obs[sys_char] = sys_obs[sys_char][:count]

    gps_codes = sys_obs.get("G", [])
    if not gps_codes:
        raise ValueError(f"No GPS observation types declared in: {path}")

    code_map = _map_rinex3_codes_to_legacy(gps_codes)  # idx -> legacy name
    n_codes  = len(gps_codes)

    # figure out the epoch-skip step so output cadence matches `subsample`
    # seconds regardless of the file's native interval
    if interval and interval > 0:
        step = max(1, round(subsample / interval))
    else:
        step = subsample  # fall back to legacy 1 Hz assumption

    epoch_list, index_list, data_list = [], [], []
    i      = hdr
    ep_num = 0
    date0  = None

    while i < len(lines):
        el = lines[i]
        if not el.startswith(">"):
            i += 1; continue

        tokens = el[1:].split()
        try:
            yyyy = int(tokens[0]); mo = int(tokens[1]); dd = int(tokens[2])
            hh   = int(tokens[3]); mm = int(tokens[4]); ss = float(tokens[5])
            n_sat = int(tokens[7])
        except Exception:
            i += 1; continue

        if date0 is None:
            date0 = [yyyy, mo, dd, hh, mm, int(ss)]

        sod  = hh * 3600 + mm * 60 + int(ss)
        keep = (ep_num % step == 0)
        i += 1

        for _ in range(n_sat):
            if i >= len(lines):
                break
            sat_line = lines[i]
            i += 1

            if not keep or not sat_line.startswith("G"):
                continue

            try:
                sv_num = int(sat_line[1:3])
            except Exception:
                continue
            if not (1 <= sv_num <= 32):
                continue

            obs_str = sat_line[3:].rstrip("\n")

            vals: dict[str, float | None] = {}
            for idx, legacy_name in code_map.items():
                chunk = obs_str[idx * 16: idx * 16 + 14]
                try:
                    vals[legacy_name] = float(chunk)
                except Exception:
                    vals[legacy_name] = None

            row = [vals.get(code_map.get(k, ""), None) for k in range(n_codes)]
            epoch_list.append(sod)
            index_list.append(sv_num)
            data_list.append(row)

        ep_num += 1

    if date0 is None:
        raise ValueError(f"No epochs parsed from: {path}")

    obs_type_out = [code_map.get(k, "") for k in range(n_codes)]
    return {
        "date"   : date0,
        "epoch"  : epoch_list,
        "type"   : obs_type_out,
        "station": station,
        "data"   : data_list,
        "index"  : index_list,
        "rcvpos" : rx_ecef,
    }


# =============================================================================
# NAV PARSER — RINEX 2, GPS-only, handles Fortran D-exponent notation
# (unchanged — brdc nav files are shared across stations and assumed 2.x)
# =============================================================================

def _parse_nav_211(path: str) -> dict:
    with open(path) as f:
        lines = f.readlines()

    hdr = next(i + 1 for i, l in enumerate(lines) if "END OF HEADER" in l)
    eph, nav_index = [], []
    i = hdr

    while i < len(lines):
        l = lines[i]
        if not l.strip():
            i += 1; continue

        # PRN: integer in cols 0-1 (RINEX 2 GPS broadcast nav)
        prn = _parse_prn_r2(l)
        if prn is None:
            i += 1; continue

        # Parse header line: PRN yy mm dd hh mm ss af0 af1 af2
        try:
            yy  = int(l[3:5].strip())
            mo  = int(l[6:8].strip())
            dd  = int(l[9:11].strip())
            hh  = int(l[12:14].strip())
            mm  = int(l[15:17].strip())
            ss  = _d2f(l[17:22])
            af0 = _d2f(l[22:41])
            af1 = _d2f(l[41:60])
            af2 = _d2f(l[60:79])
        except Exception:
            i += 1; continue

        i += 1

        # Read 7 orbit data lines, 4 params each (3-char indent, 19-char fields)
        params = []
        for _ in range(7):
            row = lines[i] if i < len(lines) else " " * 80
            for j in range(4):
                chunk = row[3 + j * 19: 3 + (j + 1) * 19]
                try:
                    params.append(_d2f(chunk))
                except Exception:
                    params.append(0.0)
            i += 1

        # Build fixed-width row
        row_data = [yy, mo, dd, hh, mm, ss, af0, af1, af2] + params
        # pad to N_EPH_COLS (37) in case of short file
        while len(row_data) < N_EPH_COLS:
            row_data.append(0.0)
        row_data = row_data[:N_EPH_COLS]

        eph.append(row_data)
        nav_index.append(prn)

    return {"eph": eph, "index": nav_index}


# backwards-compat alias in case anything imports `_parse_nav` directly
_parse_nav = _parse_nav_211


# =============================================================================
# NAV PARSER — RINEX 3.x, GPS-only
#
# RINEX 3's GPS navigation message content and field ORDER are the same as
# RINEX 2.11 -- only the header/epoch-line formatting changed (4-digit year,
# "G01"-style system+PRN prefix instead of a bare PRN). So this reuses the
# exact same eph column layout as _parse_nav_211 (verified against
# satpos_xyz_sbias in utils.py: a=eph[:,16], e=eph[:,14], w0=eph[:,23], etc.)
# rather than inventing a new one.
#
# Fields are extracted via whitespace-splitting rather than fixed-width
# column slicing, since the exact byte-column widths for RINEX 3 nav records
# vary slightly by generator -- splitting is tolerant of that, the same
# approach used for the RINEX 3 obs epoch/header lines above.
# =============================================================================

def _split_fixed19(line: str, start_col: int) -> list[str]:
    """Split a RINEX 3 nav data line into fixed 19-character-wide fields
    starting at start_col. MUST be used instead of .split() for these
    lines: fields are not reliably whitespace-separated -- a negative
    value's sign fills the slot where a leading space would otherwise be,
    so e.g. '...E-04-3.637978807090E-12...' is two adjacent 19-char fields
    with no separating space, and .split() would wrongly merge them into
    one unparseable token. This is the same issue already fixed for the
    RINEX 3 obs parser's epoch lines."""
    fields = []
    pos = start_col
    while pos < len(line):
        chunk = line[pos:pos + 19]
        if chunk.strip() == "":
            break
        fields.append(chunk)
        pos += 19
    return fields


def _parse_nav_304(path: str) -> dict:
    with open(path) as f:
        lines = f.readlines()

    hdr = 0
    for i, l in enumerate(lines):
        if "END OF HEADER" in l:
            hdr = i + 1
            break

    # RINEX 3 nav message record length (epoch line + orbit data lines)
    # varies by constellation. GPS/Galileo/BeiDou/QZSS use 8 lines total;
    # GLONASS and SBAS use only 4. Assuming a fixed 8 for every non-GPS
    # system (as an earlier version of this parser did) desyncs the whole
    # rest of the file the first time a GLONASS/SBAS record is hit.
    SYS_RECORD_LINES = {"G": 8, "E": 8, "C": 8, "J": 8, "R": 4, "S": 4}

    eph, nav_index = [], []
    i = hdr

    while i < len(lines):
        l = lines[i].rstrip("\n")
        if len(l) < 4 or not l[0].isalpha():
            i += 1; continue

        sys_char = l[0]
        if sys_char != "G":
            # Not GPS -- skip exactly this system's record length, not a
            # fixed 8 (see SYS_RECORD_LINES note above)
            i += SYS_RECORD_LINES.get(sys_char, 8)
            continue

        try:
            prn = int(l[1:3])
            # Date/time prefix (first 23 chars) IS safely whitespace-split
            # -- these fields are never negative, so no glued-sign issue.
            prefix_tokens = l[:23].split()
            yyyy = int(prefix_tokens[1]); mo = int(prefix_tokens[2]); dd = int(prefix_tokens[3])
            hh   = int(prefix_tokens[4]); mm = int(prefix_tokens[5]); ss = float(prefix_tokens[6])
            # Clock fields (af0/af1/af2) DO need fixed-width slicing --
            # this is what was broken before (naive .split() on these).
            clk_fields = _split_fixed19(l, 23)
            af0 = _d2f(clk_fields[0]); af1 = _d2f(clk_fields[1]); af2 = _d2f(clk_fields[2])
        except Exception:
            i += 1; continue

        yy = yyyy - 2000  # keep the 2-digit-year convention satpos_xyz_sbias expects
        i += 1

        # 7 orbit data lines, 4 fixed-width fields each -- same glued-sign
        # issue applies here too, so fixed-width slicing, not .split()
        params = []
        for _ in range(7):
            row = lines[i].rstrip("\n") if i < len(lines) else ""
            row_fields = _split_fixed19(row, 4)
            for j in range(4):
                if j < len(row_fields):
                    try:
                        params.append(_d2f(row_fields[j]))
                    except Exception:
                        params.append(0.0)
                else:
                    params.append(0.0)
            i += 1

        row_data = [yy, mo, dd, hh, mm, ss, af0, af1, af2] + params
        while len(row_data) < N_EPH_COLS:
            row_data.append(0.0)
        row_data = row_data[:N_EPH_COLS]

        eph.append(row_data)
        nav_index.append(prn)

    return {"eph": eph, "index": nav_index}


# =============================================================================
# HELPERS
# =============================================================================

def _parse_gps_prn(sv: str) -> int | None:
    sv = sv.strip()
    if not sv:
        return None
    try:
        if sv.startswith("G"):
            return int(sv[1:])
        n = int(sv)
        return n if 1 <= n <= 32 else None
    except Exception:
        return None


def _parse_prn_r2(line: str) -> int | None:
    try:
        prn = int(line[0:2])
        return prn if 1 <= prn <= 32 else None
    except Exception:
        return None