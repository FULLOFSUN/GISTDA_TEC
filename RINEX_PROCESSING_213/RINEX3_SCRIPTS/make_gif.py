# =============================================================================
# make_gif.py — Convert TEC map PNGs to animated GIF
# Supports both:
#   - 2-hour panel PNGs (one file per day)
#   - 5-minute frame PNGs (one file per frame, across all days)
#
# Updated:
#   - BASE path updated to the current GISTDA_TEC layout
#   - Added ROTI/IPP 5-min GIF configs for both the 17-station and
#     ~213-station networks. These previously had no entry here at all --
#     the ROTI script's frame folders (fixed to "frames_5min_{date}",
#     matching collect_5min_frames()'s expected pattern) are now wired up.
#
# Usage:
#   conda activate base
#   python make_gif.py
# =============================================================================

from pathlib import Path
from PIL import Image
from datetime import datetime, timedelta

BASE = Path("/Users/ziraa/Documents/GISTDA/GISTDA_TEC/TEC_output")

DOY_START = 14
DOY_END   = 30
YEAR      = 2026

# ── edit these settings ────────────────────────────────────────────────────────
FRAME_DURATION_MS = 800        # ms per frame for 2H panel GIFs
FRAME_DURATION_5MIN_MS = 120   # ms per frame for 5-min GIFs (faster animation)
LOOP              = 0          # 0 = loop forever
RESIZE_WIDTH      = None       # set e.g. 1200 to shrink; None = keep original

# =============================================================================
# HELPER — collect all frames from multiple day folders in order
# =============================================================================

def collect_5min_frames(method_folder_prefix: Path) -> list:
    """
    Collects all 5-minute frame PNGs across all days in chronological order.
    Looks for folders named: frames_5min_2026-01-14, frames_5min_2026-01-15, ...
    """
    base = Path(method_folder_prefix)
    all_frames = []

    for doy in range(DOY_START, DOY_END + 1):
        date     = datetime(YEAR, 1, 1) + timedelta(days=doy - 1)
        date_str = date.strftime("%Y-%m-%d")
        folder   = base / f"frames_5min_{date_str}"

        if not folder.exists():
            print(f"  [skip] {folder.name} not found")
            continue

        frames = sorted(folder.glob("frame_*.png"),
                        key=lambda p: int(p.stem.split("_")[1]))

        if not frames:
            print(f"  [skip] {folder.name} — no frames found")
            continue

        all_frames.extend(frames)
        print(f"  DOY{doy:03d} ({date_str}): {len(frames)} frames")

    return all_frames


# =============================================================================
# HELPER — make GIF from a list of file paths
# =============================================================================

def make_gif_from_files(files: list, output: Path,
                        duration_ms: int = 800,
                        loop: int = 0,
                        resize_width: int | None = None):
    if not files:
        print(f"  [skip] no files provided for {output.name}")
        return

    print(f"  {len(files)} frames → {output.name}")

    frames = []
    for fp in files:
        img = Image.open(fp).convert("RGBA")
        if resize_width:
            ratio = resize_width / img.width
            img   = img.resize((resize_width, int(img.height * ratio)),
                               Image.LANCZOS)
        frames.append(img)

    frames_p = [f.convert("RGB").convert("P", palette=Image.ADAPTIVE,
                                          colors=256)
                for f in frames]

    frames_p[0].save(
        output,
        format="GIF",
        save_all=True,
        append_images=frames_p[1:],
        duration=duration_ms,
        loop=loop,
        optimize=True,
    )
    size_mb = output.stat().st_size / 1e6
    print(f"  Saved: {output}  ({size_mb:.1f} MB)")


# =============================================================================
# HELPER — original single-folder GIF (for 2H panel PNGs)
# =============================================================================

def make_gif_from_folder(folder: Path, pattern: str, output: Path,
                         duration_ms: int = 800,
                         loop: int = 0,
                         resize_width: int | None = None):
    files = sorted(folder.glob(pattern))
    if not files:
        print(f"  [skip] no files matching {folder/pattern}")
        return
    make_gif_from_files(files, output, duration_ms, loop, resize_width)


# =============================================================================
# CONFIGS
# =============================================================================

def main():
    print(f"Generating GIFs for DOY {DOY_START:03d}–{DOY_END:03d} ({YEAR})")
    print()

    # ── 2-hour panel GIFs (one PNG per day) ────────────────────────────────
    panel_configs = [
        {"name": "TPS (17 stations)",     "folder": BASE / "TEC_maps",
         "pattern": "VTEC_map_2026-01-*.png",         "output": BASE / "VTEC_animation_TPS_17stations.gif"},
        {"name": "IDW (17 stations)",     "folder": BASE / "TEC_maps_IDW_17stations",
         "pattern": "VTEC_map_IDW_2026-01-*.png",      "output": BASE / "VTEC_animation_IDW_17stations.gif"},
        {"name": "Kriging (17 stations)", "folder": BASE / "TEC_maps_Kriging_17stations",
         "pattern": "VTEC_map_Kriging_2026-01-*.png",  "output": BASE / "VTEC_animation_Kriging_17stations.gif"},
    ]

    print("── 2-hour panel GIFs ─────────────────────────────")
    for cfg in panel_configs:
        print(f"[{cfg['name']}]")
        if not cfg["folder"].exists():
            print(f"  [skip] folder not found: {cfg['folder']}")
            continue
        make_gif_from_folder(
            cfg["folder"], cfg["pattern"], cfg["output"],
            duration_ms=FRAME_DURATION_MS,
            loop=LOOP,
            resize_width=RESIZE_WIDTH,
        )
        print()

    # ── 5-minute frame GIFs (all days combined) ────────────────────────────
    fivemin_configs = [
        {
            "name":   "ROTI/IPP (17 stations)",
            "folder": BASE / "ROTI_IPP_maps_17stations",
            "output": BASE / "ROTI_IPP_animation_17stations.gif",
            "resize": 800,
        },
        {
            "name":   "ROTI/IPP (~213 stations)",
            "folder": BASE / "ROTI_IPP_maps_213stations",
            "output": BASE / "ROTI_IPP_animation_213stations.gif",
            "resize": 800,
        },
    ]

    print("── 5-minute frame GIFs (all days combined) ────")
    for cfg in fivemin_configs:
        print(f"[{cfg['name']}]")
        if not cfg["folder"].exists():
            print(f"  [skip] folder not found: {cfg['folder']}")
            print()
            continue
        files = collect_5min_frames(cfg["folder"])
        if not files:
            print(f"  [skip] no 5-min frames found under {cfg['folder']}")
            print()
            continue
        print(f"  Total frames across all days: {len(files)}")
        make_gif_from_files(
            files, cfg["output"],
            duration_ms=FRAME_DURATION_5MIN_MS,
            loop=LOOP,
            resize_width=cfg.get("resize"),
        )
        print()

    print("Done.")


if __name__ == "__main__":
    main()
