# =============================================================================
# make_gif.py — Convert TEC map PNGs to animated GIF
# Works for TPS, IDW, Kriging, and SCHA map folders
# Uses only Pillow (already in conda base)
#
# Usage:
#   conda activate base
#   python make_gif.py
# =============================================================================

from pathlib import Path
from PIL import Image
import glob

BASE = Path("/Users/ziraa/Downloads/GISTDATA/TEC_output")

# ── edit these settings ────────────────────────────────────────────────────────
CONFIGS = [
    {
        "name":    "TPS",
        "folder":  BASE / "TEC_maps",
        "pattern": "VTEC_map_2026-01-*.png",
        "output":  BASE / "VTEC_animation_TPS.gif",
    },
    {
        "name":    "IDW",
        "folder":  BASE / "TEC_maps_IDW",
        "pattern": "VTEC_map_IDW_2026-01-*.png",
        "output":  BASE / "VTEC_animation_IDW.gif",
    },
    {
        "name":    "Kriging",
        "folder":  BASE / "TEC_maps_Kriging",
        "pattern": "VTEC_map_Kriging_2026-01-*.png",
        "output":  BASE / "VTEC_animation_Kriging.gif",
    },
    {
        "name":    "SCHA",
        "folder":  BASE / "TEC_maps_SCHA",
        "pattern": "VTEC_map_SCHA_2026-01-*.png",
        "output":  BASE / "VTEC_animation_SCHA.gif",
    },
]

FRAME_DURATION_MS = 800   # milliseconds per frame (800 = 1.25 fps — good for daily maps)
LOOP              = 0     # 0 = loop forever, 1 = play once
RESIZE_WIDTH      = None  # set e.g. 1200 to shrink large PNGs; None = keep original

# ── helper ─────────────────────────────────────────────────────────────────────

def make_gif(folder: Path, pattern: str, output: Path,
             duration_ms: int = 800, loop: int = 0,
             resize_width: int | None = None):

    files = sorted(folder.glob(pattern))
    if not files:
        print(f"  [skip] no files matching {folder/pattern}")
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

    # save as GIF (RGBA → P mode for GIF compatibility)
    frames_p = [f.convert("RGB").convert("P", palette=Image.ADAPTIVE, colors=256)
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

# ── main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"GIF settings: {FRAME_DURATION_MS}ms/frame, "
          f"{'loop forever' if LOOP==0 else 'play once'}")
    print()

    for cfg in CONFIGS:
        print(f"[{cfg['name']}]")
        if not cfg["folder"].exists():
            print(f"  [skip] folder not found: {cfg['folder']}")
            continue
        make_gif(
            cfg["folder"], cfg["pattern"], cfg["output"],
            duration_ms=FRAME_DURATION_MS,
            loop=LOOP,
            resize_width=RESIZE_WIDTH,
        )
        print()

    print("Done.")


if __name__ == "__main__":
    main()