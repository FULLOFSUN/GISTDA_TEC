#!/usr/bin/env python3
"""
plot_loocv_table.py

Summary statistics table companion to plot_loocv_scatter.py -- reads the
SAME cross_validation_errors.csv and computes the same per-method
statistics shown in the scatter panels (ME, RMSE, r), plus MAE and CI95%
for a fuller picture.

Usage:
    conda activate base
    python plot_loocv_table.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import RdYlGn

# =============================================================================
# PATHS -- same source as plot_loocv_scatter.py
# =============================================================================
TEC_OUTPUT_DIR = Path("/Users/ziraa/Documents/GISTDA/GISTDA_TEC/RINEXTEST3(20 JUN)/TEC_output")
CSV_PATH = TEC_OUTPUT_DIR / "cross_validation_errors.csv"
OUT_DIR  = TEC_OUTPUT_DIR

# =============================================================================
# METHOD CONFIG -- same colours as plot_loocv_scatter.py for visual
# consistency between the two figures
# =============================================================================
METHODS = {
    "IDW":     {"col": "pred_idw",  "color": "#1A5276"},
    "TPS/RBF": {"col": "pred_tps",  "color": "#6C3483"},
    "Kriging": {"col": "pred_krig", "color": "#922B21"},
}

# =============================================================================
# STATISTICS
# =============================================================================

def compute_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, cfg in METHODS.items():
        col = cfg["col"]
        sub = df.dropna(subset=[col, "obs"])
        errors = sub[col].values - sub["obs"].values
        n = len(errors)

        me   = errors.mean()
        mae  = np.abs(errors).mean()
        rmse = np.sqrt((errors**2).mean())
        r    = np.corrcoef(sub["obs"].values, sub[col].values)[0, 1]
        std_e = errors.std()

        within = np.abs(errors) <= 1.96 * std_e
        ci95_n = int(within.sum())
        ci95_p = 100.0 * ci95_n / n if n else np.nan

        # RMSSE (Kriging only, if krig_var column is present)
        if method == "Kriging" and "krig_var" in df.columns:
            kv_sub = df.dropna(subset=[col, "krig_var"])
            if len(kv_sub) > 0:
                std_errors = (kv_sub[col].values - kv_sub["obs"].values) / \
                             np.sqrt(np.maximum(kv_sub["krig_var"].values, 1e-6))
                rmsse = float(np.sqrt((std_errors**2).mean()))
            else:
                rmsse = np.nan
        else:
            rmsse = np.nan

        rows.append({
            "Method": method, "N": n,
            "ME": round(me, 3), "MAE": round(mae, 3),
            "RMSE": round(rmse, 3), "r": round(r, 3),
            "RMSSE": round(rmsse, 3) if np.isfinite(rmsse) else "\u2014",
            "CI95_N": ci95_n, "CI95_%": round(ci95_p, 1),
        })
    return pd.DataFrame(rows)

# =============================================================================
# PLOT
# =============================================================================

def plot_table(stats: pd.DataFrame, out_path: str):
    display_cols = ["Method", "N", "ME", "MAE", "RMSE", "r", "RMSSE", "CI95_N", "CI95_%"]
    header = ["Method", "N", "ME\n(TECU)", "MAE\n(TECU)", "RMSE\n(TECU)",
             "r", "RMSSE", "CI95 N", "CI95 %"]

    cell_text = [[str(row[c]) for c in display_cols] for _, row in stats.iterrows()]

    fig, ax = plt.subplots(figsize=(11, 3.2))
    fig.patch.set_facecolor("white")
    ax.axis("off")

    tbl = ax.table(cellText=cell_text, colLabels=header,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.05, 2.0)

    col_idx = {c: i for i, c in enumerate(display_cols)}
    n_methods = len(stats)

    def rank_color(values, higher_is_better=True):
        vals_num = pd.to_numeric(pd.Series(values), errors="coerce")
        if vals_num.isna().all():
            return ["#ffffff"] * len(values)
        norm = Normalize(vmin=vals_num.min(), vmax=vals_num.max())
        cmap = RdYlGn if higher_is_better else RdYlGn.reversed()
        return [matplotlib.colors.to_hex(cmap(norm(v)))
                if np.isfinite(v) else "#eeeeee" for v in vals_num]

    r_colors    = rank_color(stats["r"].values,                higher_is_better=True)
    rmse_colors = rank_color(stats["RMSE"].values,              higher_is_better=False)
    mae_colors  = rank_color(stats["MAE"].values,                higher_is_better=False)
    me_colors   = rank_color(np.abs(stats["ME"].values),        higher_is_better=False)
    ci95_colors = rank_color(stats["CI95_%"].values,             higher_is_better=True)

    for row_i in range(n_methods):
        tbl[(row_i+1, col_idx["ME"])].set_facecolor(me_colors[row_i])
        tbl[(row_i+1, col_idx["MAE"])].set_facecolor(mae_colors[row_i])
        tbl[(row_i+1, col_idx["RMSE"])].set_facecolor(rmse_colors[row_i])
        tbl[(row_i+1, col_idx["r"])].set_facecolor(r_colors[row_i])
        tbl[(row_i+1, col_idx["CI95_%"])].set_facecolor(ci95_colors[row_i])

    for col_i in range(len(display_cols)):
        tbl[(0, col_i)].set_facecolor("#21295C")
        tbl[(0, col_i)].set_text_props(color="white", fontweight="bold")

    for row_i, (method, cfg) in enumerate(METHODS.items(), start=1):
        tbl[(row_i, 0)].set_text_props(fontweight="bold", color=cfg["color"])

    ax.set_title(
        "LOOCV Summary Statistics \u2014 17 Thai CORS stations \u00b7 DOY 014\u2013030 \u00b7 January 2026\n"
        "Companion table to LOOCV_scatter.png  \u00b7  Colour: green = best, red = worst (per column)",
        fontsize=11, fontweight="bold", pad=14)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    if not CSV_PATH.exists():
        print(f"CSV not found: {CSV_PATH}")
        print("Run cross_validation.py first to generate the errors CSV.")
        return

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df):,} LOOCV samples")

    stats = compute_stats(df)
    print(stats.to_string(index=False))

    csv_out = OUT_DIR / "loocv_summary_stats.csv"
    stats.to_csv(csv_out, index=False)
    print(f"\nStats CSV saved -> {csv_out}")

    plot_table(stats, str(OUT_DIR / "loocv_summary_table.png"))


if __name__ == "__main__":
    main()