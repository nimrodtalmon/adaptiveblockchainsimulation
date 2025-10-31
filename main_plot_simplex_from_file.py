# ternary_three_panel_avg.py
# Input: CSV with rows:
# numapps,numops,numchains,lam_app,lam_op,lam_sys,util_app,util_op,util_sys
# Multiple rows may share the same (lam_app, lam_op, lam_sys) -> we average.
# Output: plots/simplex.png — 3 side-by-side ternary heatmaps (app/op/sys) of MEAN utilities.

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

# --------------------------- config ---------------------------
INPUT_PATH  = "logs/simplex.txt"   # change if needed
OUTPUT_PATH = "plots/simplex.png"
FIGSIZE_IN  = (15, 5)              # width, height inches
DPI         = 200
ASSUME_NORMALIZED = True           # set False if utilities are not in [0,1]
ROUND_DECIMALS = 6                 # rounding before grouping to avoid float-key issues
SHOW_POINT_COUNTS = True           # print #runs per simplex point

# --------------------- load & basic checks --------------------
df = pd.read_csv(
    INPUT_PATH,
    header=None,
    names=[
        "numapps", "numops", "numchains",
        "lam_app", "lam_op", "lam_sys",
        "util_app","util_op","util_sys"
    ],
)

# (a) fix simplex sums if needed
sums = df[["lam_app","lam_op","lam_sys"]].sum(axis=1).values
if not np.allclose(sums, 1.0, atol=1e-6):
    df[["lam_app","lam_op","lam_sys"]] = df[["lam_app","lam_op","lam_sys"]].div(sums, axis=0)

# (b) round lambdas to create stable group keys
for c in ["lam_app","lam_op","lam_sys"]:
    df[c] = df[c].round(ROUND_DECIMALS)

# ------------------- aggregate by simplex point ----------------
agg = df.groupby(["lam_app","lam_op","lam_sys"], as_index=False).agg(
    util_app_mean=("util_app","mean"),
    util_op_mean =("util_op","mean"),
    util_sys_mean=("util_sys","mean"),
    n_runs       =("util_app","count"),
    # keep some metadata from first row (purely informative)
    numapps_first=("numapps","first"),
    numops_first =("numops","first"),
    numchains_first=("numchains","first"),
)

if SHOW_POINT_COUNTS:
    print("Averaged runs per simplex point (first 10 rows):")
    print(agg[["lam_app","lam_op","lam_sys","n_runs"]].head(10).to_string(index=False))
    print(f"Total unique simplex points: {len(agg)} ; total input rows: {len(df)}")

# -------------------- barycentric -> 2D -----------------------
# Vertices of an equilateral triangle:
# Apps at (0, 0), Ops at (1, 0), Sys at (0.5, sqrt(3)/2)
h = math.sqrt(3) / 2.0

lam_app = agg["lam_app"].to_numpy()
lam_op  = agg["lam_op"].to_numpy()
lam_sys = agg["lam_sys"].to_numpy()

# Point = a*A + o*O + s*S
# A=(0,0), O=(1,0), S=(0.5,h)
x = lam_op * 1.0 + lam_sys * 0.5 + lam_app * 0.0
y = lam_sys * h

# triangulate the scattered samples (works even if points are irregular)
triang = mtri.Triangulation(x, y)

# ---------------------- color scaling -------------------------
if ASSUME_NORMALIZED:
    vmin, vmax = 0.0, 1.0
else:
    all_means = np.concatenate([
        agg["util_app_mean"].to_numpy(),
        agg["util_op_mean"].to_numpy(),
        agg["util_sys_mean"].to_numpy(),
    ])
    vmin, vmax = np.nanmin(all_means), np.nanmax(all_means)

# ------------------------- plotting ---------------------------
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_IN, dpi=DPI)

titles = ["Applications' Utility (mean)", "Operators' Utility (mean)", "System Utility (mean)"]
cols   = ["util_app_mean", "util_op_mean", "util_sys_mean"]

for ax, title, col in zip(axes, titles, cols):
    z = agg[col].to_numpy()

    # filled contours (smooth look); if you prefer flat shading use tripcolor
    tcf = ax.tricontourf(triang, z, levels=16, vmin=vmin, vmax=vmax, cmap="viridis")

    # draw triangle edges
    ax.plot([0, 1], [0, 0], color="black", lw=1)            # bottom edge (Apps ↔ Ops)
    ax.plot([0, 0.5], [0, h], color="black", lw=1)          # left edge (Apps ↔ Sys)
    ax.plot([1, 0.5], [0, h], color="black", lw=1)          # right edge (Ops ↔ Sys)

    # optional: show sample points
    ax.plot(x, y, ".", ms=2, alpha=0.35, color="black")

    # tidy formatting
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, h + 0.05)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=11)

    # vertex labels
    ax.text(0.00, -0.015, "Apps (1,0,0)", ha="left",  va="top",    fontsize=9)
    ax.text(1.00, -0.015, "Ops (0,1,0)",  ha="right", va="top",    fontsize=9)
    ax.text(0.50,  h - 0.0, "System (0,0,1)", ha="center", va="bottom", fontsize=9)

    # colorbar per panel (compact)
    cbar = fig.colorbar(tcf, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean Utility", fontsize=9)

fig.suptitle("Utility trade-offs across governance weights (λ) — mean over runs", fontsize=13, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUTPUT_PATH)
print(f"Saved: {OUTPUT_PATH}")
