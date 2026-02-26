"""
Visualization for adaptive multichain optimization.

Generates ternary plots showing utility trade-offs across governance weights.
"""

import math
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from model import Instance, Lambdas


def make_simplex_grid(resolution: int = 10) -> List[Lambdas]:
    """
    Generate a grid of points on the 2-simplex.
    
    Args:
        resolution: Number of divisions per edge (higher = finer grid)
    
    Returns:
        List of Lambdas objects covering the simplex
    """
    points = []
    for i in range(resolution + 1):
        for j in range(resolution + 1 - i):
            k = resolution - i - j
            lam_app = i / resolution
            lam_op = j / resolution
            lam_sys = k / resolution
            points.append(Lambdas(apps=lam_app, ops=lam_op, sys=lam_sys))
    return points


def barycentric_to_cartesian(lam_app: float, lam_op: float, lam_sys: float) -> Tuple[float, float]:
    """
    Convert barycentric coordinates to 2D Cartesian for plotting.
    
    Triangle vertices:
    - Apps (1,0,0) → bottom-left (0, 0)
    - Ops  (0,1,0) → bottom-right (1, 0)  
    - Sys  (0,0,1) → top (0.5, sqrt(3)/2)
    """
    h = math.sqrt(3) / 2.0
    x = lam_op * 1.0 + lam_sys * 0.5
    y = lam_sys * h
    return x, y


def plot_simplex(
    results: List[Tuple[Lambdas, float, float, float]],
    output_path: Optional[str] = None,
    title: str = "Utility Trade-offs Across Governance Weights",
    figsize: Tuple[int, int] = (15, 5),
    dpi: int = 150,
) -> None:
    """
    Plot ternary heatmaps for app, op, and system utilities.
    
    Args:
        results: List of (lambdas, avg_app_util, avg_op_util, sys_util)
        output_path: Save path (if None, displays interactively)
        title: Figure title
        figsize: Figure size in inches
        dpi: Resolution
    """
    # Extract data
    lam_apps = [r[0].apps for r in results]
    lam_ops = [r[0].ops for r in results]
    lam_sys = [r[0].sys for r in results]
    
    util_app = [r[1] for r in results]
    util_op = [r[2] for r in results]
    util_sys = [r[3] for r in results]
    
    # Convert to Cartesian
    coords = [barycentric_to_cartesian(a, o, s) for a, o, s in zip(lam_apps, lam_ops, lam_sys)]
    x = np.array([c[0] for c in coords])
    y = np.array([c[1] for c in coords])
    
    # Triangulation
    triang = mtri.Triangulation(x, y)
    
    # Plot
    h = math.sqrt(3) / 2.0
    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi)
    
    titles = ["App Utility", "Op Utility", "System Utility"]
    utils = [util_app, util_op, util_sys]
    
    for ax, panel_title, z in zip(axes, titles, utils):
        z = np.array(z)
        
        tcf = ax.tricontourf(triang, z, levels=16, cmap="viridis")
        
        # Draw triangle border
        ax.plot([0, 1], [0, 0], color="black", lw=1)
        ax.plot([0, 0.5], [0, h], color="black", lw=1)
        ax.plot([1, 0.5], [0, h], color="black", lw=1)
        
        # Show sample points
        ax.scatter(x, y, s=3, alpha=0.3, color="black")
        
        ax.set_aspect("equal")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, h + 0.05)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(panel_title, fontsize=11)
        
        # Vertex labels
        ax.text(0.0, -0.03, "Apps", ha="left", va="top", fontsize=9)
        ax.text(1.0, -0.03, "Ops", ha="right", va="top", fontsize=9)
        ax.text(0.5, h + 0.02, "System", ha="center", va="bottom", fontsize=9)
        
        cbar = fig.colorbar(tcf, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Utility", fontsize=9)
    
    fig.suptitle(title, fontsize=13, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if output_path:
        fig.savefig(output_path)
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close(fig)