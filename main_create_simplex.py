#!/usr/bin/env python3
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from pathlib import Path

# -------------------------------
# CONFIG – you can edit these
# -------------------------------
NUMBER_OF_EXPERIMENTS = 3              # like your original outer script
LOG_PATH = "logs/simplex.txt"
PLOT_PATH = "plots/simplex.png"
RUNS_PER_CALL = 100                    # was --runs in the second script
WORKERS_PER_CALL = 4                   # was --workers in the second script

# --------------------------------------------------------------------
# imports from your project – these stay as-is, you still need them
# --------------------------------------------------------------------
from config import GeneralConfig, OneInstanceConfig  # noqa: F401
from core import instance_generator
from utils.helpers import (
    take_care_of_random_seed,
    find_steady_state,
)
from core.optimizer import solve_model


# ====================================================================
# PART 1: the code that used to be in main_put_simplex_lines_in_file.py
# ====================================================================

def make_simplex_grid(K: int = 13, limit: int = 100):
    """
    Create ~uniform barycentric lattice points on the 2-simplex:
    points are (i/K, j/K, k/K) with i+j+k = K.
    Ensures the three vertices appear first.
    Returns a list of dicts with keys 'apps','ops','sys'.
    """
    pts = []
    for i in range(K + 1):
        for j in range(K + 1 - i):
            k = K - i - j
            a, o, s = i / K, j / K, k / K
            pts.append((a, o, s))

    vertices = {(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)}
    verts = [p for p in pts if p in vertices]
    rest = [p for p in pts if p not in vertices]

    ordered = verts + rest
    if limit is not None:
        ordered = ordered[:limit]

    return [{'apps': a, 'ops': o, 'sys': s} for (a, o, s) in ordered]


def do_one_run_return(instance_without_lambdas, lambdas_override=None):
    """
    Runs one simulation using validation_example_5_op with different lambda combinations.
    Returns:
        (num_apps, num_ops, steady_state, lambda_app, lambda_op, lambda_sys,
         app_util, op_util, sys_util)
    """
    take_care_of_random_seed()

    # your instance generator
    instance = instance_without_lambdas()

    # Override lambdas if provided
    if lambdas_override is not None:
        instance['lambdas'] = {
            'apps': float(lambdas_override['apps']),
            'ops': float(lambdas_override['ops']),
            'sys': float(lambdas_override['sys']),
        }

    solution, _, _, loss_hist, totvio_hist = solve_model(instance)

    steady_state = find_steady_state(
        loss_hist, totvio_hist,
        window=150, loss_std_tol=5e-3, mean_delta_tol=1e-3, viol_eps=1e-12
    )

    num_apps = len(instance['apps'])
    num_ops = len(instance['ops'])
    lambdaapp = instance['lambdas']['apps']
    lambdaop = instance['lambdas']['ops']
    lambdsys = instance['lambdas']['sys']

    # utilities:
    apputil = (
        sum(solution['utilities'][1]) / len(solution['utilities'][1])
        if len(solution['utilities'][1]) > 0 else 0
    )
    oputil = (
        sum(solution['utilities'][2]) / len(solution['utilities'][2])
        if len(solution['utilities'][2]) > 0 else 0
    )
    sysutil = solution['utilities'][3]

    return (
        num_apps,
        num_ops,
        steady_state,
        lambdaapp,
        lambdaop,
        lambdsys,
        apputil,
        oputil,
        sysutil,
    )


def _append_lines(lines, path=LOG_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        for (
            num_apps, num_ops, steady_state,
            la, lo, ls, apputil, oputil, sysutil
        ) in lines:
            # keep the exact column order the plotter expects
            f.write(
                f"{num_apps},{num_ops},{steady_state},"
                f"{la},{lo},{ls},"
                f"{apputil},{oputil},{sysutil}\n"
            )
            f.flush()
    print(f"Logged {len(lines)} line(s) -> {path}")


def run_simplex_batch(instance_without_lambdas, runs: int, workers: int, log_path: str):
    """
    Runs multiple simulations across the simplex grid
    """
    # your original code used a denser grid:
    lambda_grid = make_simplex_grid(K=20, limit=231)
    runs = max(1, runs)
    workers = max(1, workers)

    if workers == 1:
        lines = []
        for i in range(runs):
            lambdas_override = lambda_grid[i % len(lambda_grid)]
            lines.append(do_one_run_return(instance_without_lambdas, lambdas_override=lambdas_override))
        _append_lines(lines, log_path)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        lines = []
        selected = [lambda_grid[i % len(lambda_grid)] for i in range(runs)]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [
                ex.submit(do_one_run_return, instance_without_lambdas, lambdas_override=lam)
                for lam in selected
            ]
            for fut in as_completed(futures):
                try:
                    lines.append(fut.result())
                except Exception as e:
                    print("A run failed:", repr(e))
        if lines:
            _append_lines(lines, log_path)


def plot_simplex_from_file(
    input_path: str = LOG_PATH,
    output_path: str = PLOT_PATH,
    assume_normalized: bool = True,
    figsize_in=(15, 5),
    dpi=200,
    round_decimals=6,
    show_point_counts=True,
):
    # load
    df = pd.read_csv(
        input_path,
        header=None,
        names=[
            "numapps", "numops", "numchains",
            "lam_app", "lam_op", "lam_sys",
            "util_app", "util_op", "util_sys",
        ],
    )

    # fix simplex sums if needed
    sums = df[["lam_app", "lam_op", "lam_sys"]].sum(axis=1).values
    if not np.allclose(sums, 1.0, atol=1e-6):
        df[["lam_app", "lam_op", "lam_sys"]] = df[["lam_app", "lam_op", "lam_sys"]].div(sums, axis=0)

    # round lambdas to create stable group keys
    for c in ["lam_app", "lam_op", "lam_sys"]:
        df[c] = df[c].round(round_decimals)

    # aggregate
    agg = df.groupby(["lam_app", "lam_op", "lam_sys"], as_index=False).agg(
        util_app_mean=("util_app", "mean"),
        util_op_mean=("util_op", "mean"),
        util_sys_mean=("util_sys", "mean"),
        n_runs=("util_app", "count"),
        numapps_first=("numapps", "first"),
        numops_first=("numops", "first"),
        numchains_first=("numchains", "first"),
    )

    if show_point_counts:
        print("Averaged runs per simplex point (first 10 rows):")
        print(agg[["lam_app", "lam_op", "lam_sys", "n_runs"]].head(10).to_string(index=False))
        print(f"Total unique simplex points: {len(agg)} ; total input rows: {len(df)}")

    # barycentric -> 2D
    h = math.sqrt(3) / 2.0
    lam_app = agg["lam_app"].to_numpy()
    lam_op = agg["lam_op"].to_numpy()
    lam_sys = agg["lam_sys"].to_numpy()

    x = lam_op * 1.0 + lam_sys * 0.5 + lam_app * 0.0
    y = lam_sys * h

    triang = mtri.Triangulation(x, y)

    # color scaling
    if assume_normalized:
        vmin, vmax = 0.0, 1.0
    else:
        all_means = np.concatenate([
            agg["util_app_mean"].to_numpy(),
            agg["util_op_mean"].to_numpy(),
            agg["util_sys_mean"].to_numpy(),
        ])
        vmin, vmax = np.nanmin(all_means), np.nanmax(all_means)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=figsize_in, dpi=dpi)

    titles = [
        "Applications' Utility (mean)",
        "Operators' Utility (mean)",
        "System Utility (mean)",
    ]
    cols = ["util_app_mean", "util_op_mean", "util_sys_mean"]

    for ax, title, col in zip(axes, titles, cols):
        z = agg[col].to_numpy()
        tcf = ax.tricontourf(
            triang,
            z,
            levels=16,
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
        )

        # draw triangle
        ax.plot([0, 1], [0, 0], color="black", lw=1)
        ax.plot([0, 0.5], [0, h], color="black", lw=1)
        ax.plot([1, 0.5], [0, h], color="black", lw=1)

        # show points
        ax.plot(x, y, ".", ms=2, alpha=0.35, color="black")

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, h + 0.05)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=11)

        ax.text(0.00, -0.015, "Apps (1,0,0)", ha="left", va="top", fontsize=9)
        ax.text(1.00, -0.015, "Ops (0,1,0)", ha="right", va="top", fontsize=9)
        ax.text(0.50, h - 0.0, "System (0,0,1)", ha="center", va="bottom", fontsize=9)

        cbar = fig.colorbar(tcf, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Mean Utility", fontsize=9)

    fig.suptitle(
        "Utility trade-offs across governance weights (λ) — mean over runs",
        fontsize=13,
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path)
    print(f"Saved: {output_path}")


def main():
    path = Path(LOG_PATH)
    if path.exists():
        path.unlink()

    # run the simplex generator several times
    for _ in range(NUMBER_OF_EXPERIMENTS):
        run_simplex_batch(
            instance_without_lambdas=instance_generator.generate_validation_example_5_op,
            runs=RUNS_PER_CALL,
            workers=WORKERS_PER_CALL,
            log_path=LOG_PATH,
        )

    # then plot
    plot_simplex_from_file(
        input_path=LOG_PATH,
        output_path=PLOT_PATH,
        assume_normalized=True,
    )

if __name__ == "__main__":
    main()
