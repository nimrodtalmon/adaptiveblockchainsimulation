# main.py
from config import GeneralConfig, OneInstanceConfig; general_config = GeneralConfig(); one_instance_config = OneInstanceConfig
import random, numpy as np, os, argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

from core import instance_generator
from utils.helpers import clear_screen, print_instance, print_solution_with_utilities_and_constraints, take_care_of_random_seed, find_steady_state
from core.optimizer import solve_model


# --- Simplex grid of governance weights (apps, ops, sys) ---
def make_simplex_grid(K: int = 13, limit: int = 100):
    """
    Create ~uniform barycentric lattice points on the 2-simplex:
    points are (i/K, j/K, k/K) with i+j+k = K.
    K=13 gives 105 points; we take the first `limit` of them.
    Ensures the three vertices appear first.
    Returns a list of dicts with keys 'apps','ops','sys'.
    """
    pts = []
    for i in range(K + 1):
        for j in range(K + 1 - i):
            k = K - i - j
            a, o, s = i / K, j / K, k / K
            pts.append((a, o, s))

    # Put vertices first, then the rest
    vertices = {(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)}
    verts = [p for p in pts if p in vertices]
    rest  = [p for p in pts if p not in vertices]

    ordered = verts + rest
    if limit is not None:
        ordered = ordered[:limit]

    return [{'apps': a, 'ops': o, 'sys': s} for (a, o, s) in ordered]



def do_one_run_return(lambdas_override=None):
    """
    Runs one simulation and returns a tuple:
        (num_apps, num_ops, steady_state)
    No file I/O here to keep parallel writes safe.
    """
    take_care_of_random_seed()
    instance = instance_generator.generate_instance(general_config)

    # --- OVERRIDE lambdas if provided ---
    if lambdas_override is not None:
        instance['lambdas'] = {
            'apps': float(lambdas_override['apps']),
            'ops':  float(lambdas_override['ops']),
            'sys':  float(lambdas_override['sys']),
        }
    solution, _, _, loss_hist, totvio_hist = solve_model(instance)

    # print('losssssss:', loss_hist)
    # print('totviooooo:', totvio_hist)

    steady_state = find_steady_state(
        loss_hist, totvio_hist,
        window=150, loss_std_tol=5e-3, mean_delta_tol=1e-3, viol_eps=1e-12
    )
    # print("Steady-state budget:", steady_state)

    num_apps = general_config.num_apps
    num_ops  = general_config.num_ops
    lambdaapp = instance['lambdas']['apps']
    lambdaop  = instance['lambdas']['ops']
    lambdsys  = instance['lambdas']['sys']
    apputil  = sum(solution['utilities'][1]) / len(solution['utilities'][1]) if len(solution['utilities'][1])>0 else 0
    oputil   = sum(solution['utilities'][2]) / len(solution['utilities'][2]) if len(solution['utilities'][2])>0 else 0
    sysutil  = solution['utilities'][3]
    return (num_apps, num_ops, steady_state, lambdaapp, lambdaop, lambdsys, apputil, oputil, sysutil)


def _append_lines(lines, path="logs/simplex.txt"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        for a, o, s, la, lo, ls, apputil, oputil, sysutil in lines:
            f.write(f"{a},{o},{s},{la},{lo},{ls},{apputil},{oputil},{sysutil}\n")
            f.flush()
    print(f"Logged {len(lines)} line(s) -> {path}")


def main():
    parser = argparse.ArgumentParser(description="Run one or many simulations (optionally in parallel).")
    parser.add_argument("--runs", type=int, default=1, help="How many runs to execute.")
    parser.add_argument("--workers", type=int, default=1, help="How many parallel workers to use.")
    parser.add_argument("--log", type=str, default="logs/simplex.txt", help="Output log file.")
    args = parser.parse_args()

    # Build a fixed list of lambda triplets (includes vertices) to sweep over
    lambda_grid = make_simplex_grid(K=13, limit=100)  # ~uniform; 100 points incl. (1,0,0),(0,1,0),(0,0,1)

    runs = max(1, args.runs)
    workers = max(1, args.workers)
    
    # OVERRIDE COMMAND LINE
    # runs_and_workers = 100
    # runs = runs_and_workers
    # workers = runs_and_workers

    if workers == 1:
        # Sequential
        lines = []
        for i in range(runs):
            lambdas_override = lambda_grid[i % len(lambda_grid)]
            lines.append(do_one_run_return(lambdas_override=lambdas_override))
        _append_lines(lines, args.log)
    else:
        # Parallel
        lines = []
        selected = [lambda_grid[i % len(lambda_grid)] for i in range(runs)]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(do_one_run_return, lambdas_override=lam) for lam in selected]
            for fut in as_completed(futures):
                try:
                    lines.append(fut.result())
                except Exception as e:
                    print("A run failed:", repr(e))
        if lines:
            _append_lines(lines, args.log)


if __name__ == "__main__":
    main()
