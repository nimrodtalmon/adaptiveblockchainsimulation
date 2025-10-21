# main.py
from config import GeneralConfig, OneInstanceConfig
general_config = GeneralConfig()          # kept for compatibility if other modules expect it
one_instance_config = OneInstanceConfig   # ditto

import os, argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

from core import instance_generator
from utils.helpers import find_steady_state, take_care_of_random_seed
from core.optimizer import solve_model


def do_one_run_return(no_apps: int, no_ops: int):
    """
    Run one simulation for (no_apps, no_ops).
    Returns: (no_apps, no_ops, steady_state)
    """
    take_care_of_random_seed()

    # fresh config per run (process-safe)
    general_config_2 = GeneralConfig()
    general_config_2.num_apps = no_apps
    general_config_2.num_ops = no_ops

    instance = instance_generator.generate_instance(general_config_2)
    solution, utilities, constraints, loss_hist, totvio_hist = solve_model(instance)

    steady_state = find_steady_state(
        loss_hist, totvio_hist,
        window=150, loss_std_tol=5e-3, mean_delta_tol=1e-3, viol_eps=1e-12
    )

    return (no_apps, no_ops, steady_state)


def _append_lines(lines, path="logs/steady_state.txt"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        for a, o, s in lines:
            f.write(f"{a},{o},{s}\n")
            f.flush()
    print(f"Logged {len(lines)} line(s) -> {path}")


def main():
    parser = argparse.ArgumentParser(description="Run one or many simulations (optionally in parallel).")
    parser.add_argument("--runs", type=int, default=1, help="How many runs to execute.")
    parser.add_argument("--workers", type=int, default=1, help="How many parallel workers to use.")
    parser.add_argument("--log", type=str, default="logs/steady_state.txt", help="Output log file.")
    args = parser.parse_args()

    runs = max(1, args.runs)
    workers = max(1, args.workers)

    # # OVERRIDE COMMAND LINE (your original behavior)
    # runs_and_workers = 100
    # runs = runs_and_workers
    # workers = runs_and_workers

    configurations = [
        (10, 3),
        (50, 11),
        (100, 21),
        (500, 101),
        (1000, 201),
        (2000, 401),
        (5000, 1001)
    ]

    for no_apps, no_ops in configurations:
        print(f"\n=== Running config (apps={no_apps}, ops={no_ops}) with runs={runs}, workers={workers} ===")
        if workers == 1:
            # Sequential
            lines = []
            for _ in range(runs):
                lines.append(do_one_run_return(no_apps=no_apps, no_ops=no_ops))
            _append_lines(lines, args.log)
        else:
            # Parallel
            lines = []
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futures = [ex.submit(do_one_run_return, no_apps, no_ops) for _ in range(runs)]
                for fut in as_completed(futures):
                    try:
                        lines.append(fut.result())
                    except Exception as e:
                        print("A run failed:", repr(e))
            if lines:
                _append_lines(lines, args.log)


if __name__ == "__main__":
    main()
