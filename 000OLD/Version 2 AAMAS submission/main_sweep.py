# main_sweep.py
"""
Run multiple experiment settings and append ONE summary row per setting to CSV/JSONL.

Usage examples:
  python main_sweep.py
  python main_sweep.py --optimizers NGOpt,NGOpt13 --budgets 50,200,1000 --workers 1,8 \
      --sizes 5x5x5,10x10x8 --repetitions 5 --exp-name utility_vs_budget

Notes:
- Each setting runs `repetitions` independent instances and appends ONE line to logs/summary_runs.csv.
- You can also edit the CONFIG GRID block below for a no-CLI workflow.
"""

from __future__ import annotations
from dataclasses import replace
from itertools import product
from typing import Dict, Any, List, Tuple

import nevergrad as ng

# Project imports (adjust paths if your repo layout differs)
from config import GeneralConfig, ExperimentConfig
from core import instance_generator
from core.optimizer import solve_model
from utils.helpers import take_care_of_random_seed as _seed  # if your function name is different, fix here
# If your helper is named `take_care_of_random_seed`, uncomment the next line and comment the one above:
# from utils.helpers import take_care_of_random_seed as _seed
from utils.logger import append_csv, append_jsonl, timestamp_iso


# ----------------------------- CONFIG GRID (edit here for easy control) -----------------------------

# Default grid; can be overridden by CLI flags.
DEFAULT_OPTIMIZERS = ["OnePlusOne"]            # e.g., ["NGOpt", "NGOpt13", "CMA", "DE", "OnePlusOne"]
DEFAULT_BUDGETS    = [5]
DEFAULT_WORKERS    = [1]
DEFAULT_SIZES      = [(10, 10, 10)]   # (num_apps, num_ops, num_chains)

DEFAULT_REPETITIONS   = 1
DEFAULT_EXP_NAME      = "sweep_experiment"
DEFAULT_INSTANCE_GEN  = "generate_random_instance_1"  # your generator name (if relevant in your codebase)

CSV_PATH   = "logs/sweep_runs.csv"
JSONL_PATH = "logs/sweep_runs.jsonl"

CSV_HEADER = [
    "timestamp",
    "repetitions",
    "optimizer",
    "budget",
    "pct_feasible",
    "avg_util_feasible",
    "num_apps",
    "num_ops",
    "num_chains",
    "experiment_name",
    "instance_generator",
    "feasible_runs",
    "avg_util_all",
    "total_runs",
    "num_workers",
]

# ---------------------------------------- Utilities ----------------------------------------

def _resolve_optimizer(name: str):
    """Map string to Nevergrad optimizer class; extend as needed."""
    table = {
        "NGOpt": ng.optimizers.NGOpt,
        "NGOpt13": ng.optimizers.NGOpt13,
        "CMA": ng.optimizers.CMA,
        "DE": ng.optimizers.DE,
        "OnePlusOne": ng.optimization.optimizerlib.ParametrizedOnePlusOne,
        # add more mappings if you want
    }
    if name in table:
        return table[name]
    # fallback: try getattr on ng.optimizers (expert mode)
    return getattr(ng.optimizers, name, ng.optimizers.NGOpt)


def _total_violation_magnitude(viol) -> float:
    """Sum absolute values across possibly-nested violation structures."""
    def _sum_abs(x):
        if isinstance(x, (list, tuple)):
            return sum(_sum_abs(y) for y in x)
        try:
            return abs(float(x))
        except Exception:
            return 0.0
    return _sum_abs(viol)


def _run_one_setting(gcfg: GeneralConfig, ecfg: ExperimentConfig) -> Dict[str, Any]:
    """
    Run `ecfg.repetitions` instances for the given general config, collect stats, and return a summary row.
    Appends nothing to CSV here; the caller handles logging to keep responsibilities clear.
    """
    # optional: seed management once per setting
    try:
        _seed()
    except Exception:
        pass

    results: Dict[int, Dict[str, Any]] = {}

    for i in range(ecfg.repetitions):
        # Generate instance (your generator probably ignores name and uses ecfg directly)
        instance = instance_generator.generate_instance(ecfg)

        # Solve model
        solution, utilities, violations = solve_model(instance)

        # Store
        results[i] = {
            "utilities": utilities,
            "violations": violations,
            "solution": solution,
        }

    # Stats
    total_runs = len(results)
    feasible_runs = [i for i, r in results.items() if _total_violation_magnitude(r["violations"]) == 0.0]
    pct_feasible = (100.0 * len(feasible_runs) / total_runs) if total_runs else 0.0

    # Assume utilities is a scalar. If it's a dict, adapt here (e.g., utilities["total"]).
    all_utils = [r["utilities"] for r in results.values()]
    avg_util_all = (sum(all_utils) / len(all_utils)) if all_utils else None

    feas_utils = [results[i]["utilities"] for i in feasible_runs]
    avg_util_feasible = (sum(feas_utils) / len(feas_utils)) if feas_utils else None

    # Summary row
    opt_name = getattr(gcfg.nevergrad_optimizer, "__name__", str(gcfg.nevergrad_optimizer))
    row = {
        "timestamp": timestamp_iso(),
        "repetitions": ecfg.repetitions,
        "optimizer": opt_name,
        "budget": gcfg.nevergrad_budget,
        "pct_feasible": round(pct_feasible, 2),
        "avg_util_feasible": (None if avg_util_feasible is None else float(avg_util_feasible)),
        "num_apps": gcfg.num_apps,
        "num_ops": gcfg.num_ops,
        "num_chains": gcfg.num_chains,
        "experiment_name": ecfg.name,
        "instance_generator": gcfg.instance_generator,
        "feasible_runs": len(feasible_runs),
        "avg_util_all": (None if avg_util_all is None else float(avg_util_all)),
        "total_runs": total_runs,
        "num_workers": gcfg.nevergrad_num_workers,
    }
    return row


def _parse_cli():
    import argparse
    p = argparse.ArgumentParser(description="Run a sweep of settings; append one CSV row per setting.")
    p.add_argument("--optimizers", type=str, default=",".join(DEFAULT_OPTIMIZERS),
                   help="Comma-separated optimizer names, e.g., NGOpt,NGOpt13,CMA")
    p.add_argument("--budgets", type=str, default=",".join(map(str, DEFAULT_BUDGETS)),
                   help="Comma-separated budgets, e.g., 50,200,1000")
    p.add_argument("--workers", type=str, default=",".join(map(str, DEFAULT_WORKERS)),
                   help="Comma-separated worker counts, e.g., 1,8")
    p.add_argument("--sizes", type=str, default=",".join(f"{a}x{o}x{c}" for (a, o, c) in DEFAULT_SIZES),
                   help="Comma-separated sizes as AxOxC, e.g., 5x5x5,10x10x8")
    p.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS,
                   help="Repetitions per setting")
    p.add_argument("--exp-name", type=str, default=DEFAULT_EXP_NAME,
                   help="Experiment name for the CSV")
    p.add_argument("--instance-generator", type=str, default=DEFAULT_INSTANCE_GEN,
                   help="Instance generator identifier (if used in your codebase)")
    return p.parse_args()


def _parse_sizes(s: str) -> List[Tuple[int, int, int]]:
    sizes = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            a, o, c = part.lower().replace(" ", "").split("x")
            sizes.append((int(a), int(o), int(c)))
        except Exception:
            raise ValueError(f"Bad size format: '{part}'. Expected AxOxC, e.g., 10x10x8")
    return sizes


def main():
    args = _parse_cli()

    optimizers = [x.strip() for x in args.optimizers.split(",") if x.strip()]
    budgets    = [int(x) for x in args.budgets.split(",") if x.strip()]
    workers    = [int(x) for x in args.workers.split(",") if x.strip()]
    sizes      = _parse_sizes(args.sizes)

    # Base configs
    base_g = GeneralConfig()
    base_g.instance_generator = args.instance_generator

    base_e = ExperimentConfig()
    base_e.repetitions = args.repetitions
    base_e.name = args.exp_name

    # Sweep
    for opt_name, B, W, (A, O, C) in product(optimizers, budgets, workers, sizes):
        g = replace(base_g)  # new copy
        e = replace(base_e)  # new copy

        g.nevergrad_optimizer = _resolve_optimizer(opt_name)
        g.nevergrad_budget    = B
        g.nevergrad_num_workers = W
        g.num_apps, g.num_ops, g.num_chains = A, O, C

        print("=" * 90)
        print(f"SETTING: opt={opt_name}  budget={B}  workers={W}  size=({A},{O},{C})  reps={e.repetitions}")
        print("=" * 90)

        row = _run_one_setting(g, e)

        append_csv(CSV_PATH, row, header_order=CSV_HEADER)
        append_jsonl(JSONL_PATH, row)
        print(f"Appended row -> {CSV_PATH}")
        print(row)

    print("\nAll done.")


if __name__ == "__main__":
    main()