"""
Main entry point for adaptive multichain optimization.

Examples:
    # Single run (must supply lambdas)
    python main.py --mode single --instance toy --lam_app 0.33 --lam_op 0.33 --lam_sys 0.34
    
    # Simplex sweep
    python main.py --mode simplex --instance simplex --resolution 10 --budget 50
"""

import argparse
import os

from model import Lambdas
from instances import simplex_gadget, toy_instance, random_instance
from optimizer import solve
from visualize import make_simplex_grid, plot_simplex


# Registry of instance generators
INSTANCES = {
    "toy": lambda args: toy_instance(),
    "simplex": lambda args: simplex_gadget(kappa=args.kappa, sigma=args.sigma)
}


def get_instance(args):
    """Get instance from registry based on args."""
    if args.instance not in INSTANCES:
        raise ValueError(f"Unknown instance: {args.instance}. Choose from: {list(INSTANCES.keys())}")
    return INSTANCES[args.instance](args)


def run_single(args):
    """Run a single optimization with specified lambdas."""
    # Validate lambdas are provided
    if args.lam_app is None or args.lam_op is None or args.lam_sys is None:
        raise ValueError("Mode 'single' requires --lam_app, --lam_op, --lam_sys")
    
    print("=== Single Run ===")
    
    instance = get_instance(args)
    lambdas = Lambdas(apps=args.lam_app, ops=args.lam_op, sys=args.lam_sys)
    
    print(f"Instance: {args.instance} ({instance.n_apps} apps, {instance.n_ops} ops)")
    print(f"Lambdas: apps={lambdas.apps}, ops={lambdas.ops}, sys={lambdas.sys}")
    print(f"Budget: {args.budget}")
    print()
    
    solution, _ = solve(instance, lambdas, budget=args.budget, verbose=True)
    
    print()
    print("=== Result ===")
    print(f"Feasible: {solution.feasible}")
    print(f"App assignments: {solution.app_assignments}")
    print(f"Op assignments:  {solution.op_assignments}")
    print()
    print(f"App utilities: {[f'{u:.4f}' for u in solution.app_utilities]}")
    print(f"Op utilities:  {[f'{u:.4f}' for u in solution.op_utilities]}")
    print(f"Sys utility:   {solution.sys_utility:.4f}")
    print(f"Total utility: {solution.total_utility:.4f}")
    
    for c, chain in solution.chains.items():
        print(f"Chain {c}: price={chain.price}, gas={chain.gas}, feasible={chain.feasible}")


def run_simplex(args):
    """Run simplex sweep and generate plot."""
    print("=== Simplex Sweep ===")
    
    instance = get_instance(args)
    grid = make_simplex_grid(resolution=args.resolution)
    
    print(f"Instance: {args.instance} ({instance.n_apps} apps, {instance.n_ops} ops)")
    print(f"Grid: {len(grid)} points (resolution={args.resolution})")
    print(f"Budget per point: {args.budget}")
    print()
    
    results = []
    for i, lambdas in enumerate(grid):
        solution, _ = solve(instance, lambdas, budget=args.budget, verbose=False)
        results.append((
            lambdas,
            solution.avg_app_utility,
            solution.avg_op_utility,
            solution.sys_utility,
        ))
        
        if (i + 1) % 10 == 0 or i == len(grid) - 1:
            print(f"  [{i+1}/{len(grid)}] completed")
    
    os.makedirs("plots", exist_ok=True)
    output_path = f"plots/{args.output}"
    
    print()
    plot_simplex(results, output_path=output_path)


def main():
    parser = argparse.ArgumentParser(description="Adaptive Multichain Optimization")
    
    # Mode
    parser.add_argument("--mode", type=str, required=True,
                        choices=["single", "simplex"],
                        help="Run mode: 'single' or 'simplex'")
    
    # Instance
    parser.add_argument("--instance", type=str, default="toy",
                        choices=list(INSTANCES.keys()),
                        help="Instance type")
    
    # Simplex gadget params
    parser.add_argument("--kappa", type=float, default=0.3,
                        help="Price spread (simplex gadget)")
    parser.add_argument("--sigma", type=float, default=0.9,
                        help="Stake skew (simplex gadget)")
    
    # Random instance params
    parser.add_argument("--n_apps", type=int, default=5,
                        help="Number of apps (random instance)")
    parser.add_argument("--n_ops", type=int, default=3,
                        help="Number of ops (random instance)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    
    # Optimizer params
    parser.add_argument("--budget", type=int, default=50,
                        help="Optimizer budget")
    
    # Single mode params
    parser.add_argument("--lam_app", type=float, default=None,
                        help="Lambda for apps (required for single mode)")
    parser.add_argument("--lam_op", type=float, default=None,
                        help="Lambda for ops (required for single mode)")
    parser.add_argument("--lam_sys", type=float, default=None,
                        help="Lambda for system (required for single mode)")
    
    # Simplex mode params
    parser.add_argument("--resolution", type=int, default=10,
                        help="Simplex grid resolution")
    parser.add_argument("--output", type=str, default="simplex.png",
                        help="Output filename (in plots/)")
    
    args = parser.parse_args()
    
    if args.mode == "single":
        run_single(args)
    elif args.mode == "simplex":
        run_simplex(args)


if __name__ == "__main__":
    main()