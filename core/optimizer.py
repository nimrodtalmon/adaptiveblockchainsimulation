# core/optimizer.py

import nevergrad as ng
from core.model_basic import define_problem
from config import SimConfig; config = SimConfig()


def solve_model(instance, budget=1000, verbose=True):
    """
    Solves the model defined in model_basic.py using Nevergrad (NGOpt solver).
    
    Args:
        instance: instance
        budget: number of function evaluations
        verbose: whether to print intermediate output

    Returns:
        best_solution: dict with 'app_assignments' and 'op_assignments'
        score: total utility value
    """
    # Build search space and objective
    parametrization, evaluate = define_problem(instance)

    # Set up Nevergrad optimizer
    optimizer = config.nevergrad_optimizer(
        parametrization=parametrization, 
        budget=budget,
        num_workers=config.nevergrad_num_workers)

    if verbose:
        print("\n=== Solver ===")
        print(f"[Nevergrad] {config.nevergrad_optimizer.__name__} | "
              f"budget={config.nevergrad_budget} | num_workers={config.nevergrad_num_workers}")

    # Run optimization
    recommendation = optimizer.minimize(evaluate)

    # Extract best assignment
    best_kwargs = recommendation.kwargs
    best_app_assignment = best_kwargs["app_assignments"]
    best_op_assignment = best_kwargs["op_assignments"]

    # Recompute utilities (positive value this time)
    from core.model_basic import evaluate_utilities
    score = evaluate_utilities(best_app_assignment, best_op_assignment, instance)
    
    return {
        "app_assignments": best_app_assignment,
        "op_assignments": best_op_assignment
    }, score
