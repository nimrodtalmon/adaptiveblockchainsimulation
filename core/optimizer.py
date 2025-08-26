# core/optimizer.py

import nevergrad as ng
from core.model_basic import define_problem


def solve_model(instance, budget=1000, verbose=True):
    """
    Solves the model defined in model_basic.py using Nevergrad (NGOpt solver).
    
    Args:
        instance: instance
        budget: number of function evaluations
        verbose: whether to print intermediate output

    Returns:
        best_solution: dict with 'app_assignments' and 'op_assignments'
        utilities: dict with AppUtil, OpUtil, SysUtil
        score: total utility value
    """
    # Build search space and objective
    parametrization, evaluate = define_problem(instance)

    # Set up Nevergrad optimizer
    optimizer = ng.optimizers.NGOpt(parametrization=parametrization, budget=budget)
    
    # Run optimization
    recommendation = optimizer.minimize(evaluate)

    # Extract best assignment
    best_kwargs = recommendation.kwargs
    best_app_assignment = best_kwargs["app_assignments"]
    best_op_assignment = best_kwargs["op_assignments"]

    # Recompute utilities (positive value this time)
    from core.model_basic import _evaluate_utilities
    score = _evaluate_utilities(best_app_assignment, best_op_assignment, instance)
    
    return {
        "app_assignments": best_app_assignment,
        "op_assignments": best_op_assignment
    }, score
