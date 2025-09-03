# core/optimizer.py

import nevergrad as ng
from core.model_basic import define_problem
from config import GeneralConfig; general_config = GeneralConfig()


def solve_model(instance, verbose=True):
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
    parametrization, evaluate, constraints = define_problem(instance)

    # Set up Nevergrad optimizer
    opt = general_config.nevergrad_optimizer(
        parametrization=parametrization, 
        budget=general_config.nevergrad_budget,
        num_workers=general_config.nevergrad_num_workers)

    if verbose:
        print("\n=== Solver ===")
        print(f"[Nevergrad] {general_config.nevergrad_optimizer.__name__} | "
              f"budget={general_config.nevergrad_budget} | num_workers={general_config.nevergrad_num_workers}")

    # Run optimization
    # recommendation = optimizer.minimize(evaluate, constraints)
    for _ in range(opt.budget):
        cand = opt.ask()
        value = evaluate(**cand.kwargs)
        violations = constraints(**cand.kwargs)
        opt.tell(cand, value, violations) 
    recommendation = opt.provide_recommendation()

    # Extract best assignment
    best_kwargs = recommendation.kwargs
    best_app_assignment = best_kwargs["app_assignments"]
    best_op_assignment = best_kwargs["op_assignments"]
    best_fee2gas_chains = best_kwargs["fee2gas_chains"]

    # Recompute utilities (positive value this time)
    from core.model_basic import evaluate_utilities
    score = evaluate_utilities(
        best_app_assignment, 
        best_op_assignment, 
        best_fee2gas_chains, 
        instance)
    
    return {
        "app_assignments": best_app_assignment,
        "op_assignments": best_op_assignment,
        "fee2gas_chains": best_fee2gas_chains
    }, score
