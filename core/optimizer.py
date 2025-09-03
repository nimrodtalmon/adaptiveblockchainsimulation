# core/optimizer.py

import nevergrad as ng
from core.model_basic import define_problem
from config import GeneralConfig; general_config = GeneralConfig()
import matplotlib.pyplot as plt


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

    
    # Tracking
    loss_hist = []
    totvio_hist = []

    # Run optimization
    for it in range(opt.budget):
        # ask
        cand = opt.ask()
        # get value and vlist (and then total_vio)
        value = evaluate(**cand.kwargs)
        vlist = constraints(**cand.kwargs)
        total_vio = float(sum(vlist)) 

        # Log
        loss_hist.append(float(value))
        totvio_hist.append(total_vio)

        # Print this iteration
        # print(f"[{it+1:04d}] loss={value:.6f} | #constraints={len(vlist)} | total_vio={total_vio:.6f}")

        # tell
        opt.tell(cand, value, vlist)
    # Get recommendation
    recommendation = opt.provide_recommendation()

    # === Combined plot ===
    plt.figure()
    plt.plot(loss_hist, label="Loss")
    plt.plot(totvio_hist, label="Total violation")
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.title("Loss & Constraint Violation per Iteration")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Extract best assignment
    best_kwargs = recommendation.kwargs
    best_app_assignment = best_kwargs["app_assignments"]
    best_op_assignment = best_kwargs["op_assignments"]
    best_fee2gas_chains = best_kwargs["fee2gas_chains"]

    # Recompute utilities (positive value this time)
    from core.model_basic import evaluate_utilities, evaluate_constraints
    score = evaluate_utilities(
        best_app_assignment, 
        best_op_assignment, 
        best_fee2gas_chains, 
        instance)
    constraints = evaluate_constraints(
        best_app_assignment, 
        best_op_assignment, 
        best_fee2gas_chains, 
        instance)
    
    return {
        "app_assignments": best_app_assignment,
        "op_assignments": best_op_assignment,
        "fee2gas_chains": best_fee2gas_chains
    }, score, constraints
