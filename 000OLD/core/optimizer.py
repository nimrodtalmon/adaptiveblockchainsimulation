# core/optimizer.py

import nevergrad as ng
import concurrent.futures
import threading
from core.model_basic import define_problem, evaluate_utilities, evaluate_constraints
from config import GeneralConfig; general_config = GeneralConfig()

# Global lists and lock for logging
loss_hist = []
totvio_hist = []
log_lock = threading.Lock()

def solve_model(instance, verbose=True):
    """
    Solves the model defined in model_basic.py using Nevergrad (NGOpt solver).
    Uses parallelization via executor.
    """
    global loss_hist, totvio_hist
    loss_hist = []
    totvio_hist = []

    # Build search space and objective
    parametrization, evaluate, constraints = define_problem(instance)

    # Wrap evaluate to log loss and violations
    def evaluate_with_logging(*args, **kwargs):
        loss = evaluate(*args, **kwargs)
        # You may need to extract assignments from args/kwargs depending on your evaluate signature
        app_assignments = kwargs["app_assignments"]
        op_assignments = kwargs["op_assignments"]
        fee2gas_chains = kwargs["fee2gas_chains"]
        violations = evaluate_constraints(app_assignments, op_assignments, fee2gas_chains, instance)
        total_violation = sum(violations)
        with log_lock:
            loss_hist.append(loss)
            totvio_hist.append(total_violation)
        return loss

    # Set up Nevergrad optimizer
    opt = general_config.nevergrad_optimizer(
        parametrization=parametrization, 
        budget=general_config.nevergrad_budget,
        num_workers=general_config.nevergrad_num_workers,
    )

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=general_config.nevergrad_num_workers)

    if verbose:
        print("\n=== Solver ===")
        print(f"[Nevergrad] {general_config.nevergrad_optimizer.__name__} | "
              f"budget={general_config.nevergrad_budget} | num_workers={general_config.nevergrad_num_workers}")

    # Run optimization using minimize with executor
    recommendation = opt.minimize(evaluate_with_logging, executor=executor)

    # Extract best assignment
    best_kwargs = recommendation.kwargs
    best_app_assignment = best_kwargs["app_assignments"]
    best_op_assignment = best_kwargs["op_assignments"]
    best_fee2gas_chains = best_kwargs["fee2gas_chains"]

    # Recompute utilities (positive value this time)
    score = evaluate_utilities(
        best_app_assignment, 
        best_op_assignment, 
        best_fee2gas_chains, 
        instance)
    constraints_val = evaluate_constraints(
        best_app_assignment, 
        best_op_assignment, 
        best_fee2gas_chains, 
        instance)

    return {
        "app_assignments": best_app_assignment,
        "op_assignments": best_op_assignment,
        "fee2gas_chains": best_fee2gas_chains
    }, score, constraints_val, loss_hist, totvio_hist
