"""
Optimizer for adaptive multichain configuration.

Uses Nevergrad to search over assignments (app→chain, op→chain).
Prices are computed optimally via bilevel optimization inside model.evaluate().
"""

import nevergrad as ng
from typing import Optional, Tuple, List

from model import Instance, Lambdas, Solution, evaluate


def solve(
    instance: Instance,
    lambdas: Lambdas,
    budget: int = 100,
    seed: Optional[int] = None,
    verbose: bool = False,
    optimizer_class: str = "DiscreteOnePlusOne",
) -> Tuple[Solution, List[float]]:
    """
    Find optimal assignment of apps and ops to chains.
    
    Args:
        instance: Problem instance (apps, ops)
        lambdas: Governance weights
        budget: Number of Nevergrad evaluations
        seed: Random seed (optional)
        verbose: Print progress
        optimizer_class: Nevergrad optimizer name. Options include:
            - "DiscreteOnePlusOne" (default, good for discrete)
            - "NGOpt" (auto-selector)
            - "OnePlusOne" (general purpose)
            - "PortfolioDiscreteOnePlusOne" (portfolio)
    
    Returns:
        best_solution: Solution with optimal assignment and utilities
        history: List of total_utility values per iteration
    """
    n_apps = instance.n_apps
    n_ops = instance.n_ops
    max_chains = instance.max_chains
    
    # Build search space: each agent assigned to chain 0..max_chains-1, or -1 (unassigned)
    app_vars = [ng.p.Choice(list(range(-1, max_chains))) for _ in range(n_apps)]
    op_vars = [ng.p.Choice(list(range(-1, max_chains))) for _ in range(n_ops)]
    
    parametrization = ng.p.Instrumentation(
        app_assignments=ng.p.Tuple(*app_vars),
        op_assignments=ng.p.Tuple(*op_vars),
    )
    
    # Set up optimizer
    opt_cls = getattr(ng.optimizers, optimizer_class)
    optimizer = opt_cls(
        parametrization=parametrization,
        budget=budget,
        num_workers=1,
    )
    
    if seed is not None:
        parametrization.random_state.seed(seed)
        optimizer._rng.seed(seed)
    
    # Tracking
    history = []
    best_solution = None
    best_value = float('inf')
    
    # Penalty for infeasible solutions
    INFEASIBILITY_PENALTY = 1e6
    
    # Optimization loop
    for i in range(budget):
        candidate = optimizer.ask()
        
        app_assignments = tuple(candidate.kwargs["app_assignments"])
        op_assignments = tuple(candidate.kwargs["op_assignments"])
        
        # Evaluate (includes bilevel pricing)
        solution = evaluate(instance, app_assignments, op_assignments, lambdas)
        
        # Nevergrad minimizes, we want to maximize total_utility
        if solution.feasible:
            loss = -solution.total_utility
        else:
            loss = INFEASIBILITY_PENALTY
        
        optimizer.tell(candidate, loss)
        history.append(solution.total_utility if solution.feasible else 0.0)
        
        # Track best feasible solution
        if solution.feasible and loss < best_value:
            best_value = loss
            best_solution = solution
        
        if verbose and (i + 1) % 10 == 0:
            status = f"feasible, util={solution.total_utility:.4f}" if solution.feasible else "infeasible"
            print(f"[{i+1:4d}/{budget}] {status}")
    
    # If no feasible solution found, return the last one (infeasible)
    if best_solution is None:
        best_solution = solution
        if verbose:
            print("Warning: No feasible solution found!")
    
    return best_solution, history