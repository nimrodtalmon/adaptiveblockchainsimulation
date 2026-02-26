"""
Instance generators for adaptive multichain optimization.
"""

import random
from typing import Optional

from model import App, Op, Instance


def toy_instance() -> Instance:
    """
    Minimal instance for hand verification.
    
    1 app (gas=100, stake=10, price=10)
    1 op  (gas=150, stake=50, price=5)
    
    Expected behavior:
    - Feasible: stake 50 >= required 10 ✓
    - Price bounds: [5, 10]
    - Gas processed: min(100, 150) = 100
    - At price=10: sys_util=1.0, op_util=1.0, app_util=0.9
    - At price=5:  sys_util=0.5, op_util=0.5, app_util=0.95
    """
    return Instance(
        apps=[App(gas=100, stake=10, price=10)],
        ops=[Op(gas=150, stake=50, price=5)],
    )


def random_instance(
    n_apps: int = 5,
    n_ops: int = 3,
    seed: Optional[int] = None,
) -> Instance:
    """
    Generate a random instance with uniformly sampled parameters.
    
    Args:
        n_apps: Number of applications
        n_ops: Number of operators
        seed: Random seed (optional)
    
    Returns:
        Random Instance
    """
    if seed is not None:
        random.seed(seed)
    
    apps = [
        App(
            gas=random.uniform(10, 100),
            stake=random.uniform(0, 50),
            price=random.uniform(5, 20),
        )
        for _ in range(n_apps)
    ]
    
    ops = [
        Op(
            gas=random.uniform(50, 200),
            stake=random.uniform(10, 100),
            price=random.uniform(1, 10),
        )
        for _ in range(n_ops)
    ]
    
    return Instance(apps=apps, ops=ops)


def simplex_gadget(
    kappa: float = 0.3,
    sigma: float = 0.9,
    demand: float = 100.0,
    price_max: float = 10.0,
    seed: Optional[int] = None,
) -> Instance:
    """
    Classic simplex gadget: 1 app, 3 operators.
    
    Creates governance sensitivity by having:
    - 2 "low-floor" operators: price=0, high stake
    - 1 "whale" operator: high price floor, low stake (high yield potential)
    
    Args:
        kappa: Price spread (0,1]. Whale's min price = kappa * price_max
        sigma: Stake skew (0,1). Higher = whale has less stake (more yield leverage)
        demand: App's gas demand
        price_max: App's max acceptable price
        seed: Random seed (optional)
    
    Returns:
        Instance with 1 app and 3 operators
    """
    if seed is not None:
        random.seed(seed)
    
    # Single app
    app = App(gas=demand, stake=0.0, price=price_max)
    
    # Operator stakes (sum to 1 for easy reasoning)
    whale_stake = 1.0 - sigma  # small when sigma is high
    low_stake = sigma / 2.0    # split remainder between two low-floor ops
    
    # Each operator has capacity 50 (total 150 > demand 100)
    op_capacity = 50.0
    
    # Whale has high price floor
    whale_price = kappa * price_max
    
    ops = [
        Op(gas=op_capacity, stake=low_stake, price=0.0),      # low-floor 1
        Op(gas=op_capacity, stake=low_stake, price=0.0),      # low-floor 2
        Op(gas=op_capacity, stake=whale_stake, price=whale_price),  # whale
    ]
    
    return Instance(apps=[app], ops=ops)