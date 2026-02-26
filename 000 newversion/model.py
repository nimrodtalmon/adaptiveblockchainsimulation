"""
Core model for adaptive multichain optimization.

Data flow:
    Instance (apps, ops)
        ↓
    Assignment (app→chain, op→chain)
        ↓
    + Bilevel pricing (per chain, O(1))
        ↓
    Solution (assignment + prices + utilities)
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


# =============================================================================
# Data Structures
# =============================================================================

@dataclass(frozen=True)
class App:
    """Application agent."""
    gas: float      # gas demand
    stake: float    # required stake (security requirement)
    price: float    # max acceptable gas price


@dataclass(frozen=True)
class Op:
    """Operator agent."""
    gas: float      # gas capacity
    stake: float    # available stake
    price: float    # min acceptable gas price


@dataclass
class Instance:
    """Problem instance."""
    apps: List[App]
    ops: List[Op]
    
    @property
    def n_apps(self) -> int:
        return len(self.apps)
    
    @property
    def n_ops(self) -> int:
        return len(self.ops)
    
    @property
    def max_chains(self) -> int:
        return min(self.n_apps, self.n_ops)


@dataclass
class Lambdas:
    """Governance weights (must sum to 1)."""
    apps: float
    ops: float
    sys: float
    
    def __post_init__(self):
        total = self.apps + self.ops + self.sys
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Lambdas must sum to 1, got {total}")


@dataclass
class ChainState:
    """Derived state for a single chain."""
    app_indices: List[int]      # which apps are on this chain
    op_indices: List[int]       # which ops are on this chain
    demand: float               # total gas demand from apps
    supply: float               # min gas capacity among ops (bottleneck)
    gas: float                  # actual gas processed = min(demand, supply)
    total_stake: float          # sum of op stakes
    price_lo: float             # lower bound: max op price
    price_hi: float             # upper bound: min app price
    price: float                # chosen clearing price (from bilevel opt)
    feasible: bool              # is this chain feasible?


@dataclass
class Solution:
    """Complete solution with utilities."""
    app_assignments: Tuple[int, ...]    # app i → chain (or -1)
    op_assignments: Tuple[int, ...]     # op i → chain (or -1)
    chains: Dict[int, ChainState]       # chain_id → state
    
    # Utilities (per-agent)
    app_utilities: List[float]
    op_utilities: List[float]
    sys_utility: float
    
    # Aggregated utilities (averaged)
    avg_app_utility: float
    avg_op_utility: float
    
    # Weighted total
    total_utility: float
    
    # Feasibility
    feasible: bool


# =============================================================================
# Chain Derivation
# =============================================================================

def derive_chains(
    instance: Instance,
    app_assignments: Tuple[int, ...],
    op_assignments: Tuple[int, ...],
) -> Dict[int, ChainState]:
    """
    Given an assignment, derive the state of each active chain.
    
    Returns dict mapping chain_id → ChainState.
    Only includes chains that have at least one app OR one op.
    """
    # Group apps and ops by chain
    apps_on_chain: Dict[int, List[int]] = {}
    ops_on_chain: Dict[int, List[int]] = {}
    
    for a, c in enumerate(app_assignments):
        if c != -1:
            apps_on_chain.setdefault(c, []).append(a)
    
    for o, c in enumerate(op_assignments):
        if c != -1:
            ops_on_chain.setdefault(c, []).append(o)
    
    # All active chain IDs
    active_chains = set(apps_on_chain.keys()) | set(ops_on_chain.keys())
    
    chains = {}
    for c in active_chains:
        app_idxs = apps_on_chain.get(c, [])
        op_idxs = ops_on_chain.get(c, [])
        
        # Demand = sum of app gas
        demand = sum(instance.apps[a].gas for a in app_idxs)
        
        # Supply = min of op gas (bottleneck), 0 if no ops
        if op_idxs:
            supply = min(instance.ops[o].gas for o in op_idxs)
        else:
            supply = 0.0
        
        # Gas processed
        if demand > 0 and supply > 0:
            gas = min(demand, supply)
        else:
            gas = 0.0
        
        # Total stake from ops
        total_stake = sum(instance.ops[o].stake for o in op_idxs)
        
        # Price bounds
        if op_idxs:
            price_lo = max(instance.ops[o].price for o in op_idxs)
        else:
            price_lo = 0.0
        
        if app_idxs:
            price_hi = min(instance.apps[a].price for a in app_idxs)
        else:
            price_hi = float('inf')
        
        # Feasibility: price interval must be valid, and stake must cover app requirements
        price_feasible = (price_lo <= price_hi) if (app_idxs and op_idxs) else True
        
        stake_feasible = True
        if app_idxs and op_idxs:
            max_app_stake_req = max(instance.apps[a].stake for a in app_idxs)
            stake_feasible = (total_stake >= max_app_stake_req)
        
        feasible = price_feasible and stake_feasible
        
        # Price will be set by bilevel optimization (placeholder for now)
        price = 0.0
        
        chains[c] = ChainState(
            app_indices=app_idxs,
            op_indices=op_idxs,
            demand=demand,
            supply=supply,
            gas=gas,
            total_stake=total_stake,
            price_lo=price_lo,
            price_hi=price_hi,
            price=price,
            feasible=feasible,
        )
    
    return chains


# =============================================================================
# Bilevel Pricing (Inner Optimization)
# =============================================================================

def make_chain_with_price(chain: ChainState, price: float) -> ChainState:
    """Create a copy of chain with a specific price."""
    return ChainState(
        app_indices=chain.app_indices,
        op_indices=chain.op_indices,
        demand=chain.demand,
        supply=chain.supply,
        gas=chain.gas,
        total_stake=chain.total_stake,
        price_lo=chain.price_lo,
        price_hi=chain.price_hi,
        price=price,
        feasible=chain.feasible,
    )


def aggregate_utilities(
    app_utilities: List[float],
    op_utilities: List[float],
    sys_utility: float,
    lambdas: Lambdas,
) -> Tuple[float, float, float]:
    """
    Compute aggregated utilities.
    
    Returns:
        avg_app: average app utility
        avg_op: average op utility
        total: weighted total (lambdas.apps * avg_app + lambdas.ops * avg_op + lambdas.sys * sys_utility)
    """
    avg_app = sum(app_utilities) / len(app_utilities) if app_utilities else 0.0
    avg_op = sum(op_utilities) / len(op_utilities) if op_utilities else 0.0
    total = lambdas.apps * avg_app + lambdas.ops * avg_op + lambdas.sys * sys_utility
    return avg_app, avg_op, total


def compute_optimal_prices(
    instance: Instance,
    chains: Dict[int, ChainState],
    app_assignments: Tuple[int, ...],
    op_assignments: Tuple[int, ...],
    lambdas: Lambdas,
) -> Dict[int, ChainState]:
    """
    Given an assignment; for each chain, compute the optimal clearing price.
    
    The objective is affine in price, so optimum is at a boundary.
    We evaluate both boundaries using compute_utilities and pick the better one.
    
    Returns updated chains dict with price field set.
    """
    updated_chains = {}
    
    for c, chain in chains.items():
        if not chain.feasible or chain.gas == 0:
            # Infeasible or empty chain: price doesn't matter
            updated_chains[c] = make_chain_with_price(chain, -1)
            continue
        
        # Try both boundary prices by evaluating full utilities
        def total_at_price(price: float) -> float:
            test_chains = {**chains, c: make_chain_with_price(chain, price)}
            app_utils, op_utils, sys_util = compute_utilities(
                instance, test_chains, app_assignments, op_assignments, lambdas
            )
            _, _, total = aggregate_utilities(app_utils, op_utils, sys_util, lambdas)
            return total
        
        obj_lo = total_at_price(chain.price_lo)
        obj_hi = total_at_price(chain.price_hi)
        
        optimal_price = chain.price_hi if obj_hi >= obj_lo else chain.price_lo
        updated_chains[c] = make_chain_with_price(chain, optimal_price)
    
    return updated_chains


# =============================================================================
# Utility Computation
# =============================================================================

def compute_utilities(
    instance: Instance,
    chains: Dict[int, ChainState],
    app_assignments: Tuple[int, ...],
    op_assignments: Tuple[int, ...],
    lambdas: Lambdas,
) -> Tuple[List[float], List[float], float]:
    """
    Compute utilities for all agents and the system.
    
    Returns:
        app_utilities: list of per-app utilities
        op_utilities: list of per-op utilities  
        sys_utility: system utility
    """
    # Normalization constants
    total_gas_supply = sum(op.gas for op in instance.ops)
    total_gas_demand = sum(app.gas for app in instance.apps)
    max_price = max(app.price for app in instance.apps) if instance.apps else 1.0
    min_stake = min(op.stake for op in instance.ops) if instance.ops else 1.0
    
    Q_sys_max = min(total_gas_supply, total_gas_demand) * max_price + 1e-12
    Q_op_max = Q_sys_max / min_stake + 1e-12
    
    # App utilities
    app_utilities = []
    for a, c in enumerate(app_assignments):
        if c == -1 or c not in chains:
            app_utilities.append(0.0)
            continue
        
        chain = chains[c]
        if chain.demand == 0:
            app_utilities.append(0.0)
            continue
        
        # Utilization: fraction of demand served
        utilization = chain.gas / chain.demand
        
        # Price penalty
        price_penalty = chain.price / instance.apps[a].price
        
        # Utility formula: 0.1 + 0.9 * utilization - 0.1 * price_penalty
        util = 0.1 + 0.9 * utilization - 0.1 * price_penalty
        app_utilities.append(util)
    
    # Op utilities
    op_utilities = []
    for o, c in enumerate(op_assignments):
        if c == -1 or c not in chains:
            op_utilities.append(0.0)
            continue
        
        chain = chains[c]
        if chain.total_stake == 0:
            op_utilities.append(0.0)
            continue
        
        # Fee on chain
        fee = chain.price * chain.gas
        
        # Yield = fee / total_stake (each op gets same yield per stake unit)
        yield_val = fee / chain.total_stake
        
        # Normalized utility
        util = yield_val / Q_op_max
        op_utilities.append(util)
    
    # System utility
    total_fees = sum(chain.price * chain.gas for chain in chains.values())
    sys_utility = total_fees / Q_sys_max
    
    return app_utilities, op_utilities, sys_utility


# =============================================================================
# Top-Level Evaluation
# =============================================================================

def evaluate(
    instance: Instance,
    app_assignments: Tuple[int, ...],
    op_assignments: Tuple[int, ...],
    lambdas: Lambdas,
) -> Solution:
    """
    Full evaluation: assignment → bilevel pricing → utilities → Solution.
    
    This is the main entry point for evaluating any candidate assignment.
    """
    # Step 1: Derive chain states
    chains = derive_chains(instance, app_assignments, op_assignments)
    
    # Step 2: Compute optimal prices (bilevel inner optimization)
    chains = compute_optimal_prices(instance, chains, app_assignments, op_assignments, lambdas)
    
    # Step 3: Check overall feasibility
    all_feasible = all(chain.feasible for chain in chains.values())
    
    # Step 4: Compute utilities
    app_utilities, op_utilities, sys_utility = compute_utilities(
        instance, chains, app_assignments, op_assignments, lambdas
    )
    
    # Step 5: Aggregate
    avg_app, avg_op, total = aggregate_utilities(app_utilities, op_utilities, sys_utility, lambdas)
    
    return Solution(
        app_assignments=app_assignments,
        op_assignments=op_assignments,
        chains=chains,
        app_utilities=app_utilities,
        op_utilities=op_utilities,
        sys_utility=sys_utility,
        avg_app_utility=avg_app,
        avg_op_utility=avg_op,
        total_utility=total,
        feasible=all_feasible,
    )