# core/instance_generator.py
from __future__ import annotations
import random
from utils.helpers import sample_int, generate_random_lambdas
from config import GeneralConfig; general_config = GeneralConfig()
from typing import Any, Dict, List, Optional, Sequence, Tuple
from typing import Any, Dict, Optional
import random
import math
from typing import Any, Dict, List, Optional
import numpy as np






def generate_big_drama(n: int = 10):
    assert n >= 1
    apps = [
        {"gas": 100, "stake": 0.0, "price": 100}
        for _ in range(n)
    ]
    ops = (
        [{"gas": 50, "stake": 999, "price": 0.0} for _ in range(2 * n)] +  
        [{"gas": 50, "stake": 0.1, "price": 100} for _ in range(n)]    
    )
    chains = [0]
    lambdas = {"apps": 1/3, "ops": 1/3, "sys": 1/3}
    return {
        "apps": apps,
        "ops": ops,
        "chains": chains,
        "lambdas": lambdas,
        "params": {"n": n}
    }











def generate_small_drama():
    # App
    apps = [
        {"gas": 100, "stake": 0.0, "price": 100}
    ]
    ops = [
        {"gas": 50, "stake": 999, "price": 0.0},
        {"gas": 50, "stake": 999, "price": 0.0},
        {"gas": 50, "stake": 0.1, "price": 100}
    ]
    chains = [0]
    lambdas = {"apps": 1/3, "ops": 1/3, "sys": 1/3}
    return {
        "apps": apps,
        "ops": ops,
        "chains": chains,
        "lambdas": lambdas,
    }












def _pareto_trunc(
    rng: np.random.Generator,
    alpha: float,
    xmin: float,
    xmax: float,
    size: int,
) -> np.ndarray:
    """
    Sample X ~ Pareto(xmin, alpha) and clamp to [xmin, xmax].
    Here Pareto(xmin, alpha) has CDF: 1 - (xmin/x)^alpha for x >= xmin.
    """
    assert alpha > 0.0
    assert xmin > 0.0 and xmax > xmin
    u = rng.random(size)
    x = xmin * (1.0 - u) ** (-1.0 / alpha)
    return np.clip(x, xmin, xmax)
def generate_realworldish_instance(
    n_app: int = 50,
    n_op: int = 30,
    seed: int = 0,
    # Apps: gas ~ ParetoTrunc(alpha_app, gmin, gmax)
    alpha_app: float = 1.6,
    gmin: float = 1.0,
    gmax: float = 500.0,
    # Ops: stake ~ ParetoTrunc(alpha_op, smin, smax)
    alpha_op: float = 1.3,
    smin: float = 1.0,
    smax: float = 1000.0,
    # Multiplicative couplings (lognormal factors)
    # stake_app = gas_app * U,   U ~ LogNormal(mu_U, sigma_U)
    mu_U: float = math.log(0.5),
    sigma_U: float = 0.2,
    # gas_op = stake_op * V,     V ~ LogNormal(mu_V, sigma_V)
    mu_V: float = math.log(0.2),
    sigma_V: float = 0.2,
    # Prices (lognormal)
    # price_app ~ LogNormal(mu_p, sigma_p)
    # price_op  ~ LogNormal(mu_p + log(lambda_floor), sigma_p)
    mu_p: float = math.log(10.0),
    sigma_p: float = 0.4,
    lambda_floor: float = 0.7,
    # Optional slack control: target_slack = (sum gas_op)/(sum gas_app)
    target_slack: Optional[float] = None,
    rescale_slack_on: str = "ops",  # {"ops","apps"}
) -> Dict[str, Any]:
    """
    Implements the LaTeX generator:

    Apps:
      gas_a      ~ ParetoTrunc(alpha_app, gmin, gmax)
      stake_a    = gas_a * U_a,   U_a ~ LogNormal(mu_U, sigma_U)
      gasprice_a ~ LogNormal(mu_p, sigma_p)

    Ops:
      stake_o    ~ ParetoTrunc(alpha_op, smin, smax)
      gas_o      = stake_o * V_o, V_o ~ LogNormal(mu_V, sigma_V)
      gasprice_o ~ LogNormal(mu_p + log(lambda_floor), sigma_p)

    Returns an instance dict compatible with your existing format.
    """
    assert n_app > 0 and n_op > 0
    assert 0.0 < lambda_floor <= 1.0
    assert sigma_U >= 0.0 and sigma_V >= 0.0 and sigma_p >= 0.0
    assert rescale_slack_on in {"ops", "apps"}

    rng = np.random.default_rng(seed)

    # --- Applications ---
    gas_app = _pareto_trunc(rng, alpha_app, gmin, gmax, n_app)
    U = rng.lognormal(mean=mu_U, sigma=sigma_U, size=n_app)
    stake_app = gas_app * U
    price_app = rng.lognormal(mean=mu_p, sigma=sigma_p, size=n_app)

    # --- Operators ---
    stake_op = _pareto_trunc(rng, alpha_op, smin, smax, n_op)
    V = rng.lognormal(mean=mu_V, sigma=sigma_V, size=n_op)
    gas_op = stake_op * V
    price_op = rng.lognormal(mean=mu_p + math.log(lambda_floor), sigma=sigma_p, size=n_op)

    apps: List[Dict[str, float]] = [
        {"gas": float(gas_app[i]), "stake": float(stake_app[i]), "price": float(price_app[i])}
        for i in range(n_app)
    ]
    ops: List[Dict[str, float]] = [
        {"gas": float(gas_op[j]), "stake": float(stake_op[j]), "price": float(price_op[j])}
        for j in range(n_op)
    ]

    chains = [0]  # ephemeral chains are formed downstream
    lambdas = {"apps": 1 / 3, "ops": 1 / 3, "sys": 1 / 3}

    params = {
        "n_app": n_app,
        "n_op": n_op,
        "seed": seed,
        "alpha_app": alpha_app,
        "gmin": gmin,
        "gmax": gmax,
        "alpha_op": alpha_op,
        "smin": smin,
        "smax": smax,
        "mu_U": mu_U,
        "sigma_U": sigma_U,
        "mu_V": mu_V,
        "sigma_V": sigma_V,
        "mu_p": mu_p,
        "sigma_p": sigma_p,
        "lambda_floor": lambda_floor,
        "target_slack": target_slack,
        "rescale_slack_on": rescale_slack_on,
    }

    return {
        "apps": apps,
        "ops": ops,
        "chains": chains,
        "lambdas": lambdas,
        "params": params,
    }







def generate_simplex_instance_big_1(
    price_spread: float = 0.6,   # kappa in (0,1]
    stake_skew: float = 0.95,    # "extremeness": closer to 1 => tinier whale stake (see eps below)
    D: int = 100,                # total demand (see note: pick ~100*K for strongest effect)
    pmax: float = 10.0,
    n_apps: int = 10,
    n_ops: int = 21,             # rounded down to 3K
    whale_gas_mult: float = 2.4, # whale gas = whale_gas_mult * low_gas
    whale_price_noise: float = 0.6,  # per-triple whale floor multiplier in [1-noise, 1+noise]
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    "Big" instance = K replicated copies of the original 1-app/3-op misalignment gadget,
    with *extreme* governance sensitivity.

    Minimal changes relative to generate_simplex_instance_1:
      (i) replicate the 3-op gadget K times (disjoint triples),
      (ii) split the single app into n_apps apps (equal split),
      (iii) make the whale stake tiny (yield leverage) via eps,
      (iv) make the whale have larger capacity (so it matters),
      (v) add mild heterogeneity in whale floors across triples.

    Returns only what an instance needs: apps, ops, chains, lambdas.
    """
    assert 0.0 < price_spread <= 1.0, "kappa must be in (0,1]"
    assert 0.0 < stake_skew < 1.0, "stake_skew must be in (0,1)"
    assert D > 0 and pmax > 0.0
    assert n_apps >= 1 and n_ops >= 3
    assert whale_gas_mult >= 1.0
    assert 0.0 <= whale_price_noise < 1.0

    rng = random.Random(seed)

    # Number of 3-op gadgets
    K = max(1, n_ops // 3)
    n_ops_eff = 3 * K

    # Apps: n_apps copies, split total demand evenly
    gas_per_app = float(D) / float(n_apps)
    apps = [{"gas": gas_per_app, "stake": 0.0, "price": float(pmax)} for _ in range(n_apps)]

    # Operators: K triples
    low_gas = 50.0
    whale_gas = float(whale_gas_mult) * low_gas

    # Extreme yield lever: tiny whale stake
    # Interpret stake_skew as "how extreme": close to 1 => eps ~ 0
    eps = max(1e-5, 1.0 - float(stake_skew))      # whale stake inside each triple before global scaling
    low_stake = (1.0 - eps) / 2.0

    base_whale_floor = float(price_spread) * float(pmax)

    ops = []
    for t in range(K):
        # mild heterogeneity: some whales are "more expensive" than others
        mult = (1.0 - whale_price_noise) + 2.0 * whale_price_noise * rng.random()  # in [1-noise, 1+noise]
        whale_floor_t = min(float(pmax), base_whale_floor * mult)

        triple = [
            {"gas": low_gas,  "stake": low_stake, "price": 0.0},
            {"gas": low_gas,  "stake": low_stake, "price": 0.0},
            {"gas": whale_gas,"stake": eps,       "price": whale_floor_t},
        ]

        # break index symmetry (doesn't change the gadget)
        rng.shuffle(triple)

        # scale stakes so total stake across all ops sums to 1
        for o in triple:
            o["stake"] = float(o["stake"]) / float(K)
            ops.append(o)

    chains = [0]
    lambdas = {"apps": 1 / 3, "ops": 1 / 3, "sys": 1 / 3}

    return {
        "apps": apps,
        "ops": ops,
        "chains": chains,
        "lambdas": lambdas,
        "params": {
            "price_spread": price_spread,
            "stake_skew": stake_skew,
            "D": D,
            "pmax": pmax,
            "n_apps": n_apps,
            "n_ops": n_ops_eff,
            "K": K,
            "whale_gas_mult": whale_gas_mult,
            "whale_price_noise": whale_price_noise,
            "seed": seed,
            "eps_whale_stake_per_triple": eps,
        },
    }





def generate_simplex_instance_1(price_spread: float = 0.6,
                                stake_skew: float = 0.7,
                                D: int = 100,
                                pmax: float = 10.0):
    """
    Parametric instance (1 app, 3 ops) exhibiting all three misalignments.
    - App: gas D, stake 0 (non-binding), price cap pmax.
    - Ops: three operators, each gas=50; two low-floor (p_min=0), one high-floor (p_min=kappa*pmax).
            Stakes sum to 1: whale has sigma, each low-floor has (1-sigma)/2.
    Notes:
      * Midpoint pricing and utilities are applied downstream (core model).
      * Any bundle with >=2 ops can serve full demand (sum gas >= D).
    """
    assert 0.0 < price_spread <= 1.0, "kappa must be in (0,1]"
    assert 0.0 < stake_skew < 1.0, "sigma must be in (0,1)"

    # App
    apps = [
        {"gas": float(D), "stake": 0.0, "price": float(pmax)}  # price = p_max(a)
    ]

    # Operators: two low-floor, one high-floor ("whale")
    op_gas = 50.0
    low_stake = (1.0 - stake_skew) / 2.0
    ops = [
        {"gas": op_gas, "stake": low_stake, "price": 0.0},                 # o_L^(1): p_min = 0
        {"gas": op_gas, "stake": low_stake, "price": 0.0},                 # o_L^(2): p_min = 0
        {"gas": op_gas, "stake": float(stake_skew), "price": float(price_spread*pmax)} # o_H:     p_min = kappa * p_max
    ]

    # Single logical “market” (ephemeral chains are formed downstream by subsets of ops)
    chains = [0]

    # Default lambdas (you’ll sweep these over the simplex in experiments)
    lambdas = {"apps": 1/3, "ops": 1/3, "sys": 1/3}

    return {
        "apps": apps,
        "ops": ops,
        "chains": chains,
        "lambdas": lambdas,
        "params": {"price_spread": price_spread, "stake_skew": stake_skew, "D": D, "pmax": pmax}
    }











def generate_validation_example_1():
    """
    Instance (1 chain; 1 app; 1 operator):
      a: gas 100; stake 50; price 10
      o: gas 100; stake 50; price 10
    
    Solution:
      Qmax = 1000
      Solution {(a, o, 10)}
      Ua = 1
      Uo = 0.02
      Us = 1
      With lambdas 0.5, 0.3, 0.2:
        U = 0.5 * 1 + 0.3 * 0.02 + 0.2 * 1 = 0.706
    """
    return {
        "apps": [
            {"gas": 100, "stake": 50, "price": 10},   
        ],
        "ops": [
            {"gas": 100, "stake": 50, "price": 10},   
        ],
        "chains": [0],
        "lambdas": {"apps": 0.5, "ops": 0.3, "sys": 0.2},
    }


def generate_validation_example_2():
    """
    Instance (1 chain; 1 app; 1 operator):
      a: gas 100; stake 50; price 10
      o: gas 100; stake 50; price 10
    
    Solution:
      Qmax = 1000
      Solution {(a, o, 10)}
      Ua = 1
      Uo = 0.02
      Us = 1
      With lambdas 0.5, 0.3, 0.2:
        U = 0.5 * 1 + 0.3 * 0.02 + 0.2 * 1 = 0.706
    """
    return {
        "apps": [
            {"gas": 100, "stake": 50, "price": 10},   
        ],
        "ops": [
            {"gas": 100, "stake": 50, "price": 10},   
        ],
        "chains": [0],
        "lambdas": {"apps": 0.9, "ops": 0.05, "sys": 0.05},
    }


def generate_validation_example_3_app():
    """
    VE3 (see overleaf)
    """
    return {
        "apps": [
            {"gas": 100, "stake": 50, "price": 10},   
        ],
        "ops": [
            {"gas": 100, "stake": 50, "price": 0},   
            {"gas": 100, "stake": 50, "price": 10},   
        ],
        "chains": [0],
        "lambdas": {"apps": 1, "ops": 0, "sys": 0},
    }

def generate_validation_example_3_op():
    """
    VE3 (see overleaf)
    """
    return {
        "apps": [
            {"gas": 100, "stake": 50, "price": 10},   
        ],
        "ops": [
            {"gas": 100, "stake": 50, "price": 0},   
            {"gas": 100, "stake": 50, "price": 10},   
        ],
        "chains": [0],
        "lambdas": {"apps": 0, "ops": 1, "sys": 0},
    }


def generate_validation_example_4_app():
    """
    VE3 (see overleaf)
    """
    return {
        "apps": [
            {"gas": 100, "stake": 50, "price": 10},   
        ],
        "ops": [
            {"gas": 100, "stake": 50, "price": 0},   
            {"gas": 100, "stake": 50, "price": 10},   
        ],
        "chains": [0],
        "lambdas": {"apps": 1, "ops": 0, "sys": 0},
    }


def generate_validation_example_4_sys():
    """
    VE3 (see overleaf)
    """
    return {
        "apps": [
            {"gas": 100, "stake": 50, "price": 10},   
        ],
        "ops": [
            {"gas": 100, "stake": 50, "price": 0},   
            {"gas": 100, "stake": 50, "price": 10},   
        ],
        "chains": [0],
        "lambdas": {"apps": 0, "ops": 0, "sys": 1},
    }


def generate_validation_example_5_op():
    """
    VE3 (see overleaf)
    """
    return {
        "apps": [
            {"gas": 100, "stake": 50, "price": 10},   
        ],
        "ops": [
            {"gas": 100, "stake": 50, "price": 0},   
            {"gas": 100, "stake": 1000, "price": 10},   
        ],
        "chains": [0],
        "lambdas": {"apps": 0, "ops": 1, "sys": 0},
    }


def generate_validation_example_5_sys():
    """
    VE3 (see overleaf)
    """
    return {
        "apps": [
            {"gas": 100, "stake": 50, "price": 10},   
        ],
        "ops": [
            {"gas": 100, "stake": 50, "price": 0},   
            {"gas": 100, "stake": 1000, "price": 10},   
        ],
        "chains": [0],
        "lambdas": {"apps": 0, "ops": 0, "sys": 1},
    }


def generate_toy_instance_1():
    """
    Instance (1 chain; 1 app; 1 operator):
      App0: demands 10 gas; requires 5 stake; can offer fee2gas=2
      Op0: supplies 15 gas; has 6 stake; requires fee2gas=1  
    
    Solution:
      App and op on the chain (stake is fine)
      There is 10 gas on the chain
      The chain's fee2gas is 2 (this is what the app agrees; and lambda.ops = 1)
      So the total fee on the chain is 20
       
    Utilities:
      App base util is 10 (gas computed for is 10)
      Op base util is 20 / 6 = 3.333 (fee on chain is 20, only one op on chain, op stake is 6)
      Sys base util is 20 (total network fee)
      Weighted total = 0 * 10 + 1 * 3.333 + 0 * 20 = 3.333
    """
    return {
        "apps": [
            {"gas": 10, "stake": 5, "fee2gas": 2},   
        ],
        "ops": [
            {"gas": 15, "stake": 6, "fee2gas": 1},   
        ],
        "chains": [0],
        "lambdas": {"apps": 0, "ops": 1, "sys": 0},
    }


def generate_toy_instance_2():
    """
    Instance (3 chains; 3 app; 3 operators):
      App0: demands 10 gas; requires 5 stake; can offer fee2gas=2
      Op0: supplies 15 gas; has 6 stake; requires fee2gas=1  
    
    Solution:
      App and op on the chain (stake is fine)
      There is 10 gas on the chain
      The chain's fee2gas is 2 (this is what the app agrees; and lambda.ops = 1)
      So the total fee on the chain is 20
       
    Utilities:
      App base util is 10 (gas computed for is 10)
      Op base util is 20 / 6 = 3.333 (fee on chain is 20, only one op on chain, op stake is 6)
      Sys base util is 20 (total network fee)
      Weighted total = 0 * 10 + 1 * 3.333 + 0 * 20 = 3.333
    """
    return {
        "apps": [
            {"gas": 50.0, "stake": 50.0, "fee2gas": 10.0},
            {"gas": 50.0, "stake": 50.0, "fee2gas": 10.0},
            {"gas": 50.0, "stake": 50.0, "fee2gas": 10.0}
        ],
        "ops": [
            {"gas": 200.0, "stake": 50.0, "fee2gas": 10.0},  
            {"gas": 200.0, "stake": 25.0, "fee2gas": 10.0},  
            {"gas": 200.0, "stake": 25.0, "fee2gas": 10.0} 
        ],
        "chains": list(range(3)),
        "lambdas": {"apps": 0, "ops": 1, "sys": 0},
    }


def generate_random_instance_1(
        num_apps = general_config.num_apps,
        num_ops = general_config.num_ops,
        num_chains = general_config.num_chains):
    """
    Generates a random instance with uniformly sampled integral parameters for apps and ops.

    Returns:
        instance: dict with keys:
            - "apps": list of dicts, each with gas, stake, fee2gas
            - "ops": list of dicts, each with gas, stake, fee2gas
            - "chains": list of chain IDs (no attributes in basic model)
            - "lambdas": dict with apps, ops, sys (summing to 1)
    """
    # Populate entities
    apps = []
    for _ in range(num_apps):
        app = {
            "gas": sample_int(10, 100),
            "stake": sample_int(10, 50),
            "price": sample_int(1, 10)
        }
        apps.append(app)

    ops = []
    for _ in range(num_ops):
        op = {
            "gas": sample_int(10, 500),
            "stake": sample_int(10, 100),
            "price": sample_int(1, 10)
        }
        ops.append(op)

    chains = list(range(num_chains))

    # Random convex combination for lambdas
    lambdas = generate_random_lambdas(("apps", "ops", "sys"))

    return {
        "apps": apps,
        "ops": ops,
        "chains": chains,
        "lambdas": lambdas
    }


def generate_simplex_instance_old_1(
    num_apps=general_config.num_apps,
    num_ops=general_config.num_ops,
    num_chains=general_config.num_chains,
):
    """
    Generates a 'showcase' instance for simplex visualization.
    Produces mild congestion (Demand ≈ 1.2 × Supply), overlapping fee bands,
    and diverse stakes to make trade-offs visible across governance weights.

    Returns:
        instance: dict with keys:
            - "apps": list of dicts, each with gas, stake, fee2gas
            - "ops": list of dicts, each with gas, stake, fee2gas
            - "chains": list of chain IDs
            - "lambdas": dict with "apps", "ops", "sys" (summing to 1)
    """
    import random

    # --- helpers ------------------------------------------------------------
    def rescale_total(vals, target_sum):
        s = sum(vals)
        if s <= 0:
            return vals
        f = target_sum / s
        return [max(1, int(round(v * f))) for v in vals]

    def sample_bimodal(n, low_range, hi_range, hi_frac):
        """Returns n integers split between low and high ranges."""
        k_hi = int(round(hi_frac * n))
        vals = [random.randint(*low_range) for _ in range(n - k_hi)]
        vals += [random.randint(*hi_range) for _ in range(k_hi)]
        random.shuffle(vals)
        return vals

    # --- main sampling ------------------------------------------------------
    rng = random.Random()

    # 1) gas
    apps_gas = [rng.randint(10, 120) for _ in range(num_apps)]
    ops_gas = [rng.randint(40, 240) for _ in range(num_ops)]

    # mild congestion: total demand ≈ 1.2 × total supply
    target_supply = sum(ops_gas)
    target_demand = int(round(1.2 * target_supply))
    apps_gas = rescale_total(apps_gas, target_demand)

    # 2) stake (diverse)
    apps_stake = sample_bimodal(num_apps, (10, 40), (60, 120), hi_frac=0.2)
    ops_stake = sample_bimodal(num_ops, (20, 80), (120, 200), hi_frac=0.2)

    # 3) fee2gas (bimodal, overlapping)
    apps_fee2gas = sample_bimodal(num_apps, (2, 6), (7, 12), hi_frac=0.3)
    ops_fee2gas = sample_bimodal(num_ops, (2, 6), (6, 9), hi_frac=0.3)

    # --- build entities -----------------------------------------------------
    apps = [
        {"gas": g, "stake": s, "fee2gas": f}
        for g, s, f in zip(apps_gas, apps_stake, apps_fee2gas)
    ]
    ops = [
        {"gas": g, "stake": s, "fee2gas": f}
        for g, s, f in zip(ops_gas, ops_stake, ops_fee2gas)
    ]

    # chains = list(range(num_chains))

    # Random convex combination for governance weights
    lambdas = generate_random_lambdas(("apps", "ops", "sys"))

    return {
        "apps": apps,
        "ops": ops,
        "chains": [0],
        "lambdas": lambdas,
    }
