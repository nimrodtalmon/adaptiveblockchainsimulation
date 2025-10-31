# core/instance_generator.py
import random
from typing import Dict, Tuple
from utils.helpers import sample_int, generate_random_lambdas
from config import GeneralConfig; general_config = GeneralConfig()


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


def generate_simplex_instance_1(
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

    chains = list(range(num_chains))

    # Random convex combination for governance weights
    lambdas = generate_random_lambdas(("apps", "ops", "sys"))

    return {
        "apps": apps,
        "ops": ops,
        "chains": chains,
        "lambdas": lambdas,
    }
