# core/instance_generator.py
import random
from typing import Dict, Tuple
from utils.helpers import sample_int, generate_random_lambdas
from config import SimConfig; config = SimConfig()


def generate_instance(config):
    """
    Look up and call the generator function specified by name in config.instance_generator.
    The name must match a zero-arg callable defined in THIS module (core.instance_generator).
    """
    name = config.instance_generator
    if not isinstance(name, str) or not name:
        raise TypeError("config.instance_generator must be a non-empty string")

    try:
        fn = globals()[name]  # directly from this module's namespace
    except KeyError as e:
        raise ValueError(
            f"Unknown instance generator '{name}' in core.instance_generator"
        ) from e

    if not callable(fn):
        raise TypeError(f"Attribute '{name}' exists but is not callable")

    return fn()


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


def generate_random_instance(
        num_apps = config.num_apps,
        num_ops = config.num_ops,
        num_chains = config.num_chains):
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
            "stake": sample_int(10, 100),
            "fee2gas": sample_int(1, 10)
        }
        apps.append(app)

    ops = []
    for _ in range(num_ops):
        op = {
            "gas": sample_int(10, 100),
            "stake": sample_int(10, 100),
            "fee2gas": sample_int(1, 10)
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
