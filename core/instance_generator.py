# core/instance_generator.py

import random
from typing import Dict, Tuple

from config import SIM_CONFIG

def _sample_int(low: int, high: int) -> int:
    """
    Inclusive integer sampler.
    """
    return int(random.randint(low, high))

def _generate_random_lambdas(keys: Tuple[str, str, str] = ("apps", "ops", "sys")) -> Dict[str, float]:
    """
    Sample a random convex combination over the given keys (i.e., Dirichlet(1,...,1)).
    Ensures values are >=0 and sum to 1 (up to float precision).
    """
    raw = [random.random() for _ in keys]     # each in [0,1)
    total = sum(raw) or 1.0                   # guard against pathological zero (theoretically impossible)
    return {k: v / total for k, v in zip(keys, raw)}

def generate_toy_instance_1():
    """
    Returns a small, fixed instance for manual verification.

    Instance details:
      Chains: 1 (ID 0)
      Apps:
        App0: gas=10, stake=5, fee2gas=2
        App1: gas=20, stake=8, fee2gas=3
      Ops:
        Op0: gas=15, stake=6, fee2gas=1
        Op1: gas=25, stake=7, fee2gas=2
      Lambdas: apps=0.5, ops=0.3, sys=0.2

    Optimal assignment (by inspection):
      - Assign both apps to Chain0
      - Assign both ops to Chain0

    Reasoning:
      Demand_chain0 = 10 + 20 = 30
      Supply_chain0 = 15 + 25 = 40
      Gas_chain0 = min(30, 40) = 30
      Stake_chain0 = 6 + 7 = 13
      Fee2gas_chain0 = min(2, 3) = 2
      Fee_chain0 = 2 * 30 = 60

      App utilities:
        App0 = gas_app0 * Gas/Demand = 10 * (30/30) = 10
        App1 = 20 * (30/30) = 20
        sum = 30

      Op utilities:
        Each op gets share = 1 / 2 = 0.5
        Fee per op share = 60 * 0.5 = 30
        Op0 util = 30 / stake0 = 30 / 6 = 5
        Op1 util = 30 / 7 ≈ 4.2857
        sum ≈ 9.2857

      Sys utility:
        Sys = Fee_chain0 = 60

      Weighted total utility:
        = 0.5 * 30 + 0.3 * 9.2857 + 0.2 * 60
        = 15 + 2.7857 + 12
        ≈ 29.7857
    """
    return {
        "apps": [
            {"gas": 10, "stake": 5, "fee2gas": 2},
            {"gas": 20, "stake": 8, "fee2gas": 3},
        ],
        "ops": [
            {"gas": 15, "stake": 6, "fee2gas": 1},
            {"gas": 25, "stake": 7, "fee2gas": 2},
        ],
        "chains": [0],
        "lambdas": {"apps": 0.5, "ops": 0.3, "sys": 0.2},
    }


def generate_random_instance(
        num_apps = SIM_CONFIG["num_apps"],
        num_ops = SIM_CONFIG["num_ops"],
        num_chains = SIM_CONFIG["num_chains"]):
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
            "gas": _sample_int(10, 100),
            "stake": _sample_int(10, 100),
            "fee2gas": _sample_int(1, 10)
        }
        apps.append(app)

    ops = []
    for _ in range(num_ops):
        op = {
            "gas": _sample_int(10, 100),
            "stake": _sample_int(10, 100),
            "fee2gas": _sample_int(1, 10)
        }
        ops.append(op)

    chains = list(range(num_chains))

    # Random convex combination for lambdas
    lambdas = _generate_random_lambdas(("apps", "ops", "sys"))

    return {
        "apps": apps,
        "ops": ops,
        "chains": chains,
        "lambdas": lambdas
    }
