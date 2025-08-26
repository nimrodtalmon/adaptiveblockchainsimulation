# config.py
from dataclasses import dataclass

@dataclass
class SimConfig:
    # Instance parameters
    num_apps: int = 3
    num_ops: int = 3
    num_chains: int = 3

    # Instance generator: "toy_1" or "random"
    instance_generator: str = "toy_1"

    # Randomness (useful for reproducibility)
    random_seed: int = 42

    # Nevergrad
    # Budget correlates with #improvement_steps
    nevergrad_budget: int = 50
