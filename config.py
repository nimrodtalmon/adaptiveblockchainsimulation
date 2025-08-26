# config.py
from typing import Callable, Any
from dataclasses import dataclass
import nevergrad as ng


@dataclass
class SimConfig:
    # Randomness
    random_seed: int = 42

    # Instance parameters
    num_apps: int = 3
    num_ops: int = 3
    num_chains: int = 3

    # Instance generator: "toy_1" or "random"
    instance_generator: str = \
        "generate_toy_instance_1"
        # "generate_random_instance_1"

    # Nevergrad parameters
    nevergrad_optimizer: Callable[..., Any] = ng.optimizers.NgIohTuned
    nevergrad_budget: int = 50
    nevergrad_num_workers: int = 1
