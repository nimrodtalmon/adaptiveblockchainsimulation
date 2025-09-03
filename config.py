# config.py
from typing import Callable, Any
from dataclasses import dataclass
import nevergrad as ng

@dataclass
class GeneralConfig:
    # Randomness
    random_seed: int = 42

    # Nevergrad parameters
    nevergrad_optimizer: Callable[..., Any] = ng.optimizers.NgIohTuned
    nevergrad_budget: int = 5000
    nevergrad_num_workers: int = 1

    # Instance parameters
    num_apps: int = 3
    num_ops: int = 3
    num_chains: int = 3

    # Instance generator: "toy_1" or "random"
    instance_generator: str = \
        "generate_random_instance_1"
        # "generate_toy_instance_1"

@dataclass
class OneInstanceConfig:
    # Instance generator: "toy_1" or "random"
    instance_generator: str = \
        "generate_random_instance_1"
        # "generate_toy_instance_1"

@dataclass
class ExperimentConfig:
    # Name of the experiment
    name: str = "utility_as_a_function_of_budget"

    # How many times to repeat each setting
    repetitions: int = 5
