# config.py
from typing import Callable, Any
from dataclasses import dataclass
import nevergrad as ng
# from alt_opt import MyAdaptiveDiscretePortfolio
# import alt_opt.adaptive_portfolio_opt4 as altopt
# import alt_opt.adaptive_portfolio_opt3 as altopt
                                           

@dataclass
class GeneralConfig:
    # Randomness
    random_seed: int = -1

    # Nevergrad parameters
    # nevergrad_optimizer: Callable[..., Any] = ng.optimizers.NgIohTuned
    # nevergrad_optimizer: Callable[..., Any] = ng.optimization.optimizerlib.ParametrizedCMA 
    nevergrad_optimizer: Callable[..., Any] = ng.optimization.optimizerlib.ParametrizedOnePlusOne
    # nevergrad_optimizer: Callable[..., Any] =  ng.optimizers.NGOpt
    # nevergrad_optimizer: Callable[..., Any] = altopt.MyAdaptiveDiscretePortfolio
    
    nevergrad_budget: int = 5000
    nevergrad_num_workers: int = 1

    # Instance parameters
    num_apps: int = 10
    num_ops: int = 3
    num_chains: int = 3

    # Instance generator: "toy_1" or "random"
    instance_generator: str = \
        "generate_random_instance_1"
        # "generate_simplex_instance_1"
        # "generate_toy_instance_2"
        # "generate_random_instance_1"
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
    repetitions: int = 1
