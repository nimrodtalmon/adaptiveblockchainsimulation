# config.py

SIM_CONFIG = {
    # Instance parameters
    ### Pick #apps, #ops, and #chains
    "num_apps": 3,
    "num_ops": 3,
    "num_chains": 3,

    # Instance generator
    ### Choose between "toy_1" and "random"
    "instance_generator": "toy_1",

    # Randomness
    ### This is useful for reproducibility
    "random_seed": 42,

    # Nevergrad
    ### Configuration for Nevergrad solver
    ### Budget correlates with #improvement_steps
    "nevergrad_budget": 50
}