# main.py

import random, numpy as np
from config import SimConfig; config = SimConfig()
from utils.helpers import clear_screen, print_instance, print_solution_with_utilities, take_care_of_random_seed
from core import instance_generator
from core.optimizer import solve_model


def main():
    # Clear the screen
    clear_screen()
    # Take care for random seed
    take_care_of_random_seed()

    # Print hello
    print("\n>>> Adaptive Multichain Blockchain Simulation <<<")

    # # Step 1: Generate (and print) a random instance
    if config.instance_generator == "random":
        instance = instance_generator.generate_random_instance()
    elif config.instance_generator == "toy_1":
        instance = instance_generator.generate_toy_instance_1()
    print_instance(instance)

    # Step 2: Build and solve the optimization model 
    solution, utilities = solve_model(instance, budget=config.nevergrad_budget)

    # Step 4: Summarize results
    print_solution_with_utilities(solution, utilities)

if __name__ == "__main__":
    main()
