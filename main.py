# main.py

import random, numpy as np
from config import SIM_CONFIG
from utils.helpers import clear_screen, print_instance, print_solution_with_utilities, take_care_of_random_seed
from core import instance_generator
from core.optimizer import solve_model


def main():
    # Clear the screen
    clear_screen()
    # Take care for random seed
    take_care_of_random_seed()

    print("X MARKS THE SPOT")
    print("(1) refactor the instance generator (and also the config file to switch)")
    print("(2) refactor this TODO thing ;)")
    print("(3) solve toy_1 by hand and see that we get it")
    print("(4) add a way to see the action of Nevergrad as it converges..")
    print("(5) continue with the simulation work plan")
    exit()

    # Print hello
    print("\n>>> Adaptive Multichain Blockchain Simulation <<<")

    # # Step 1: Generate (and print) a random instance
    if SIM_CONFIG["instance_generator"] == "random":
        instance = instance_generator.generate_random_instance()
    elif SIM_CONFIG["instance_generator"] == "toy_1":
        instance = instance_generator.generate_toy_instance_1()
    print_instance(instance)

    # Step 2: Build and solve the optimization model 
    solution, utilities = solve_model(instance, budget=SIM_CONFIG["nevergrad_budget"])

    # Step 4: Summarize results
    print_solution_with_utilities(solution, utilities)

if __name__ == "__main__":
    main()
