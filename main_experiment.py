# main.py
from config import GeneralConfig, ExperimentConfig; general_config = GeneralConfig(); experiment_config = ExperimentConfig()
import random, numpy as np
from core import instance_generator
from utils.helpers import clear_screen, print_instance, print_solution_with_utilities, take_care_of_random_seed
from core.optimizer import solve_model


def main():
    # Clear the screen
    clear_screen()

    # Print hello
    print("\n>>> Adaptive Multichain Blockchain Simulation <<<")
    # Take care for random seed
    take_care_of_random_seed()

    # # Step 1: Generate (and print) a random instance
    for i in range(experiment_config.repetitions):
        print("\n>>> INSTANCE %d\n\n"%i)
        instance = instance_generator.generate_instance(experiment_config)
        print_instance(instance)d

        # Step 2: Build and solve the optimization model
        solution, utilities = solve_model(instance)

        # Step 3: Summarize results
        print_solution_with_utilities(solution, utilities)

if __name__ == "__main__":
    main()
