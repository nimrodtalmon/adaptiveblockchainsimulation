# main.py
from config import GeneralConfig, OneInstanceConfig; general_config = GeneralConfig(); one_instance_config = OneInstanceConfig
import random, numpy as np
import time
from core import instance_generator
from utils.helpers import clear_screen, print_instance, print_solution_with_utilities_and_constraints, take_care_of_random_seed, plot_loss_and_violations_per_iteration
from core.optimizer import solve_model


def main():
    # Clear the screen
    # clear_screen()

    # Print hello
    print("\n>>> Adaptive Multichain Blockchain Simulation <<<")
    # Take care for random seed
    take_care_of_random_seed()

    # Step 1: Generate (and print) a random instance
    instance = instance_generator.generate_instance(general_config)
    print_instance(instance)

    # Step 2: Build and solve the optimization model
    print("Starting optimization...")
    start_time = time.time()
    solution, utilities, constraints, loss_hist, totvio_hist = solve_model(instance)
    end_time = time.time()

    # Summarize results
    print_solution_with_utilities_and_constraints(solution, utilities, constraints)

    # Show plot of util+violations as a function of iteration num
    plot_loss_and_violations_per_iteration(loss_hist, totvio_hist)
    print(f"Computation time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
