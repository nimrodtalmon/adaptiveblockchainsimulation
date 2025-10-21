# main.py
import nevergrad as ng
import random, numpy as np
from core import instance_generator
from core.optimizer import solve_model
from utils.helpers import clear_screen, print_instance, print_solution_with_utilities_and_constraints

def main():
    # Clear the screen
    clear_screen()

    # Print hello
    print("\n>>> Adaptive Multichain Blockchain Simulation <<<")
    # Take care for random seed
    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    repetitions = 1

    # # Step 1: Generate (and print) a random instance
    for i in range(repetitions):
        print("\n>>> INSTANCE %d\n\n"%i)
        instance = instance_generator.generate_toy_instance_0()
        print_instance(instance)

        # Step 2: Build and solve the optimization model
        solution, score, constraints, loss_hist, totvio_hist = solve_model(instance)

        # Step 3: Summarize results
        print_solution_with_utilities_and_constraints(solution, solution["utilities"], constraints, instance)

if __name__ == "__main__":
    main()
