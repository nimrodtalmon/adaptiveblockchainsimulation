# main.py
import nevergrad as ng
import random, numpy as np
from core import instance_generator
from core.optimizer import solve_model
from utils.helpers import clear_screen, print_optimal_utility

def main():
    # Clear the screen
    clear_screen()

    # Print hello
    print("\n>>> Adaptive Multichain Blockchain Simulation <<<")
    # Take care for random seed
    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    # Do several examples (by IDs)
    scores = {}
    for example_id in [1, 2]:
        # Step 1: Generate instance
        if example_id == 1:
            instance = instance_generator.generate_validation_example_1()
        elif example_id == 2:
            instance = instance_generator.generate_validation_example_2()
        print(f"\n=== Example ID: {example_id} ===")
        # print_instance(instance)

        # Step 2: Build and solve the optimization model
        solution, score, constraints, loss_hist, totvio_hist = solve_model(instance)

        # Step 3: Summarize results
        # print_optimal_utility(solution, solution["utilities"], constraints, instance)
        scores[example_id] = score[0]
    for id, score in scores.items():
        print(f"Example ID {id}: Score = {score:.3f}")

if __name__ == "__main__":
    main()
