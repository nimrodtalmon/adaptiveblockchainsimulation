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

    # List of instance generator functions to run
    instance_generators = [
        ("Validation Example 1", instance_generator.generate_validation_example_1),
        ("Validation Example 2", instance_generator.generate_validation_example_2)
    ]

    # Run each instance
    for name, generator in instance_generators:
        print(f"\n>>> {name} <<<\n")
        
        # Generate and print instance
        instance = generator()
        # print_instance(instance)

        # Solve the optimization model
        solution, score, constraints, loss_hist, totvio_hist = solve_model(instance)

        # Print results
        print(f"\nFinal Score for {name}: {score[0]:.6f}")
        # print_solution_with_utilities_and_constraints(solution, solution["utilities"], constraints, instance)
        print("\n" + "="*50)

if __name__ == "__main__":
    main()
