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

    # Run each instance
    for scale in [10]:
        print(f"\n>>> Running an instance of scale: {scale} <<<\n")
        
        # Generate an instance of scale
        instance = instance_generator.generate_random_instance_1(
            num_apps=scale, 
            num_ops=scale, 
            num_chains=int(3))
        
        # Solve the optimization model
        solution, score, constraints, loss_hist, totvio_hist = solve_model(instance, real_budget=100)

        # Print results
        print(f"\nFinal Score for {scale}: {score[0]:.6f}")
        print_solution_with_utilities_and_constraints(solution, solution["utilities"], constraints, instance)
        print("\n" + "="*50)

if __name__ == "__main__":
    main()
