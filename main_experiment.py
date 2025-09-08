# main.py
from config import GeneralConfig, ExperimentConfig; general_config = GeneralConfig(); experiment_config = ExperimentConfig()
import random, numpy as np
from core import instance_generator
from utils.helpers import clear_screen, print_instance, print_solution_with_utilities_and_constraints, take_care_of_random_seed, plot_loss_and_violations_per_iteration
from core.optimizer import solve_model


def main():
    # Clear the screen
    clear_screen()

    # Print hello
    print("\n>>> Adaptive Multichain Blockchain Simulation <<<")
    # Take care for random seed
    take_care_of_random_seed()

    # Step 1: Generate repetitions-many random instances
    all_loss = []
    all_vio  = []

    for i in range(experiment_config.repetitions):
        print(">>> INSTANCE %d"%i)
        instance = instance_generator.generate_instance(experiment_config)
        # print_instance(instance)

        # Step 2: Build and solve the optimization model
        solution, utilities, constraints, loss_hist, totvio_hist = solve_model(instance)
       
        all_loss.append([float(v) for v in loss_hist])
        all_vio.append([float(v) for v in totvio_hist])
     
    # --- aggregate across repetitions (allow different lengths) ---
    max_len = max(len(x) for x in all_loss + all_vio)

    def pad_nan(seq, L):
        return np.asarray(seq + [np.nan] * (L - len(seq)), dtype=float)

    loss_mat = np.vstack([pad_nan(x, max_len) for x in all_loss])   # shape: (R, T)
    vio_mat  = np.vstack([pad_nan(x, max_len) for x in all_vio])    # shape: (R, T)

    avg_loss = np.nanmean(loss_mat, axis=0)  # length T
    avg_vio  = np.nanmean(vio_mat, axis=0)   # length T

    # Show plot of util+violations as a function of iteration num
    plot_loss_and_violations_per_iteration(avg_loss, avg_vio)
if __name__ == "__main__":
    main()
