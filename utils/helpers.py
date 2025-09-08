# utils/helpers.py

import os
import random
import numpy as np
from typing import Dict, Tuple
import matplotlib.pyplot as plt
from config import GeneralConfig; general_config = GeneralConfig()


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear && printf "\\033[3J"')


def take_care_of_random_seed():
    if general_config.random_seed == -1:
        # pick a random seed
        seed = random.randint(0, 2**32 - 1)
        print(f"Random seed: {seed}")
    else:
        seed = general_config.random_seed
        print(f"Random seed: {seed}")

    random.seed(seed)
    np.random.seed(seed)


def sample_int(low: int, high: int) -> int:
    """
    Inclusive integer sampler.
    """
    return int(random.randint(low, high))


def generate_random_lambdas(keys: Tuple[str, str, str] = ("apps", "ops", "sys")) -> Dict[str, float]:
    """
    Sample a random convex combination over the given keys (i.e., Dirichlet(1,...,1)).
    Ensures values are >=0 and sum to 1 (up to float precision).
    """
    raw = [random.random() for _ in keys]     # each in [0,1)
    total = sum(raw) or 1.0                   # guard against pathological zero (theoretically impossible)
    return {k: v / total for k, v in zip(keys, raw)}


def print_instance(instance, max_rows=5):
    """
    Pretty-print an instance (apps, ops, chains).
    Limits output for readability.
    """
    print("\n=== Instance ===")

    print(f"Applications ({len(instance['apps'])} total):")
    for i, app in enumerate(instance["apps"][:max_rows]):
        print(f"  App {i}: gas={app['gas']:.1f}, stake={app['stake']:.1f}, fee2gas={app['fee2gas']:.1f}")
    if len(instance["apps"]) > max_rows:
        print(f"  ... ({len(instance['apps']) - max_rows} more)")

    print(f"\nOperators ({len(instance['ops'])} total):")
    for i, op in enumerate(instance["ops"][:max_rows]):
        print(f"  Op {i}: gas={op['gas']:.1f}, stake={op['stake']:.1f}, fee2gas={op['fee2gas']:.1f}")
    if len(instance["ops"]) > max_rows:
        print(f"  ... ({len(instance['ops']) - max_rows} more)")

    print(f"\nChains: {len(instance['chains'])} (IDs: {instance['chains'][:max_rows]})")
    if len(instance["chains"]) > max_rows:
        print(f"  ... ({len(instance['chains']) - max_rows} more)")


def plot_loss_and_violations_per_iteration(loss_hist, totvio_hist):
    # === Combined plot ===
    plt.figure()
    plt.plot(loss_hist, label="Loss")
    plt.plot(totvio_hist, label="Total violation")
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.title("Loss & Constraint Violation per Iteration")
    plt.legend()
    plt.tight_layout()
    plt.show()

def print_solution_with_utilities_and_constraints(solution, utilities, constraints):
    """
    Pretty-print a solution with utility breakdown.
    """
    print("\n=== Solution ===")

    apps = solution["app_assignments"]
    ops = solution["op_assignments"]
    fee2gas_chains = solution["fee2gas_chains"]

    print("\nApp Assignments:")
    for a, c in enumerate(apps):
        print(f"  App {a} → Chain {c}" if c != -1 else f"  App {a} → Unassigned")

    print("\nOperator Assignments:")
    for o, c in enumerate(ops):
        print(f"  Op {o} → Chain {c}" if c != -1 else f"  Op {o} → Unassigned")

    print("\nChain fee2gas values:")
    for c, fee in enumerate(fee2gas_chains):
        print(f"  Chain {c}: fee2gas = {fee:.3f}")

    print("\nUtilities:")
    print(f"  Total (weighted):    {utilities:.3f}")

    print("\nConstraints:")
    print(f"  Violations: {constraints}")
