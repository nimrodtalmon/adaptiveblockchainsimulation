# utils/helpers.py

import os
import random
import numpy as np
from typing import Dict, Tuple
from config import GeneralConfig; general_config = GeneralConfig()

import nevergrad as ng  # Haim


def find_steady_state(loss_history, violation_history, window=150,
                      loss_std_tol=5e-3, mean_delta_tol=1e-3, viol_eps=1e-12):
    """
    Returns first 1-based iteration (budget) where:
      - last `window` iterations have zero violations, and
      - utility stability: rolling std <= loss_std_tol, and
      - mean-drift between consecutive windows <= mean_delta_tol.
    If not found, returns None.
    """
    L = np.asarray(loss_history, dtype=float)
    V = np.asarray(violation_history, dtype=float)
    n = len(L)
    if n < window or len(V) != n:
        return None

    util = -L  # utility is negative loss

    # Rolling sums to check zero violations in window
    viol_mask = (V > viol_eps).astype(int)
    viol_win = np.convolve(viol_mask, np.ones(window, dtype=int), mode="valid")  # length n-window+1
    zero_viol = (viol_win == 0)

    # Rolling means and stds for utility
    c1 = np.cumsum(np.insert(util, 0, 0.0))
    c2 = np.cumsum(np.insert(util*util, 0, 0.0))
    win_mean = (c1[window:] - c1[:-window]) / window
    win_mean_prev = np.concatenate([[win_mean[0]], win_mean[:-1]])  # shifted
    win_var = (c2[window:] - c2[:-window]) / window - win_mean**2
    win_std = np.sqrt(np.maximum(win_var, 0.0))

    stable_std = (win_std <= loss_std_tol)
    stable_drift = (np.abs(win_mean - win_mean_prev) <= mean_delta_tol)

    ok = zero_viol & stable_std & stable_drift
    if not np.any(ok):
        return None

    i_valid = int(np.argmax(ok))           # index in the 'valid' range
    steady_iter_0based = i_valid + window - 1
    return steady_iter_0based + 1          # 1-based budget


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
    return seed #Haim


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


def print_instance(instance, max_rows=10):
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

    print(f"\nlambdas ({len(instance['lambdas'])}):")
    for comp, lamb in instance["lambdas"].items():
        print(f"{comp} = {lamb:.1f} ")
   

# def print_solution_with_utilities_and_constraints(solution, utilities, constraints): 
def print_solution_with_utilities_and_constraints(solution, utilities, constraints, instance):  #Haim
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
    # for c, fee in enumerate(fee2gas_chains):
    #    print(f"  Chain {c}: fee2gas = {fee:.3f}")
    fee2gas_chain = get_fee2gas_solution_chain(solution, instance)
    for i, (c, fee) in enumerate(fee2gas_chain.items()):
        print(f"  Chain {c}: fee2gas = {fee:.3f}")
    

    print("\nUtilities:")
    print(f"  Total (weighted):    {utilities:.3f}")

    print("\nConstraints:")
    print(f"  Violations: {constraints}")



def get_fee2gas_solution_chain(solution, instance):
    # Definition of entities
    apps = instance["apps"]
    ops = instance["ops"]
    chains = instance["chains"]

    # Find out apps/ops on each chain
    app_assignments = solution["app_assignments"]
    op_assignments = solution["op_assignments"]

    apps_on_chain = {c: [] for c in chains}
    ops_on_chain = {c: [] for c in chains}

    for a, c in enumerate(app_assignments):
        if c != -1:
             apps_on_chain[c].append(a)

    for o, c in enumerate(op_assignments):
        if c != -1:
            ops_on_chain[c].append(o)

    fee2gas_chain = get_fee2gas_chain(chains, ops, apps, apps_on_chain, ops_on_chain) #Haim-tmp
    return fee2gas_chain


def get_fee2gas_chain(chains, ops, apps, apps_on_chain, ops_on_chain):
    lo = []
    hi = []
    for c in chains:
        if apps_on_chain[c]:
            lo.append (min(apps[a]["fee2gas"] for a in apps_on_chain[c]))
        else:
            lo.append(0.0)
        if ops_on_chain[c]:
            hi.append(max(ops[o]["fee2gas"] for o in ops_on_chain[c]))
        else:
           hi.append(0.0) 
    fee2gas_chain = {c: lo[i]+(hi[i]-lo[i])/2 if hi[i] > 0 and lo[i] > 0 else 0.0 for i, c in enumerate(chains)}
    return fee2gas_chain


# added by Haim
def print_utility_improvement (budget, loss_hist, totvio_hist):
    print(f" Utility improvements")
    bestUtilHist = {}
    bestUtil = 0
    for i in range(budget):   # Haim
        if totvio_hist[i] == 0:
            if loss_hist[i] < bestUtil:
                bestUtil = loss_hist[i]
                bestUtilHist.update({i+1 : loss_hist[i]})
                print(f" iter= [{i+1:04d}],  utility= {bestUtil:.6f} ")


# added by Haim: print constraint handler data of the base optimizaer employed
# Data_from = "source": the current sources are anylyzd for defaults. Use when the data isn't available / accessible on the optimizer, 
# or if the global constraint_handler can't be accessed
# Data_source = "from_opt":  Looking for data on the optimizer-embedded comstraint handler. However, this isn't possible for all optimizers.
# Typically supported with modern optimizers
def print_constraint_penalty_constants(opt, data_from = "source"):
    if data_from=="from_source":
        # Assuming an optimizer that uses the penalty formula but has no embedded conmstraint handler, extract default penalty constants
        # from the source. 
        # Get the tell method's source code to extract the default penalty constants
        import inspect
        import re
        tell_source = inspect.getsource(opt.tell)        

        # Look for the line: a, b, c, d, e, f = (1e5, 1.0, 0.5, 1.0, 0.5, 1.0)
        penalty_match = re.search(r'a,\s*b,\s*c,\s*d,\s*e,\s*f\s*=\s*\(([^)]+)\)', tell_source)
        if penalty_match:
            penalty_values_str = penalty_match.group(1)
            # Split and clean the values
            values = [val.strip() for val in penalty_values_str.split(',')]
            letters = ['a', 'b', 'c', 'd', 'e', 'f']
            print("Default penalty constants:")
            for letter, value in zip(letters, values):
                print(f"  {letter} = {value}")
        else:
            print("Could not extract penalty constants from source")
    elif data_from =="from_opt":
        # Looking for the optimizer-embedded comstraint handler

        # Force it to initialize (otherwise it hasn’t picked a sub-optimizer yet)
        cand = opt.ask()
        opt.tell(cand, 0.0)  

        # dig down recursively
        def get_base_optimizers(optim):
            # if this optimizer has a constraint handler, we're at the base
            if hasattr(optim, "_constraint_handler"):
                return [optim]

            children = []
            # Look for child optimizers under different names
            for attr in ["optim", "optimizers", "_optim", "_optimizers"]:
                if hasattr(optim, attr):
                    obj = getattr(optim, attr)
                    if isinstance(obj, (list, tuple)):
                        for o in obj:
                            children.extend(get_base_optimizers(o))
                    else:
                        children.extend(get_base_optimizers(obj))
            return children

        base_opts = get_base_optimizers(opt)

        for b in base_opts:
            print(type(b))
            handler = getattr(b, "_constraint_handler", None)
            if handler is not None:
                print(handler.__dict__)

    