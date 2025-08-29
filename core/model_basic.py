# core/model_basic.py

import nevergrad as ng
import numpy as np


def define_problem(instance):
    """
    Defines the optimization problem:
    - Builds the Nevergrad parametrization (the search space),
    - Returns an evaluation function to score solutions.

    Args:
        instance: dict with 'apps', 'ops', 'chains', 'lambdas'

    Returns:
        parametrization: Nevergrad search space
        evaluate: function to score candidate solutions
    """
    num_apps = len(instance["apps"])
    num_ops = len(instance["ops"])
    num_chains = len(instance["chains"])

    # Define search space
    app_vars = [ng.p.Choice(list(range(-1, num_chains))) for _ in range(num_apps)]
    op_vars = [ng.p.Choice(list(range(-1, num_chains))) for _ in range(num_ops)]
    FEE2GAS_MAX = max([a["fee2gas"] for a in instance["apps"]], default=1.0)
    print('MMMMMMMM:', FEE2GAS_MAX)
    fee2gas_vars = [ng.p.Scalar(lower=0.0, upper=FEE2GAS_MAX) for _ in range(num_chains)]

    parametrization = ng.p.Instrumentation(
        app_assignments=ng.p.Tuple(*app_vars),
        op_assignments=ng.p.Tuple(*op_vars),
        fee2gas_chains=ng.p.Tuple(*fee2gas_vars)
    )

    # Define evaluation function
    def evaluate(*args, **kwargs):
        app_assignments = kwargs["app_assignments"]
        op_assignments = kwargs["op_assignments"]
        fee2gas_chains = kwargs["fee2gas_chains"]
        return -evaluate_utilities(app_assignments, op_assignments, fee2gas_chains, instance)

    return parametrization, evaluate


def evaluate_utilities(app_assignments, op_assignments, fee2gas_chains, instance):
    """
    Computes the utility function to maximize (returned as negative for minimization).
    """
    # Definition of entities
    apps = instance["apps"]
    ops = instance["ops"]
    chains = instance["chains"]
    lambdas = instance["lambdas"]

    # Find out apps/ops on each chain
    apps_on_chain = {c: [] for c in chains}
    ops_on_chain = {c: [] for c in chains}

    for a, c in enumerate(app_assignments):
        if c != -1:
            apps_on_chain[c].append(a)

    for o, c in enumerate(op_assignments):
        if c != -1:
            ops_on_chain[c].append(o)

    # Compute expressions
    demand = {}
    supply = {}
    gas = {}
    stake_chain = {}

    # fee2gas per chain (decision variables)
    fee2gas_chain = {c: float(fee2gas_chains[i]) for i, c in enumerate(chains)}

    for c in chains:
        # 1) Demand_c = Σ_{app ∈ apps_on_chain[c]} gas_app
        demand[c] = sum(apps[a]["gas"] for a in apps_on_chain[c])

        # 2) Supply_c = Σ_{op ∈ ops_on_chain[c]} gas_op
        supply[c] = sum(ops[o]["gas"] for o in ops_on_chain[c])

        # 3) Gas_c = min(Demand_c, Supply_c) ; if Demand_c = 0 or Supply_c = 0 then Gas_c = 0
        gas[c] = min(demand[c], supply[c]) if (demand[c] > 0 and supply[c] > 0) else 0

        # 4) Stake_c = Σ_{op ∈ ops_on_chain[c]} stake_op
        stake_chain[c] = sum(ops[o]["stake"] for o in ops_on_chain[c])

    # Compute hard constraints
    fee2gas_violations = 0
    stake_violations = 0

    # 1) y_{op,c} ⇒ fee2gas_op ≤ Fee2gas_c
    for c in chains:
        for o in ops_on_chain[c]:
            if ops[o]["fee2gas"] > fee2gas_chain[c]:
                fee2gas_violations += 1

    # 1) max{fee2gas_op | y_op,c=1} ≤ fee2gas_chain[c] ≤ min{fee2gas_app | x_app,c=1}
    for c in chains:
        if ops_on_chain[c]:  # lower bound from operators
            lo = max(ops[o]["fee2gas"] for o in ops_on_chain[c])
            if fee2gas_chain[c] < lo:
                fee2gas_violations += 1
        if apps_on_chain[c]:  # upper bound from apps
            hi = min(apps[a]["fee2gas"] for a in apps_on_chain[c])
            if fee2gas_chain[c] > hi:
                fee2gas_violations += 1

    # 2) stake_c ≥ max{stake_app | x_{app,c} = 1}
    for c in chains:
        for a in apps_on_chain[c]:
            if stake_chain[c] < apps[a]["stake"]:
                stake_violations += 1

    # Compute utilities
    # app util
    app_util = []
    for a, c in enumerate(app_assignments):
        if c == -1 or demand[c] == 0:
            app_util.append(0)
        else:
            share = gas[c] / demand[c]
            app_util.append(apps[a]["gas"] * share)

    # op util
    op_util = []
    for o, c in enumerate(op_assignments):
        if c == -1 or not ops_on_chain[c]:
            op_util.append(0)
        else:
            share = 1 / len(ops_on_chain[c])
            fee = fee2gas_chain[c] * gas[c]
            op_util.append((fee * share) / ops[o]["stake"])

    # sys util
    sys_util = sum(fee2gas_chain[c] * gas[c] for c in chains)

    # total util
    total_util = (lambdas["apps"] * sum(app_util) +
             lambdas["ops"] * sum(op_util) +
             lambdas["sys"] * sys_util)

    # score
    PENALTY = 1e15
    return total_util - PENALTY * (fee2gas_violations + stake_violations) 
