# core/model_basic.py

import nevergrad as ng
import numpy as np

from utils.helpers import get_fee2gas_chain #Haim-tmp


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
        constraints: constraint violations
    """
    num_apps = len(instance["apps"])
    num_ops = len(instance["ops"])
    num_chains = len(instance["chains"])

    # Define search space
    app_vars = [ng.p.Choice(list(range(-1, num_chains))) for _ in range(num_apps)]
    op_vars = [ng.p.Choice(list(range(-1, num_chains))) for _ in range(num_ops)]

    """
    # need only when clearing proce ia give tpo NG as a decision varaible

    FEE2GAS_MAX = max([a["fee2gas"] for a in instance["apps"]])
    # fee2gas_vars = [ng.p.Scalar(lower=0.0, upper=FEE2GAS_MAX) for _ in range(num_chains)]
    FEE2GAS_MIN = min([o["fee2gas"] for o in instance["ops"]]) #Haim
    #fee2gas_vars = [ng.p.Scalar(lower=FEE2GAS_MIN, upper=FEE2GAS_MAX) for _ in range(num_chains)] #Haim
    """

    parametrization = ng.p.Instrumentation(
        app_assignments=ng.p.Tuple(*app_vars),
        op_assignments=ng.p.Tuple(*op_vars),
        fee2gas_chains=10.0  # Haim - dummy value, will flow back and forth with no. Kept to minimize changes.
        #fee2gas_chains=ng.p.Tuple(*fee2gas_vars)

    )

    # get adaptive penalty style
    # penalty_style =calibrate_penalty_style(evaluate, constraints, parametrization, instance)

    # Define evaluation function
    def evaluate(*args, **kwargs):
        app_assignments = kwargs["app_assignments"]
        op_assignments = kwargs["op_assignments"]
        fee2gas_chains = kwargs["fee2gas_chains"]
        return -evaluate_utilities(app_assignments, op_assignments, fee2gas_chains, instance)[0]
    
    # Define constraints function
    def constraints(*args, **kwargs):
        app_assignments = kwargs["app_assignments"]
        op_assignments = kwargs["op_assignments"]
        fee2gas_chains = kwargs["fee2gas_chains"]
        return evaluate_constraints(app_assignments, op_assignments, fee2gas_chains, instance)
    
    return parametrization, evaluate, constraints


def define_problem_2(instance):
    """
    Defines the optimization problem:
    - Builds the Nevergrad parametrization (the search space),
    - Returns an evaluation function to score solutions.

    Args:
        instance: dict with 'apps', 'ops', 'chains', 'lambdas'

    Returns:
        parametrization: Nevergrad search space
        evaluate: function to score candidate solutions
        constraints: constraint violations
    """
    num_apps = len(instance["apps"])
    num_ops = len(instance["ops"])
    num_chains = len(instance["chains"])

    """
    Create a partially-symmetry-free parametrization for assigning num_apps apps and num_ops ops 
    into at most num_chains chains.
    """
    # total resources
    total = num_apps + num_ops


    # For each app and op, Choice of chain IDs:
    # Apps and ops are joined together into a single Choice list
    # Resource i (either app or op) can go into any "existing" chain (chain 0..i-1) or "new" chain i, but not beyond num_chains -1

    assignments = []
    for i in range(total):
        # max chain index available at this step
        max_new_chain = min(i, num_chains - 1)
        choices = list(range(-1, max_new_chain + 1))
        assignments.append(ng.p.Choice(choices))

    # To avoid code changes in other places, soplitting the created list of assignment Choices into a list for apps and a list of ops.
    app_vars = assignments[0:num_apps]  
    op_vars = assignments[num_apps:total]  
    parametrization = ng.p.Instrumentation(
        app_assignments=ng.p.Tuple(*app_vars),
        op_assignments=ng.p.Tuple(*op_vars),
        fee2gas_chains=10.0  # Haim - dummy value, will do nothing. Kept as a dummy decision variuabled to minimize changes.
    )
    
    # Alternatively, transfrer all assignment as a single tuple to NG. Wrap in a Dict so we can separate apps and ops later. 
    # This requires code changes. Have a separate funvction to decode into apps and ops assignments from that single tupple.
    # parametrization = ng.p.Dict(assignments=ng.p.Tuple(*assignments))

    # Define evaluation function
    def evaluate(*args, **kwargs):
        app_assignments = kwargs["app_assignments"]
        op_assignments = kwargs["op_assignments"]
        fee2gas_chains = kwargs["fee2gas_chains"]
        return -evaluate_utilities(app_assignments, op_assignments, fee2gas_chains, instance)[0]
    
    # Define constraints function
    def constraints(*args, **kwargs):
        app_assignments = kwargs["app_assignments"]
        op_assignments = kwargs["op_assignments"]
        fee2gas_chains = kwargs["fee2gas_chains"]
        return evaluate_constraints(app_assignments, op_assignments, fee2gas_chains, instance)
    
    return parametrization, evaluate, constraints


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
    # fee2gas_chain = {c: float(fee2gas_chains[i]) for i, c in enumerate(chains)} 
    fee2gas_chain = get_fee2gas_chain(chains, ops, apps, apps_on_chain, ops_on_chain) #Haim-tmp
    

    for c in chains:
        # 1) Demand_c = Σ_{app ∈ apps_on_chain[c]} gas_app
        demand[c] = sum(apps[a]["gas"] for a in apps_on_chain[c])

        # 2) Supply_c = Σ_{op ∈ ops_on_chain[c]} gas_op
        #supply[c] = sum(ops[o]["gas"] for o in ops_on_chain[c])
        supply[c] = min(ops[o]["gas"] for o in ops_on_chain[c]) if ops_on_chain[c] else 0


        # 3) Gas_c = min(Demand_c, Supply_c) ; if Demand_c = 0 or Supply_c = 0 then Gas_c = 0
        gas[c] = min(demand[c], supply[c]) if (demand[c] > 0 and supply[c] > 0) else 0

        # 4) Stake_c = Σ_{op ∈ ops_on_chain[c]} stake_op
        stake_chain[c] = sum(ops[o]["stake"] for o in ops_on_chain[c])

    # Compute utilities

    # First, some upper bounds for normalization
    total_gas_supply = sum(op["gas"] for op in ops)
    max_fee2gas = max(app["fee2gas"] for app in apps)
    Qmax_opsys = total_gas_supply * max_fee2gas + 1e-12

    # app util
    app_util = []
    for a, c in enumerate(app_assignments):
        if c == -1 or demand[c] == 0:
            app_util.append(0)
        else:
            share = gas[c] / demand[c]
            app_util.append(share) # note that it is already normalized

    # op util
    op_util = []
    for o, c in enumerate(op_assignments):
        if c == -1 or not ops_on_chain[c]:
            op_util.append(0)
        else:
            share = 1 / len(ops_on_chain[c])
            fee = fee2gas_chain[c] * gas[c] / Qmax_opsys

            op_util.append((fee * share) / ops[o]["stake"])

    # sys util
    sys_util = sum(fee2gas_chain[c] * gas[c] for c in chains) / Qmax_opsys

    # total util
    total_util = (lambdas["apps"] * sum(app_util) / len(apps) +
             lambdas["ops"] * sum(op_util) / len(ops) +
             lambdas["sys"] * sys_util)

    # score
    return [total_util, app_util, op_util, sys_util]


def evaluate_constraints(app_assignments, op_assignments, fee2gas_chains, instance):
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
    #fee2gas_chain = {c: float(fee2gas_chains[i]) for i, c in enumerate(chains)} 
    fee2gas_chain = get_fee2gas_chain(chains, ops, apps, apps_on_chain, ops_on_chain) #Haim-tmp

    #for c, f in fee2gas_chain.items():
    #    if f <= 0: 
    #        print(f"\nWarning: fee2gas_chain[{c}] = {f} <= 0")


    for c in chains:
        # 1) Demand_c = Σ_{app ∈ apps_on_chain[c]} gas_app
        demand[c] = sum(apps[a]["gas"] for a in apps_on_chain[c])

        # 2) Supply_c = Σ_{op ∈ ops_on_chain[c]} gas_op
        #supply[c] = sum(ops[o]["gas"] for o in ops_on_chain[c])
        supply[c] = min(ops[o]["gas"] for o in ops_on_chain[c]) if ops_on_chain[c] else 0 

        # 3) Gas_c = min(Demand_c, Supply_c) ; if Demand_c = 0 or Supply_c = 0 then Gas_c = 0
        gas[c] = min(demand[c], supply[c]) if (demand[c] > 0 and supply[c] > 0) else 0

        # 4) Stake_c = Σ_{op ∈ ops_on_chain[c]} stake_op
        stake_chain[c] = sum(ops[o]["stake"] for o in ops_on_chain[c])

    # Compute hard constraints
    fee2gas_violations = 0
    stake_violations = 0

    # Haim-full constraints violatons support 
    f_fee2gas_violations = []
    f_stake_violations = []


    # 1) y_{op,c} ⇒ fee2gas_op ≤ Fee2gas_c
    #for c in chains:
    #    for o in ops_on_chain[c]:
    #        if ops[o]["fee2gas"] > fee2gas_chain[c]:
    #            fee2gas_violations += 1

    # 1) max{fee2gas_op | y_op,c=1} ≤ fee2gas_chain[c] ≤ min{fee2gas_app | x_app,c=1}
    for c in chains:
        if ops_on_chain[c]:  # lower bound from operators
            if apps_on_chain[c]:
                lo = max(ops[o]["fee2gas"] for o in ops_on_chain[c])
                if fee2gas_chain[c] < lo:
                    fee2gas_violations += 1
                    f_fee2gas_violations.append(lo-fee2gas_chain[c]) # By Haim
            
        if apps_on_chain[c]:  # upper bound from apps
            if ops_on_chain[c]:
                hi = min(apps[a]["fee2gas"] for a in apps_on_chain[c])
                if fee2gas_chain[c] > hi:
                    fee2gas_violations += 1
                    f_fee2gas_violations.append(fee2gas_chain[c]-hi) # By Haim

    # 2) stake_c ≥ max{stake_app | x_{app,c} = 1}
    for c in chains:
        if not ops_on_chain[c]:
            continue
        for a in apps_on_chain[c]:
            if stake_chain[c] < apps[a]["stake"]:
                stake_violations += 1
                f_stake_violations.append(apps[a]["stake"]-stake_chain[c])
                #app_stake = apps[a]["stake"]
                #print(f"\nStake violation, c: {c}, stake_chain[c]: {stake_chain[c]}, a: {a}, apps[a] stake: {app_stake}")


    # score
    #return [fee2gas_violations, stake_violations] 
    return [f_fee2gas_violations, f_stake_violations] 


# using a warm up with sampling from the parametrization to obtain a penalty_style constant vector (alternative to the dfefault one).
def calibrate_penalty_style(utility_eval, constraint_eval, parametrization, instance, warmup=50, verbose=True):
    """
    Calibrate balanced penalty constants (a,b,c,d,e,f) from warm-up samples.
    
    - evaluator: function(cand) -> (loss, violations)
    - parametrization: NG parameterization (Tuple of Choices, etc.)
    - warmup: number of random samples to draw
    - verbose: if True, print warm-up statistics
    
    Returns:
      (a,b,c,d,e,f), stats_dict
    """
    feasible_values, infeasible_viols = [], []
    all_values = []

    for _ in range(warmup):
        cand = parametrization.sample()
        val = utility_eval(**cand.kwargs)
        violations = constraint_eval(**cand.kwargs)
        val = float(val)
        all_values.append(val)

        v_sum = sum(max(0, v) for v in violations)
        if v_sum > 0:
            infeasible_viols.append(v_sum)
        else:
            feasible_values.append(val)

    # --- Loss scale (S_L) ---
    if feasible_values:
        S_L = np.median(np.abs(feasible_values))
    else:
        S_L = np.median(np.abs(all_values)) if all_values else 1.0
    if S_L <= 0:
        S_L = 1.0   # defensive fallback

    # --- Violation scale (S_V) ---
    S_V = np.median(infeasible_viols) if infeasible_viols else 1.0

    # --- Base term a ---
    a = 3 * S_L

    # --- Recommended balanced parameters ---
    c = 0.5   # dampening exponent for violations
    d = 1.0   # linear aggregation after root
    b = S_L / (S_V ** c) if S_V > 0 else S_L
    e = 0.15  # mild tightening with iterations
    f = 20.0

    penalty_style = (a, b, c, d, e, f)

    stats = {
        "warmup_samples": warmup,
        "num_feasible": len(feasible_values),
        "num_infeasible": len(infeasible_viols),
        "median_abs_feasible_value": np.median(np.abs(feasible_values)) if feasible_values else None,
        "median_abs_all_value": np.median(np.abs(all_values)) if all_values else None,
        "median_violation": S_V,
        "S_L": S_L,
        "S_V": S_V,
        "penalty_style": penalty_style,
    }

    if verbose:
        print("Warm-up summary:")
        for k,v in stats.items():
            print(f"  {k}: {v}")

    return penalty_style, stats
