# core/optimizer.py

import nevergrad as ng
from core.model_basic import define_problem, define_problem_2
from config import GeneralConfig; general_config = GeneralConfig()
import matplotlib.pyplot as plt
import utils.helpers as hlp
from core.model_basic import evaluate_utilities, evaluate_constraints
from core.canonicalized_evaluator import EvaluatorWithStats

# import warnings #Haim
# warnings.filterwarnings("ignore", message="elements of z2") #Haim occur when usng NgOpt


def solve_model(instance, verbose=True):
    """
    Solves the model defined in model_basic.py using Nevergrad (NGOpt solver).
    
    Args:
        instance: instance
        budget: number of function evaluations
        verbose: whether to print intermediate output

    Returns:
        best_solution: dict with 'app_assignments' and 'op_assignments'
        score: total utility value
    """
    # Build search space and objective
    parametrization, evaluate, constraints = define_problem_2(instance) #Haim

    # Set up Nevergrad optimizer - use for non-factory-indstantiated optimizers
    #opt = general_config.nevergrad_optimizer(
    #    parametrization=parametrization, 
    #    budget=general_config.nevergrad_budget,
    #    num_workers=general_config.nevergrad_num_workers)
    
    # Set up Nevergrad optimizer - use for factory-instantiated optimizers
    opt = general_config.nevergrad_optimizer()(
          parametrization=parametrization, 
          budget=general_config.nevergrad_budget,
          num_workers=general_config.nevergrad_num_workers)
    parametrization.random_state.seed(hlp.take_care_of_random_seed()) #Haim
    opt._rng.seed(hlp.take_care_of_random_seed()) #Haim
   
   
    if False:
        print("\n=== Solver ===")
        print(f"[Nevergrad] {general_config.nevergrad_optimizer.__name__} | "
              f"budget={general_config.nevergrad_budget} | num_workers={general_config.nevergrad_num_workers}")
        # Added by Haim
        print(opt.__class__.__module__, opt.__class__.__name__)
        hlp.print_constraint_penalty_constants(opt)

        """
        print("Nevergrad version:", ng.__version__)
        print("\nAvailable optimizers with 'Portfolio' or 'Discrete':")
        for name in sorted(ng.optimizers.registry.keys()):
            if 'Portfolio' in name or 'Discrete' in name:
                print(f"  - {name}")
        """
    
    # Tracking
    loss_hist = []
    totvio_hist = []
    stats = EvaluatorWithStats(len(instance["apps"]), len(instance["ops"]), evaluate, evaluate_utilities, constraints, evaluate_constraints)

    # Run optimization
    for it in range(opt.budget):
        # ask
        cand = opt.ask()
        # get value and vlist (and then total_vio)
        # value = evaluate(**cand.kwargs) #Haim - changed to call the canonicalizd evaluator (below)
        value = stats.evaluate_cannocalized(**cand.kwargs) #Haim 
        # vlist = constraints(**cand.kwargs)  #Haim - changed to call the canonicalized constraints computation (below)
        vlist = stats.constraints_canonicalized(**cand.kwargs) 
        #total_vio = float(sum(vlist)) 
        total_vio = float(sum(vlist[0]) + sum(vlist[1])) #Haim - vlist is now a list of lists
        #if (total_vio > 0):
        #    print(f"\nvlist: {vlist}")
        #    print(f"\ntotal_vio: {total_vio}, kwargs = {cand.kwargs}")
            
          
            
        # Log
        loss_hist.append(float(value))
        totvio_hist.append(total_vio)

        # Print this iteration
        # print(f"[{it+1:04d}] loss={value:.6f} | #constraints={len(vlist)} | total_vio={total_vio:.6f}")

        # tell
        # opt.tell(cand, value, vlist)

        #Haim - vlist is now a list of lists
        vlist = vlist[0] + vlist[1] #Haim  
        if vlist:
            opt.tell(cand, value, vlist)
        else:
            opt.tell(cand, value, None)
        #End of Haim
    
    # Get recommendation
    recommendation = opt.provide_recommendation()
    

    # === Combined plot ===
    # plt.figure()
    # plt.plot([1000 * x for x in loss_hist], label="Loss")
    # plt.plot([1 * x for x in totvio_hist], label="Total violation")
    # plt.xlabel("Iteration")
    # plt.ylabel("Value")
    # plt.title("Loss & Constraint Violation per Iteration")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # # Haim
    # hlp.print_utility_improvement (opt.budget, loss_hist, totvio_hist)
    

    # Extract best assignment
    best_kwargs = recommendation.kwargs
    best_app_assignment = best_kwargs["app_assignments"]
    best_op_assignment = best_kwargs["op_assignments"]
    best_fee2gas_chains = best_kwargs["fee2gas_chains"]

    # Recompute utilities (positive value this time)
    # Haim - changed to use the canonicalized methods class (below)
    """
    from core.model_basic import evaluate_utilities, evaluate_constraints
    score = evaluate_utilities(
        best_app_assignment, 
        best_op_assignment, 
        best_fee2gas_chains, 
        instance)
    constraints = evaluate_constraints(
        best_app_assignment, 
        best_op_assignment, 
        best_fee2gas_chains, 
        instance)
    """

    
    score = stats.evaluate_utilities_canonicalized(best_app_assignment, best_op_assignment, best_fee2gas_chains, instance) #Haim - to update stats
    constraints = stats.evaluate_constraints_canonicalized(best_app_assignment, best_op_assignment, best_fee2gas_chains, instance) #Haim - to update stats 

    # Stats added by Haim
    print("Total evals:", stats.total_calls, "Unique canonical evals:", stats.unique_calls)
    print("Duplication ratio:", stats.duplication_ratio())
    print("Good-Turing unseen mass ~", stats.good_turing_unseen_mass())
    print("Chao1 total-state estimate ~", stats.chao1_estimate_total_states())
    meff = stats.effective_support_meff()
    print("Effective support (Simpson) ~", meff)

    # unique_states = stats.partitions_with_unassignment_at_most_p(len(instance["apps"]) + len(instance["ops"]), len(instance["chains"]))
    # print(f"Expected number of unique states under uniform sampling over {unique_states} unique states after budget of {general_config.nevergrad_budget} ~", 
    #       stats.uniform_expected_unique(unique_states, general_config.nevergrad_budget))

    target_coverage = 0.95
    print(f"Budget estimated needed to hit {target_coverage} of all effective unique states", 
          stats.uniform_T_for_coverage(meff, target_coverage))
    budget = general_config.nevergrad_budget
    print(f"The expected number of effective unique states given the curent experimentation with  a budget of {budget} ~", 
          stats.expected_unique(meff, budget))
    # Print text histogram
    print("{canonical state frequency: canonical key count):")
    from collections import Counter
    freq_counts = dict(Counter(stats.freq.values()))
    sorted_freq_counts = dict(sorted(freq_counts.items(), key=lambda x: x[1]))
    for key, value in sorted_freq_counts.items():
        print(f"{key}: {'█' * value}")

    # End of stat added by Haim    

    return {
        "app_assignments": best_app_assignment,
        "op_assignments": best_op_assignment,
        "fee2gas_chains": best_fee2gas_chains,
        "utilities": evaluate_utilities(best_app_assignment, best_op_assignment, best_fee2gas_chains, instance)
    }, score, constraints, loss_hist, totvio_hist

