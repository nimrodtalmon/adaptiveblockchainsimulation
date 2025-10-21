# core/canonicalized_evaluator.py

import os
import numpy as np
from typing import Dict, Tuple
from config import GeneralConfig; general_config = GeneralConfig()

import nevergrad as ng  # Haim

from collections import Counter
import math

def canonicalize(seq):
    """Relabel buckets by first appearance; keep -1 unchanged."""
    mapping, next_id, out = {}, 0, []
    for v in seq:
        if v == -1:
            out.append(-1)
            continue
        if v not in mapping:
            mapping[v] = next_id
            next_id += 1
        out.append(mapping[v])
    return tuple(out)

class EvaluatorWithStats:
    """
    Wraps a true evaluation function and:
      - canonicalizes candidates (preserves -1 = unassigned)
      - memoizes results
      - tracks frequencies of canonical states
      - exposes coverage metrics
    """
    def __init__(self, n, m, true_evaluate, true_evaluate_utilities, true_constraints, true_evaluate_constraints):
        self.n, self.m = n, m
        self.true_evaluate = true_evaluate
        self.true_evaluate_utilities = true_evaluate_utilities
        self.true_constraints = true_constraints
        self.true_evaluate_constraints = true_evaluate_constraints
        self.eval_cache = {}
        self.viol_cache = {}
        self.freq = Counter()
        self.total_calls = 0
        self.unique_calls = 0

    def evaluate_cannocalized(self, *args, **kwargs):
        """
        kwarge is a dict with {<"app_assignments"|op_assignments"|"fee2gas_chains" : tuple/list of ints}
        where ints ∈ { -1, 0, 1, ..., num_chains-1 }.
        """
        app_assignments = kwargs["app_assignments"]
        op_assignments = kwargs["op_assignments"]
        fee2gas_chains = kwargs["fee2gas_chains"]

        seq = tuple(app_assignments) + tuple(op_assignments)
       
        key = canonicalize(seq)
        self.total_calls += 1
        self.freq[key] += 1
        if key in self.eval_cache:
            return self.eval_cache[key]
        canon_apps_assign, canon_ops_assign = key[:self.n], key[self.n:self.n + self.m]
        dummy_kwargs = {"app_assignments": canon_apps_assign, "op_assignments": canon_ops_assign, "fee2gas_chains": fee2gas_chains}
        val = self.true_evaluate(**dummy_kwargs)
        # tmp for debugging
        non_can_val = self.true_evaluate(**kwargs)
        if val != non_can_val:
            raise ValueError(f"canonical utility {val} and utility {non_can_val} are different, \
                             app-assignments {app_assignments} , op-assignments {op_assignments}")
        # end tmp for debugging
        self.eval_cache[key] = val
        self.unique_calls += 1
        return val
    
    def constraints_canonicalized(self, *args, **kwargs):
        app_assignments = kwargs["app_assignments"]
        op_assignments = kwargs["op_assignments"]
        fee2gas_chains = kwargs["fee2gas_chains"]
        seq = tuple(app_assignments) + tuple(op_assignments)
       
        key = canonicalize(seq)
        if key in self.viol_cache:
            return self.viol_cache[key]
        canon_apps_assign, canon_ops_assign = key[:self.n], key[self.n:self.n + self.m]
        dummy_kwargs = {"app_assignments": canon_apps_assign, "op_assignments": canon_ops_assign, "fee2gas_chains": fee2gas_chains}
        can_constraints = self.true_constraints(**dummy_kwargs)
        # tmp for debugging
        non_can_constraints = self.true_constraints(**kwargs)
        sorted_can_constraints = [sorted(lst) for lst in can_constraints]
        sorted_non_can_constraints = [sorted(lst) for lst in non_can_constraints]

        if sorted_can_constraints != sorted_non_can_constraints:
            raise ValueError(f"canonical constraints {can_constraints} and constraints {non_can_constraints} are different, \
                             app-assignments {app_assignments} , op-assignments {op_assignments}")
        # end tmp for debugging
        self.viol_cache[key] = can_constraints
        return can_constraints
    

    def evaluate_utilities_canonicalized(self, app_assignments, op_assignments, fee2gas_chains, instance): #????
        seq = tuple(app_assignments) + tuple(op_assignments)
        key = canonicalize(seq)
        if key not in self.eval_cache:
            raise ValueError("Recommended key must be in cache.")
        cannon_apps_assign, cannon_ops_assign = key[:self.n], key[self.n:self.n + self.m]
        val = self.true_evaluate_utilities(cannon_apps_assign, cannon_ops_assign, fee2gas_chains, instance) 
        return val


    def evaluate_constraints_canonicalized(self, app_assignments, op_assignments, fee2gas_chains, instance): #????
        seq = tuple(app_assignments) + tuple(op_assignments)
        key = canonicalize(seq)
        if key not in self.eval_cache:
            raise ValueError("Recommended key must be in cache.")
        cannon_apps_assign, cannon_ops_assign = key[:self.n], key[self.n:self.n + self.m]
        constraints = self.true_evaluate_constraints(cannon_apps_assign, cannon_ops_assign, fee2gas_chains, instance) 
        return constraints





    # ------- Diagnostics -------
    def unique_states(self):
        return len(self.cache)

    def duplication_ratio(self):
        """How many total evals per unique canonical state (≥1)."""
        return self.total_calls / max(1, self.unique_calls)

    def good_turing_unseen_mass(self):
        """Estimated probability mass of unseen states: p0 ≈ f1/N."""
        counts = Counter(self.freq.values())
        f1, N = counts.get(1, 0), self.total_calls
        return (f1 / N) if N > 0 else 1.0

    def chao1_estimate_total_states(self):
        """Chao1 richness estimate of total canonical states explored by this sampler."""
        S_obs = len(self.freq)
        counts = Counter(self.freq.values())
        f1, f2 = counts.get(1, 0), counts.get(2, 0)
        if f2 > 0:
            return S_obs + (f1 * f1) / (2 * f2)
        else:
            return S_obs + f1 * (f1 - 1) / 2

    def expected_unique(self, M, T):
        """Expected unique states, either theoretically, under uniform sampling or effectively, """
        """based on the experimentation"""
        """M is either (1) the theoretical number of unique canonical states. or:"""
        """            (2) the effective support Meff """
        """T is the number of samples/evaluations (the budget -no of iterations)"""
        return 0.0 if M <= 0 else M * (1 - (1 - 1/M) ** T)

    def uniform_T_for_coverage(self, M, target_frac):
        """Solve for T to hit a coverage fraction under uniform sampling."""
        """M is either (1) the theoretical number of unique canonical states. or:"""
        """            (2) the effective support Meff """
        """ target_frac is the coverage target, as a fraction of M, e.g., 0.95"""
        if not (0 < target_frac < 1): 
            raise ValueError("target_frac must be in (0,1).")
        if M <= 1:
            return 0.0
        return math.log(1 - target_frac) / math.log(1 - 1 / M)

    def coupon_collector_all(self, M):
        """Expected T to see all M states (uniform): ~ M (ln M + γ + 1/(2M))."""
        """ M is M the number of unique canonical states"""
        if M <= 1:
            return M
        gamma = 0.5772156649015329
        return M * (math.log(M) + gamma + 1/(2*M))

    def effective_support_meff(self):
        """
        Inverse-Simpson effective support 1/sum p_i^2 estimated from freq.
        Lower M_eff => more peaked sampling => more duplication expected.
        """
        N = self.total_calls
        if N == 0:
            return 0.0
        s2 = sum((c / N) ** 2 for c in self.freq.values())
        return (1.0 / s2) if s2 > 0 else 0.0
    
    # ---computing the total number of distinct possible states (canonical assignments,
    import math
    @staticmethod
    def stirling2(n, k):
        """Stirling number S(n,k): partitions of n labeled items into k nonempty unlabeled blocks."""
        S = [[0]*(k+1) for _ in range(n+1)]
        S[0][0] = 1
        for i in range(1, n+1):
            for j in range(1, k+1):
                S[i][j] = j*S[i-1][j] + S[i-1][j-1]
        return S[n][k]
    
    @staticmethod
    def partitions_at_most_p_no_unassignment(N, p):
        """# partitions of N items into ≤ p blocks."""
        p = min(p, N)
        return sum(EvaluatorWithStats.stirling2(N, k) for k in range(1, p+1))

    @staticmethod
    def partitions_exactly_p_no_unassignment(N, p):
        """# partitions of N items into exactly p blocks."""
        if p > N or p <= 0:
            return 0
        return EvaluatorWithStats.stirling2(N, p)

    @staticmethod
    def partitions_with_unassignment_at_most_p(N, p):
        """
        # structures where each of N items may be unassigned (-1) or assigned into ≤ p blocks.
        Sum over k assigned items: C(N,k) * (partitions of k items into ≤ p blocks).
        """
        total = 0
        for k in range(0, N+1):
            choose = math.comb(N, k)
            inner = sum(EvaluatorWithStats.stirling2(k, j) for j in range(1, min(k, p) + 1))
            total += choose * inner
        return total

    @staticmethod
    def partitions_with_unassignment_exactly_p(N, p):
        """Same as above but require exactly p nonempty blocks among the assigned."""
        total = 0
        for k in range(p, N+1):
            total += math.comb(N, k) * EvaluatorWithStats.stirling2(k, p)
        return total
    