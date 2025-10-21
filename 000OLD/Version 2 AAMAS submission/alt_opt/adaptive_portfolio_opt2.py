import math
import numpy as np
from nevergrad.optimization import optimizerlib, base


# =====================================================================
# Canonical NG-style constraint penalty (final version)
# =====================================================================
def ng_constraint_penalty(loss, violations, num_tell,
                          maximize=True,
                          penalty_style=None,
                          stationary=False):
    """
    NG-style penalty with 6 constants (a,b,c,d,e,f).
    Default minimization:
        base = a + max(loss, 0)
    Maximization (negative objectives):
        base = a - min(loss, 0)   # equivalent to a + |loss| if loss < 0
    """
    if violations is None:
        violations = []
    a, b, c, d, e, f = penalty_style or (1e5, 1.0, 0.5, 1.0, 0.5, 1.0)

    v = np.asarray(list(violations), dtype=float)
    v = np.maximum(v, 0.0)
    sum_v_c = float(np.sum(v ** c))

    if not maximize:
        base = a + max(float(loss), 0.0)
    else:
        base = a - min(float(loss), 0.0)

    time_factor = 1.0 if stationary else (f + float(num_tell)) ** e
    scale = b * (sum_v_c ** d) if sum_v_c > 0.0 else 0.0

    violation = float(base * time_factor * scale)
    return float(loss + violation), violation


# =====================================================================
# Streaming median/MAD tracker
# =====================================================================
class RunningMedianLike:
    def __init__(self, m0=0.0, s0=1.0, eta_m=1e-3, eta_s=1e-3):
        self.m = float(m0)
        self.s = float(s0)
        self.eta_m = float(eta_m)
        self.eta_s = float(eta_s)

    def update(self, x: float) -> None:
        x = float(x)
        if x > self.m:
            self.m += self.eta_m
        elif x < self.m:
            self.m -= self.eta_m
        self.s = (1.0 - self.eta_s) * self.s + self.eta_s * abs(x - self.m)

    @property
    def median(self):
        return self.m

    @property
    def mad(self):
        return max(self.s, 1e-12)


# =====================================================================
# Adaptive Portfolio Optimizer
# =====================================================================
class _AdaptivePortfolioOptimizer(base.Optimizer):
    """
    Portfolio of OnePlusOne variants with:
      - Canonical NG-style constraint penalties (a,b,c,d,e,f)
      - UCB1 arm selection with hybrid normalization:
          normalize_mode="none"      → raw eff loss
          normalize_mode="mad_mean"  → blend MAD and mean normalization
          normalize_mode="mad_minmax"→ blend MAD and min–max normalization
        Controlled by α ∈ [0,1]:
          α=1.0 → pure MAD; α=0.0 → pure companion normalization
      - Optional streaming median/MAD after threshold
      - Diagnostics, selection log, batch tell helper
    """

    def __init__(self, parametrization, budget=None, num_workers=1,
                 *,
                 penalty_style=(1e5, 1.0, 0.5, 1.0, 0.5, 1.0),
                 stationary_penalty=False,
                 maximize=False,
                 normalize_mode="none",   # "none", "mad_mean", "mad_minmax"
                 alpha=0.5,
                 streaming_threshold=50000):

        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self.penalty_style = penalty_style
        self.stationary_penalty = stationary_penalty
        self.maximize = maximize
        self.normalize_mode = normalize_mode
        self.alpha = float(alpha)
        self.streaming_threshold = int(streaming_threshold)
        self.stats: RunningMedianLike | None = None

        self.total_steps = 0
        self.loss_history = []
        self.step_log = []
        self.selection_log = []  # new: record arm-selection decisions

        # Sub-optimizers (3 OnePlusOne mutation strategies)
        self.sub_factories = [
            optimizerlib.ParametrizedOnePlusOne(mutation="standard"),
            optimizerlib.ParametrizedOnePlusOne(mutation="gaussian"),
            optimizerlib.ParametrizedOnePlusOne(mutation="cauchy"),
        ]
        self.sub_opts = [f(parametrization, budget=budget, num_workers=num_workers)
                         for f in self.sub_factories]
        self.loss_history = [[] for _ in self.sub_opts]

    # --------------------------- ASK ---------------------------
    def _internal_ask_candidate(self):
        if self.total_steps >= (self.budget or float("inf")):
            raise RuntimeError("Budget exhausted")

        idx = self._select_optimizer_index()  # wrapper for NG consistency
        cand = self.sub_opts[idx].ask()
        cand._from_portfolio_idx = idx
        return cand

    # --------------------------- TELL --------------------------
    def _internal_tell_candidate(self, candidate, value):
        """Receives evaluated candidate, computes penalties, updates logs/statistics."""
        idx = getattr(candidate, "_from_portfolio_idx", None)
        if idx is None:
            return

        # --- Parse incoming (loss, violations) ---
        if isinstance(value, tuple) and len(value) == 2:
            raw_loss, violations = value
        else:
            raw_loss, violations = float(value), ()

        # --- Compute effective loss using NG-style penalty ---
        eff_loss, penalty = ng_constraint_penalty(
            raw_loss, violations, self._num_tell,
            maximize=self.maximize,
            penalty_style=self.penalty_style,
            stationary=self.stationary_penalty,
        )

        # --- Log into sub-optimizer and portfolio history ---
        self.sub_opts[idx].tell(candidate, eff_loss)
        self.loss_history[idx].append((float(raw_loss), tuple(violations), float(eff_loss)))
        self.step_log.append((float(raw_loss), penalty, float(eff_loss), tuple(violations), idx))
        self.total_steps += 1

        # --- Maintain streaming median tracker continuously for smooth transition ---
        if self.stats is None:
            # Initialize after ~50 samples (or earlier if small problems)
            if self.total_steps >= 50:
                # compute real median and std from available eff_losses so far
                all_eff = [h[2] for hist in self.loss_history for h in hist]
                m0 = np.median(all_eff) if all_eff else raw_loss
                s0 = np.std(all_eff) if all_eff else 1.0
                self.stats = RunningMedianLike(m0=m0, s0=s0)
            else:
                # before 50 samples: initialize with current loss
                self.stats = RunningMedianLike(m0=raw_loss, s0=1.0)

        # always update (but used only after threshold)
        self.stats.update(eff_loss)

    # -------------------- Batch tell helper --------------------
    def tell_batch(self, cand_values):
        for entry in cand_values:
            if isinstance(entry, tuple) and len(entry) == 3:
                cand, loss, violations = entry
                self.tell(cand, loss, violations)
            else:
                cand, value = entry
                self.tell(cand, value)

    # -------------------- Recommendation ----------------------
    def _internal_provide_recommendation(self):
        best_cand, best_val = None, float("inf")
        for sub in self.sub_opts:
            try:
                rec = sub.recommend()
                val = getattr(rec, "loss", None)
                if val is not None and val < best_val:
                    best_val, best_cand = val, rec
            except Exception:
                continue
        return best_cand

    # --------------- Wrapper for NG consistency ----------------
    def _select_optimizer_index(self):
        """Thin wrapper kept for Nevergrad consistency (delegates to hybrid)."""
        return self._select_optimizer_index_hybrid(
            alpha=self.alpha,
            normalize_mode=self.normalize_mode,
        )

    # --------------- Hybrid normalization selector -------------
    def _select_optimizer_index_hybrid(self, *, alpha=0.5, normalize_mode="none"):
        T = max(1, self.total_steps)
        for i, hist in enumerate(self.loss_history):
            if not hist:
                return i

        all_eff = [h[2] for hist in self.loss_history for h in hist] or [0.0]
        global_mean = float(np.mean(all_eff))
        mn, mx = min(all_eff), max(all_eff)
        span = (mx - mn) or 1.0

        if self.stats and self.total_steps > self.streaming_threshold:
            med, mad = self.stats.median, self.stats.mad
        else:
            med = np.median(all_eff)
            mad = np.median(np.abs(np.array(all_eff) - med)) or 1.0

        scores = []
        for hist in self.loss_history:
            mean_eff = float(np.mean([h[2] for h in hist]))
            if normalize_mode == "none":
                scaled_mean = mean_eff
            else:
                mad_scaled = (mean_eff - med) / mad
                if normalize_mode == "mad_mean":
                    comp = (mean_eff - global_mean) / (abs(global_mean) + 1e-12)
                elif normalize_mode == "mad_minmax":
                    comp = (mean_eff - mn) / span
                else:
                    raise ValueError(f"Unknown normalize_mode: {normalize_mode}")
                scaled_mean = float(alpha) * mad_scaled + (1.0 - float(alpha)) * comp

            bonus = (2.0 * math.log(T + 1) / len(hist)) ** 0.5
            scores.append(-scaled_mean + bonus)

        best_idx = max(range(len(scores)), key=scores.__getitem__)

        # --- new: record selection step
        self.selection_log.append({
            "step": self.total_steps,
            "chosen_arm": best_idx,
            "scores": scores,
            "normalize_mode": normalize_mode,
            "alpha": alpha
        })

        return best_idx

    # ------------------------ Diagnostics ---------------------
    def get_diagnostics(self):
        per_arm = []
        for i, hist in enumerate(self.loss_history):
            if not hist:
                per_arm.append((i, None, None, 0))
                continue
            eff_vals = [h[2] for h in hist]
            per_arm.append((i, np.mean(eff_vals), np.median(eff_vals), len(hist)))
        return {
            "per_arm": per_arm,
            "steps": list(self.step_log),
            "selections": list(self.selection_log),
            "total_steps": self.total_steps,
            "normalize_mode": self.normalize_mode,
            "alpha": self.alpha,
            "streaming": self.stats is not None
        }

    def summarize_diagnostics(self):
        d = self.get_diagnostics()
        print("=== Diagnostics Summary ===")
        print(f"total_steps={d['total_steps']}, mode={d['normalize_mode']}, "
              f"alpha={d['alpha']}, "
              f"streaming={'yes' if d['streaming'] else 'no'}")
        for (i, meanv, medv, n) in d["per_arm"]:
            if n == 0:
                print(f"Arm {i}: unused")
            else:
                print(f"Arm {i}: n={n:5d}, mean_eff={meanv:12.6f}, median_eff={medv:12.6f}")
        print("===========================")


# =====================================================================
# Factory for Nevergrad compatibility
# =====================================================================
class MyAdaptiveDiscretePortfolio(optimizerlib.OptimizerFamily):
    """Factory so you can instantiate like NG built-ins."""
    """Drop-in replacement for PortfolioDiscreteOnePlusOne."""
    def __init__(self, **kwargs):
        super().__init__(_AdaptivePortfolioOptimizer)
        self._parameters = kwargs