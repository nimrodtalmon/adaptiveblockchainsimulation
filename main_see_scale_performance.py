# main_plateau_benchmark.py
import time
import random
import numpy as np
import matplotlib.pyplot as plt

from core import instance_generator
from core.optimizer import solve_model_until_plateau
from utils.helpers import clear_screen


def main():
    clear_screen()
    print("\n>>> Adaptive Multichain Blockchain Simulation (Plateau Benchmark) <<<")

    # Reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    scales = list(range(1, 101, 10))
    total_budget = 10000

    iters_list = []
    time_list = []

    for scale in scales:
        instance = instance_generator.generate_random_instance_1(
            num_apps=scale,
            num_ops=scale,
            num_chains=scale,
        )

        # run (and time)
        t0 = time.perf_counter()

        try:
            out = solve_model_until_plateau(instance, real_budget=total_budget, verbose=False)
            dt = time.perf_counter() - t0

            solution, score, constraints, loss_hist, totvio_hist, iters_used = out
            ok = True
            err = None

        except Exception as e:
            # record failure, keep the benchmark running
            dt = time.perf_counter() - t0
            ok = False
            err = f"{type(e).__name__}: {e}"
            solution = score = constraints = loss_hist = totvio_hist = None
            iters_used = 0  # or total_budget, but 0 makes "failed" obvious

        if ok:
            iters_list.append(int(iters_used))
            time_list.append(float(dt))
            print(f"scale={scale:2d} | plateau iters={iters_used:4d} | time={dt:7.3f}s")
        else:
            # keep array lengths aligned for plotting; use NaN so matplotlib breaks the line
            iters_list.append(np.nan)
            time_list.append(np.nan)
            print(f"scale={scale:2d} | FAILED after {dt:7.3f}s | {err}")
        
        dt = time.perf_counter() - t0

        # unpack
        solution, score, constraints, loss_hist, totvio_hist, iters_used = out

        iters_list.append(int(iters_used))
        time_list.append(float(dt))

        # print(f"scale={scale:2d} | plateau iters={iters_used:4d} | time={dt:7.3f}s")

    # ---- Pretty Plot: scale vs iterations + time ----
    fig, ax1 = plt.subplots()

    c_iters = "tab:blue"
    c_time = "tab:orange"

    # Left axis: iterations
    l1, = ax1.plot(scales, iters_list, marker="o", linewidth=2, color=c_iters, label="Iterations until plateau")
    ax1.set_xlabel("Scale (num_apps = num_ops = num_chains)")
    ax1.set_ylabel("Iterations until plateau", color=c_iters)
    ax1.tick_params(axis="y", labelcolor=c_iters)
    ax1.spines["left"].set_color(c_iters)

    # Light grid on primary axis
    ax1.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.4)

    # Right axis: time
    ax2 = ax1.twinx()
    l2, = ax2.plot(scales, time_list, marker="s", linewidth=2, color=c_time, label="Seconds until plateau")
    ax2.set_ylabel("Seconds until plateau", color=c_time)
    ax2.tick_params(axis="y", labelcolor=c_time)
    ax2.spines["right"].set_color(c_time)

    # Title + combined legend
    plt.title("Plateau vs Instance Scale")
    ax1.legend(handles=[l1, l2], loc="upper left")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
