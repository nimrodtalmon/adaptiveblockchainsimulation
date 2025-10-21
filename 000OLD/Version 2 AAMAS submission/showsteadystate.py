import matplotlib.pyplot as plt
import numpy as np
import os

# Path to your data file
path = "logs/steady_state.txt"

# Read and parse data
data = []
with open(path, "r") as f:
    for line in f:
        parts = line.strip().split(",")
        if len(parts) != 3:
            continue
        try:
            apps = int(parts[0])
            ops = int(parts[1])
            steady = None if parts[2] == "None" else float(parts[2])
            if steady is not None:
                data.append((apps, ops, steady))
        except ValueError:
            continue

# Group by instance size (assuming apps == ops)
from collections import defaultdict
grouped = defaultdict(list)
for a, o, s in data:
    grouped[a].append(s)

sizes = sorted(grouped.keys())
means = [np.mean(grouped[k]) for k in sizes]
stds = [np.std(grouped[k]) for k in sizes]
counts = [len(grouped[k]) for k in sizes]

# --- Plot ---
plt.figure(figsize=(6.5, 4.2))
plt.errorbar(sizes, means, yerr=stds, fmt='o-', capsize=4, lw=2)
for x, y, c in zip(sizes, means, counts):
    plt.text(x, y * 1.02, f"n={c}", ha='center', fontsize=8, color='gray')

plt.title("Steady-State Iteration vs. Instance Size", fontsize=12)
plt.xlabel("Number of Applications", fontsize=11)
plt.ylabel("Steady-State Iteration (Budget)", fontsize=11)
plt.grid(True, alpha=0.4)
plt.tight_layout()

# Save plot
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/steadystate.png", dpi=300)
# plt.show()
