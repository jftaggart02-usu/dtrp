"""Example: solve a 4-circle DTRP and visualise the result."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import matplotlib
matplotlib.use("Agg")          # headless rendering
import matplotlib.pyplot as plt

from dtrp import DTRPSolver, plot_solution

# Four circles arranged in a rough square
circles = [
    (0.0, 0.0, 0.8),
    (6.0, 0.0, 0.8),
    (6.0, 5.0, 0.8),
    (0.0, 5.0, 0.8),
]

solver = DTRPSolver(circles, kappa_max=1.0, steps_per_circle=30)
result = solver.solve(x0=-2, y0=3, visit_order=[0, 1, 2, 3])

print(f"Converged   : {result['success']}")
print(f"Visit order : {result['visit_order']}")
print(f"Total length: {result['total_length']:.4f}")
print(f"Arc-length step ds = {result['ds']:.4f}")

fig, ax = plot_solution(circles, result, title="4-Circle DTRP Example")
out = os.path.join(os.path.dirname(__file__), "dtrp_example.png")
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"Plot saved to {out}")
plt.close(fig)
