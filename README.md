# DTRP — Dubins Touring Region Problem Solver

Given a set of **circular regions** to visit, find the **shortest
curvature-constrained path** that passes through every circle.

## Approach

The problem is formulated as a **nonlinear optimal control problem** and
solved via **direct collocation**:

| Symbol | Meaning |
|--------|---------|
| (xᵢ, yᵢ) | vehicle position at step i |
| θᵢ | heading angle at step i |
| ds | uniform arc-length step (decision variable) |
| κ_max | maximum curvature = 1 / min-turning-radius |

**Objective** — minimise total path length = N · ds

**Dynamics** (midpoint rule):
```
x_{i+1} = x_i + cos((θ_i + θ_{i+1}) / 2) · ds
y_{i+1} = y_i + sin((θ_i + θ_{i+1}) / 2) · ds
```

**Curvature constraint**:
```
|θ_{i+1} − θ_i| / ds ≤ κ_max
```

**Visit constraint** (one per circle j):
```
(x_{k_j} − c_xʲ)² + (y_{k_j} − c_yʲ)² ≤ r_j²
```

The off-the-shelf **SLSQP** optimizer from `scipy.optimize.minimize` is used
to solve the resulting NLP.  Analytical Jacobians are provided for all
constraints to maximise solver speed.

Visit order is determined automatically:
- **exhaustive search** over all permutations when n ≤ 7 circles
- **nearest-neighbour + 2-opt** for larger instances

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from dtrp import DTRPSolver, plot_solution

circles = [
    (0.0, 0.0, 0.8),   # (cx, cy, radius)
    (6.0, 0.0, 0.8),
    (6.0, 5.0, 0.8),
    (0.0, 5.0, 0.8),
]

solver = DTRPSolver(circles, kappa_max=1.0, steps_per_circle=30)
result = solver.solve()

print(f"Total length : {result['total_length']:.4f}")
print(f"Visit order  : {result['visit_order']}")

fig, ax = plot_solution(circles, result)
fig.savefig("path.png", dpi=120)
```

## Running the Example

```bash
python examples/example.py
```

## Running Tests

```bash
pytest tests/ -v
```

## API

### `DTRPSolver(circles, kappa_max=1.0, steps_per_circle=25, max_permutations=7)`

| Parameter | Description |
|-----------|-------------|
| `circles` | List of `(cx, cy, r)` tuples |
| `kappa_max` | Maximum curvature (1 / min-turning-radius) |
| `steps_per_circle` | Discretisation steps per circle segment |
| `max_permutations` | Exhaustive permutation search threshold |

### `solver.solve(x0=None, y0=None, theta0=None, visit_order=None) → dict`

Returns a dict with keys: `x`, `y`, `theta`, `ds`, `total_length`,
`visit_order`, `success`, `message`.
