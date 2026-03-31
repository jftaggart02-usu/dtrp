"""Solver for the Dubins Touring Region Problem (DTRP).

Optimal Control Formulation
----------------------------
State at step i: position (x_i, y_i) and heading θ_i.
Uniform arc-length step: ds  (a decision variable).

    minimize    N · ds                      (total path length)

    subject to  x_{i+1} = x_i + cos(θ̄_i) · ds          (x-dynamics)
                y_{i+1} = y_i + sin(θ̄_i) · ds          (y-dynamics)
                |θ_{i+1} − θ_i| ≤ κ_max · ds           (curvature bound)
                (x_{k_j} − c_x^j)² + (y_{k_j} − c_y^j)² ≤ r_j²  (visit)
                ds > 0

where θ̄_i = (θ_i + θ_{i+1}) / 2  (midpoint rule) and
k_j is the designated visit step for circle j.

The off-the-shelf SLSQP solver from ``scipy.optimize.minimize`` is used.
Visit order is determined by a Euclidean TSP heuristic (nearest-neighbour +
2-opt), with an exhaustive search over all permutations when there are at
most ``max_permutations`` circles.
"""

from __future__ import annotations

import warnings
from itertools import permutations
from typing import Sequence

import numpy as np
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
Circle = tuple[float, float, float]   # (cx, cy, r)


# ---------------------------------------------------------------------------
# Public solver class
# ---------------------------------------------------------------------------

class DTRPSolver:
    """Solve the Dubins Touring Region Problem via direct collocation.

    Parameters
    ----------
    circles:
        Sequence of ``(cx, cy, r)`` tuples that describe the regions to visit.
    kappa_max:
        Maximum path curvature (1 / minimum turning radius).  Default 1.0.
    steps_per_circle:
        Number of arc-length discretisation steps allocated to each circle
        segment.  Larger values give a more accurate trajectory at the cost
        of a bigger NLP.  Default 25.
    max_permutations:
        Exhaustive visit-order search is used when ``len(circles) <=
        max_permutations``; otherwise the nearest-neighbour + 2-opt heuristic
        is used.  Default 7.
    """

    def __init__(
        self,
        circles: Sequence[Circle],
        kappa_max: float = 1.0,
        steps_per_circle: int = 25,
        max_permutations: int = 7,
    ) -> None:
        self.circles = [tuple(c) for c in circles]
        self.kappa_max = float(kappa_max)
        self.steps_per_circle = int(steps_per_circle)
        self.max_permutations = int(max_permutations)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(
        self,
        x0: float | None = None,
        y0: float | None = None,
        theta0: float | None = None,
        visit_order: list[int] | None = None,
    ) -> dict:
        """Solve the DTRP and return the optimal trajectory.

        Parameters
        ----------
        x0, y0:
            Optional fixed start position.  When omitted the centre of the
            first circle (after ordering) is used.
        theta0:
            Optional fixed start heading in radians.  When omitted the
            heading towards the next circle is used.
        visit_order:
            Explicit visit order as a list of indices into ``self.circles``.
            When ``None`` an order is determined automatically.

        Returns
        -------
        dict with keys:

        ``x``, ``y``, ``theta``
            Trajectory arrays (shape ``(N+1,)``).
        ``ds``
            Arc-length step size.
        ``total_length``
            Total path length = N · ds.
        ``visit_order``
            The visit order used.
        ``success``
            ``True`` if the optimizer converged.
        ``message``
            Optimizer status message.
        """
        n = len(self.circles)

        if n == 0:
            return {
                "x": np.array([]),
                "y": np.array([]),
                "theta": np.array([]),
                "ds": 0.0,
                "total_length": 0.0,
                "visit_order": [],
                "success": True,
                "message": "No circles provided.",
            }

        if visit_order is None:
            visit_order = self._find_visit_order()

        # Normalise coordinates for numerical stability
        coords = np.array([[c[0], c[1]] for c in self.circles])
        centre = coords.mean(axis=0)
        scale = max(np.ptp(coords[:, 0]), np.ptp(coords[:, 1]), 1e-6)

        circles_n = [
            ((c[0] - centre[0]) / scale,
             (c[1] - centre[1]) / scale,
             c[2] / scale)
            for c in self.circles
        ]
        kappa_n = self.kappa_max * scale

        x0_n = (x0 - centre[0]) / scale if x0 is not None else None
        y0_n = (y0 - centre[1]) / scale if y0 is not None else None

        result, (x_s, y_s, theta_s, ds_s) = self._solve_nlp(
            circles_n, kappa_n, visit_order, x0_n, y0_n, theta0
        )

        N = len(self.circles) * self.steps_per_circle
        return {
            "x": x_s * scale + centre[0],
            "y": y_s * scale + centre[1],
            "theta": theta_s,
            "ds": float(ds_s * scale),
            "total_length": float(N * ds_s * scale),
            "visit_order": visit_order,
            "success": result.success,
            "message": result.message,
        }

    # ------------------------------------------------------------------
    # Visit-order helpers
    # ------------------------------------------------------------------

    def _find_visit_order(self) -> list[int]:
        n = len(self.circles)
        if n == 1:
            return [0]

        centres = np.array([[c[0], c[1]] for c in self.circles])

        if n <= self.max_permutations:
            best: list[int] = list(range(n))
            best_cost = _tour_length(centres, best)
            for perm in permutations(range(n)):
                cost = _tour_length(centres, list(perm))
                if cost < best_cost:
                    best_cost = cost
                    best = list(perm)
            return best

        order = _nearest_neighbour(centres)
        return _two_opt(centres, order)

    # ------------------------------------------------------------------
    # NLP solver
    # ------------------------------------------------------------------

    def _solve_nlp(
        self,
        circles: list[Circle],
        kappa_max: float,
        visit_order: list[int],
        x0: float | None,
        y0: float | None,
        theta0: float | None,
    ) -> tuple:
        S = self.steps_per_circle
        nc = len(circles)
        N = nc * S                    # total number of arc-length steps
        nv = 3 * (N + 1) + 1         # number of decision variables

        ordered = [circles[i] for i in visit_order]

        # --------------------------------------------------------------
        # Variable layout:  z = [x(0..N), y(0..N), θ(0..N), ds]
        # Indices:
        #   x[i]     → i
        #   y[i]     → (N+1) + i
        #   theta[i] → 2*(N+1) + i
        #   ds       → 3*(N+1)
        # --------------------------------------------------------------

        def unpack(z: np.ndarray):
            x = z[:N + 1]
            y = z[N + 1: 2 * (N + 1)]
            theta = z[2 * (N + 1): 3 * (N + 1)]
            ds = z[-1]
            return x, y, theta, ds

        # ------ objective ---------------------------------------------

        def objective(z):
            _, _, _, ds = unpack(z)
            return N * ds

        def objective_jac(z):
            g = np.zeros(nv)
            g[-1] = N                # ∂/∂ds = N
            return g

        # ------ equality constraints: x-dynamics ----------------------
        # g_i = x[i+1] - x[i] - cos(θ̄_i)·ds = 0

        def dyn_x(z):
            x, _, theta, ds = unpack(z)
            tm = (theta[:-1] + theta[1:]) / 2.0
            return x[1:] - x[:-1] - np.cos(tm) * ds

        def dyn_x_jac(z):
            _, _, theta, ds = unpack(z)
            tm = (theta[:-1] + theta[1:]) / 2.0
            ii = np.arange(N)
            J = np.zeros((N, nv))
            J[ii, ii] = -1.0                              # ∂/∂x[i]
            J[ii, ii + 1] = 1.0                           # ∂/∂x[i+1]
            J[ii, 2 * (N + 1) + ii] = np.sin(tm) * ds / 2.0    # ∂/∂θ[i]
            J[ii, 2 * (N + 1) + ii + 1] = np.sin(tm) * ds / 2.0  # ∂/∂θ[i+1]
            J[ii, -1] = -np.cos(tm)                       # ∂/∂ds
            return J

        # ------ equality constraints: y-dynamics ----------------------

        def dyn_y(z):
            _, y, theta, ds = unpack(z)
            tm = (theta[:-1] + theta[1:]) / 2.0
            return y[1:] - y[:-1] - np.sin(tm) * ds

        def dyn_y_jac(z):
            _, _, theta, ds = unpack(z)
            tm = (theta[:-1] + theta[1:]) / 2.0
            ii = np.arange(N)
            J = np.zeros((N, nv))
            J[ii, (N + 1) + ii] = -1.0                    # ∂/∂y[i]
            J[ii, (N + 1) + ii + 1] = 1.0                 # ∂/∂y[i+1]
            J[ii, 2 * (N + 1) + ii] = -np.cos(tm) * ds / 2.0   # ∂/∂θ[i]
            J[ii, 2 * (N + 1) + ii + 1] = -np.cos(tm) * ds / 2.0  # ∂/∂θ[i+1]
            J[ii, -1] = -np.sin(tm)                        # ∂/∂ds
            return J

        # ------ inequality constraints: curvature upper ---------------
        # c_i = κ_max·ds − (θ[i+1] − θ[i]) ≥ 0

        def curv_upper(z):
            _, _, theta, ds = unpack(z)
            return kappa_max * ds - (theta[1:] - theta[:-1])

        def curv_upper_jac(z):
            ii = np.arange(N)
            J = np.zeros((N, nv))
            J[ii, 2 * (N + 1) + ii] = 1.0                 # ∂/∂θ[i]  = +1
            J[ii, 2 * (N + 1) + ii + 1] = -1.0            # ∂/∂θ[i+1] = -1
            J[ii, -1] = kappa_max                          # ∂/∂ds
            return J

        # ------ inequality constraints: curvature lower ---------------
        # d_i = κ_max·ds + (θ[i+1] − θ[i]) ≥ 0

        def curv_lower(z):
            _, _, theta, ds = unpack(z)
            return kappa_max * ds + (theta[1:] - theta[:-1])

        def curv_lower_jac(z):
            ii = np.arange(N)
            J = np.zeros((N, nv))
            J[ii, 2 * (N + 1) + ii] = -1.0                # ∂/∂θ[i]  = -1
            J[ii, 2 * (N + 1) + ii + 1] = 1.0             # ∂/∂θ[i+1] = +1
            J[ii, -1] = kappa_max                          # ∂/∂ds
            return J

        # ------ inequality constraints: visit circles -----------------
        # v_j = r_j² − (x[k_j] − cx_j)² − (y[k_j] − cy_j)² ≥ 0

        visit_constraints: list[dict] = []
        for j, (cx, cy, r) in enumerate(ordered):
            k = j * S + S // 2  # designated visit step

            def _make_visit(k_=k, cx_=cx, cy_=cy, r_=r):
                def fun(z):
                    x, y, _, _ = unpack(z)
                    return r_ ** 2 - (x[k_] - cx_) ** 2 - (y[k_] - cy_) ** 2

                def jac(z):
                    x, y, _, _ = unpack(z)
                    g = np.zeros(nv)
                    g[k_] = -2.0 * (x[k_] - cx_)         # ∂/∂x[k]
                    g[(N + 1) + k_] = -2.0 * (y[k_] - cy_)  # ∂/∂y[k]
                    return g

                return fun, jac

            vfun, vjac = _make_visit()
            visit_constraints.append(
                {"type": "ineq", "fun": vfun, "jac": vjac}
            )

        # ------ optional start-point equality constraints -------------
        pin_constraints: list[dict] = []
        if x0 is not None:
            def pin_x(z):
                x, _, _, _ = unpack(z)
                return np.array([x[0] - x0])

            def pin_x_jac(z):
                g = np.zeros((1, nv))
                g[0, 0] = 1.0
                return g

            pin_constraints.append({"type": "eq", "fun": pin_x, "jac": pin_x_jac})

        if y0 is not None:
            def pin_y(z):
                _, y, _, _ = unpack(z)
                return np.array([y[0] - y0])

            def pin_y_jac(z):
                g = np.zeros((1, nv))
                g[0, (N + 1)] = 1.0
                return g

            pin_constraints.append({"type": "eq", "fun": pin_y, "jac": pin_y_jac})

        if theta0 is not None:
            def pin_theta(z):
                _, _, theta, _ = unpack(z)
                return np.array([theta[0] - theta0])

            def pin_theta_jac(z):
                g = np.zeros((1, nv))
                g[0, 2 * (N + 1)] = 1.0
                return g

            pin_constraints.append(
                {"type": "eq", "fun": pin_theta, "jac": pin_theta_jac}
            )

        # ------ assemble all constraints ------------------------------
        constraints = (
            [
                {"type": "eq", "fun": dyn_x, "jac": dyn_x_jac},
                {"type": "eq", "fun": dyn_y, "jac": dyn_y_jac},
                {"type": "ineq", "fun": curv_upper, "jac": curv_upper_jac},
                {"type": "ineq", "fun": curv_lower, "jac": curv_lower_jac},
            ]
            + visit_constraints
            + pin_constraints
        )

        # Bounds: ds ≥ ε > 0; all other variables unbounded
        bounds = [(None, None)] * (3 * (N + 1)) + [(1e-9, None)]

        # ------ initial guess -----------------------------------------
        z0 = _initial_guess(ordered, S, N, x0, y0, theta0)

        # ------ optimise ----------------------------------------------
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                objective,
                z0,
                jac=objective_jac,
                method="SLSQP",
                constraints=constraints,
                bounds=bounds,
                options={"maxiter": 2000, "ftol": 1e-7, "disp": False},
            )

        return result, unpack(result.x)


# ---------------------------------------------------------------------------
# Tour-order helpers (module-level so they are easily testable)
# ---------------------------------------------------------------------------

def _tour_length(centres: np.ndarray, order: list[int]) -> float:
    """Sum of Euclidean distances along an open tour."""
    total = 0.0
    for i in range(len(order) - 1):
        total += float(np.linalg.norm(centres[order[i + 1]] - centres[order[i]]))
    return total


def _nearest_neighbour(centres: np.ndarray) -> list[int]:
    n = len(centres)
    visited = [False] * n
    order = [0]
    visited[0] = True
    for _ in range(n - 1):
        curr = order[-1]
        best_j, best_d = -1, np.inf
        for j in range(n):
            if not visited[j]:
                d = float(np.linalg.norm(centres[j] - centres[curr]))
                if d < best_d:
                    best_d = d
                    best_j = j
        order.append(best_j)
        visited[best_j] = True
    return order


def _two_opt(centres: np.ndarray, order: list[int]) -> list[int]:
    n = len(order)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                new = order[: i + 1] + order[i + 1: j + 1][::-1] + order[j + 1:]
                if _tour_length(centres, new) < _tour_length(centres, order):
                    order = new
                    improved = True
    return order


# ---------------------------------------------------------------------------
# Initial-guess construction
# ---------------------------------------------------------------------------

def _initial_guess(
    ordered_circles: list[Circle],
    S: int,
    N: int,
    x0: float | None,
    y0: float | None,
    theta0: float | None,
) -> np.ndarray:
    """Build a straight-line initial guess that passes through circle centres.

    The designated visit step for circle j (middle of its segment) is
    initialised at the centre of that circle.  Heading is derived from the
    velocity direction along the interpolated path.
    """
    nc = len(ordered_circles)
    visit_steps = [j * S + S // 2 for j in range(nc)]

    key_x = [c[0] for c in ordered_circles]
    key_y = [c[1] for c in ordered_circles]

    start_x = key_x[0] if x0 is None else x0
    start_y = key_y[0] if y0 is None else y0

    # Anchor points along the trajectory (step index → position)
    anchor_steps = [0] + visit_steps + [N]
    anchor_x = [start_x] + key_x + [key_x[-1]]
    anchor_y = [start_y] + key_y + [key_y[-1]]

    t = np.arange(N + 1, dtype=float)
    x_init = np.interp(t, anchor_steps, anchor_x)
    y_init = np.interp(t, anchor_steps, anchor_y)

    dx = np.diff(x_init)
    dy = np.diff(y_init)
    angles = np.arctan2(dy, dx)
    theta_init = np.append(angles, angles[-1])

    if theta0 is not None:
        theta_init[0] = theta0

    seg_len = np.sqrt(dx ** 2 + dy ** 2)
    ds_init = float(np.mean(seg_len))
    if ds_init < 1e-9:
        ds_init = 1e-3

    return np.concatenate([x_init, y_init, theta_init, [ds_init]])
