"""Tests for the DTRP solver."""

import math

import numpy as np
import pytest

from dtrp.solver import (
    DTRPSolver,
    _initial_guess,
    _nearest_neighbour,
    _tour_length,
    _two_opt,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _check_visit(result: dict, circles: list, tol: float = 1e-2) -> None:
    """Assert that the trajectory visits every circle."""
    x, y = result["x"], result["y"]
    for j, idx in enumerate(result["visit_order"]):
        cx, cy, r = circles[idx]
        dists = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        assert np.min(dists) <= r + tol, (
            f"Circle {idx} (cx={cx}, cy={cy}, r={r}) was not visited; "
            f"closest point was {np.min(dists):.4f} away."
        )


def _check_curvature(result: dict, kappa_max: float, tol: float = 1e-3) -> None:
    """Assert that the curvature constraint is satisfied everywhere."""
    theta = result["theta"]
    ds = result["ds"]
    if ds <= 0:
        return
    kappas = np.abs(np.diff(theta)) / ds
    assert np.all(kappas <= kappa_max + tol), (
        f"Curvature violated: max |κ| = {kappas.max():.4f}, κ_max = {kappa_max}"
    )


# ---------------------------------------------------------------------------
# Tour-order helpers
# ---------------------------------------------------------------------------

class TestTourHelpers:
    def test_tour_length_two_points(self):
        centres = np.array([[0.0, 0.0], [3.0, 4.0]])
        assert math.isclose(_tour_length(centres, [0, 1]), 5.0)

    def test_tour_length_single(self):
        centres = np.array([[1.0, 2.0]])
        assert _tour_length(centres, [0]) == 0.0

    def test_nearest_neighbour_returns_all(self):
        centres = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.5, 1.0]])
        order = _nearest_neighbour(centres)
        assert sorted(order) == [0, 1, 2, 3]

    def test_two_opt_improves_or_equal(self):
        rng = np.random.default_rng(42)
        centres = rng.uniform(0, 10, (6, 2))
        order = list(range(6))
        optimised = _two_opt(centres, order)
        assert _tour_length(centres, optimised) <= _tour_length(centres, order) + 1e-9


# ---------------------------------------------------------------------------
# Initial-guess construction
# ---------------------------------------------------------------------------

class TestInitialGuess:
    def test_shape(self):
        circles = [(0.0, 0.0, 1.0), (5.0, 0.0, 1.0)]
        S = 20
        N = len(circles) * S
        z0 = _initial_guess(circles, S, N, None, None, None)
        assert z0.shape == (3 * (N + 1) + 1,)

    def test_ds_positive(self):
        circles = [(0.0, 0.0, 1.0), (4.0, 3.0, 1.0), (8.0, 0.0, 1.0)]
        S = 10
        N = len(circles) * S
        z0 = _initial_guess(circles, S, N, None, None, None)
        ds = z0[-1]
        assert ds > 0

    def test_start_position_pinned(self):
        circles = [(0.0, 0.0, 1.0), (5.0, 0.0, 1.0)]
        S = 15
        N = len(circles) * S
        z0 = _initial_guess(circles, S, N, x0=2.0, y0=3.0, theta0=0.5)
        x = z0[: N + 1]
        y = z0[N + 1: 2 * (N + 1)]
        theta = z0[2 * (N + 1): 3 * (N + 1)]
        assert math.isclose(x[0], 2.0, abs_tol=1e-10)
        assert math.isclose(y[0], 3.0, abs_tol=1e-10)
        assert math.isclose(theta[0], 0.5, abs_tol=1e-10)


# ---------------------------------------------------------------------------
# Trivial single-circle case
# ---------------------------------------------------------------------------

class TestSingleCircle:
    def test_returns_success(self):
        solver = DTRPSolver([(0.0, 0.0, 2.0)], kappa_max=1.0, steps_per_circle=20)
        result = solver.solve()
        assert result["success"]

    def test_visits_circle(self):
        circles = [(0.0, 0.0, 2.0)]
        solver = DTRPSolver(circles, kappa_max=1.0, steps_per_circle=20)
        result = solver.solve()
        _check_visit(result, circles)

    def test_total_length_positive(self):
        solver = DTRPSolver([(3.0, 4.0, 1.5)], kappa_max=1.0, steps_per_circle=20)
        result = solver.solve()
        assert result["total_length"] > 0.0

    def test_trajectory_shape(self):
        S = 20
        solver = DTRPSolver([(0.0, 0.0, 1.0)], kappa_max=1.0, steps_per_circle=S)
        result = solver.solve()
        N = 1 * S
        assert result["x"].shape == (N + 1,)
        assert result["y"].shape == (N + 1,)
        assert result["theta"].shape == (N + 1,)


# ---------------------------------------------------------------------------
# Two-circle case
# ---------------------------------------------------------------------------

class TestTwoCircles:
    @pytest.fixture
    def circles(self):
        return [(0.0, 0.0, 1.0), (6.0, 0.0, 1.0)]

    def test_visits_both_circles(self, circles):
        solver = DTRPSolver(circles, kappa_max=2.0, steps_per_circle=25)
        result = solver.solve()
        _check_visit(result, circles)

    def test_curvature_satisfied(self, circles):
        kappa_max = 2.0
        solver = DTRPSolver(circles, kappa_max=kappa_max, steps_per_circle=25)
        result = solver.solve()
        _check_curvature(result, kappa_max)

    def test_path_length_reasonable(self, circles):
        """Path length should be at least the straight-line distance minus radii."""
        solver = DTRPSolver(circles, kappa_max=2.0, steps_per_circle=25)
        result = solver.solve()
        # Straight-line distance between centres minus both radii
        lower_bound = 6.0 - 1.0 - 1.0
        assert result["total_length"] >= lower_bound - 0.1

    def test_fixed_start(self, circles):
        solver = DTRPSolver(circles, kappa_max=2.0, steps_per_circle=25)
        result = solver.solve(x0=0.0, y0=0.0, theta0=0.0)
        assert math.isclose(result["x"][0], 0.0, abs_tol=0.05)
        assert math.isclose(result["y"][0], 0.0, abs_tol=0.05)


# ---------------------------------------------------------------------------
# Three-circle case (triangle arrangement)
# ---------------------------------------------------------------------------

class TestThreeCircles:
    @pytest.fixture
    def circles(self):
        return [
            (0.0, 0.0, 1.0),
            (5.0, 0.0, 1.0),
            (2.5, 4.0, 1.0),
        ]

    def test_visits_all_three(self, circles):
        solver = DTRPSolver(circles, kappa_max=1.5, steps_per_circle=25)
        result = solver.solve()
        _check_visit(result, circles)

    def test_curvature_satisfied(self, circles):
        kappa_max = 1.5
        solver = DTRPSolver(circles, kappa_max=kappa_max, steps_per_circle=25)
        result = solver.solve()
        _check_curvature(result, kappa_max)

    def test_visit_order_length(self, circles):
        solver = DTRPSolver(circles, kappa_max=1.5, steps_per_circle=25)
        result = solver.solve()
        assert len(result["visit_order"]) == 3
        assert sorted(result["visit_order"]) == [0, 1, 2]


# ---------------------------------------------------------------------------
# Explicit visit order
# ---------------------------------------------------------------------------

class TestExplicitVisitOrder:
    def test_explicit_order_respected(self):
        circles = [(0.0, 0.0, 1.0), (4.0, 0.0, 1.0), (8.0, 0.0, 1.0)]
        solver = DTRPSolver(circles, kappa_max=2.0, steps_per_circle=20)
        order = [2, 0, 1]
        result = solver.solve(visit_order=order)
        assert result["visit_order"] == order

    def test_explicit_order_visits_all(self):
        circles = [(0.0, 0.0, 1.0), (4.0, 0.0, 1.0), (8.0, 0.0, 1.0)]
        solver = DTRPSolver(circles, kappa_max=2.0, steps_per_circle=20)
        result = solver.solve(visit_order=[2, 0, 1])
        _check_visit(result, circles)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_circles(self):
        solver = DTRPSolver([])
        result = solver.solve()
        assert result["success"]
        assert len(result["x"]) == 0
        assert result["total_length"] == 0.0

    def test_large_radius_trivial(self):
        """Circle so large the initial position is already inside it."""
        circles = [(0.0, 0.0, 100.0)]
        solver = DTRPSolver(circles, kappa_max=1.0, steps_per_circle=15)
        result = solver.solve()
        _check_visit(result, circles)

    def test_high_curvature_limit(self):
        """With large κ_max, the path should still visit both circles."""
        circles = [(0.0, 0.0, 1.0), (3.0, 3.0, 1.0)]
        solver = DTRPSolver(circles, kappa_max=10.0, steps_per_circle=20)
        result = solver.solve()
        _check_visit(result, circles)

    def test_result_keys(self):
        solver = DTRPSolver([(0.0, 0.0, 1.0)], steps_per_circle=15)
        result = solver.solve()
        for key in ("x", "y", "theta", "ds", "total_length", "visit_order",
                    "success", "message"):
            assert key in result, f"Missing key: {key}"
