"""Dubins Touring Region Problem (DTRP) solver.

Given a set of circles to visit, find the shortest curvature-constrained
path that passes through every circle.  The problem is posed as a nonlinear
optimal control problem, transcribed via direct collocation and solved with
the off-the-shelf SLSQP optimizer from scipy.
"""

from .solver import DTRPSolver
from .visualize import plot_solution

__all__ = ["DTRPSolver", "plot_solution"]
