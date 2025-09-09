"""Utilities for the revised Bayesian optimisation prototype.

This package contains simulation models and hardware connector stubs used to
experiment with Bayesian optimisation strategies without access to the full
production environment.
"""

from .connector import MockHardware, RealHardware
from .simulate import get_response

__all__ = ["MockHardware", "RealHardware", "get_response"]
