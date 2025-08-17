"""Hardware connector abstractions.

This package exposes high level interfaces for both the mocked and the real
hardware backends used by the optimisation workflow.
"""

from .mock_hardware import MockHardware
from .real_hardware import RealHardware

__all__ = ["MockHardware", "RealHardware"]

