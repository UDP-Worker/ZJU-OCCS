"""DAC interface placeholder."""

import numpy as np


def apply(volts: np.ndarray) -> None:
    """Send voltages to DAC (mock)."""
    # In placeholder, just log values
    print(f"[DAC] apply {volts}")
