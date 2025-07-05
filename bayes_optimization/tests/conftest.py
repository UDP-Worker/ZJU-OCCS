import csv
import numpy as np
import os
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayes_optimization.bayes_optimizer.simulate import optical_chip

@pytest.fixture(autouse=True)
def load_waveform1():
    path = ROOT / "bayes_optimizer" / "simulate" / "ideal_waveform1.csv"
    with open(path, "r", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    wl = np.asarray(rows[0], dtype=float)
    resp = np.asarray(rows[1], dtype=float)
    optical_chip.set_target_waveform(wl, resp)
    yield


