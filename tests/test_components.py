import numpy as np
from opt_sim.components import tunable_mzi_in, tunable_mzi_out

def test_mzi_matrices():
    mat_in = tunable_mzi_in(np.pi / 2)
    mat_out = tunable_mzi_out(np.pi / 2)
    assert mat_in.shape == (2, 2)
    assert mat_out.shape == (2, 2)

