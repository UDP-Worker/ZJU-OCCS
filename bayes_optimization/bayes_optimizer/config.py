"""Global configuration for bayes optimizer."""

from pathlib import Path

# Reduce channel count for faster simulation-based testing
NUM_CHANNELS = 5
V_RANGE = (0.0, 2.0)
OSA_TIMEOUT = 10.0
TARGET_WAVEFORM_PATH = str(Path(__file__).resolve().parent / 'simulate' / 'ideal_waveform.csv')
BO_MAX_STEPS = 60
SPSA_STEPS = 20
LOG_DIR = str(Path(__file__).resolve().parent.parent / 'data' / 'logs')
