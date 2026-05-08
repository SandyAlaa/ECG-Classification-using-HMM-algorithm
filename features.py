"""
Feature extraction for ECG heartbeat sequences.

Each raw beat (1-D array of length BEAT_LENGTH) is converted into
a 2-D observation matrix of shape (T_STEPS, N_FEATURES) suitable
for the HMM.

Temporal structure is preserved so the HMM can model the sequential
dynamics of the heartbeat morphology.

Features per time-step (N_FEATURES = 6)
----------------------------------------
  0  amplitude          — normalised signal value
  1  first derivative   — local slope
  2  second derivative  — local curvature
  3  abs(amplitude)     — envelope
  4  rolling energy     — squared amplitude in a local window
  5  cumulative energy  — energy accumulated up to this step
"""

from __future__ import annotations
import numpy as np
from ecg_data import normalize_beat, BEAT_LENGTH

T_STEPS    = 30     # temporal resolution fed to the HMM
N_FEATURES = 6      # features per time-step


def _rolling_energy(x: np.ndarray, window: int = 5) -> np.ndarray:
    """Local squared energy using a simple box filter."""
    pad = np.pad(x**2, window // 2, mode="edge")
    return np.array([pad[i: i + window].mean()
                     for i in range(len(x))])


def extract_sequence(beat: np.ndarray,
                     t_steps: int = T_STEPS,
                     normalise: bool = True) -> np.ndarray:
    """
    Convert a raw beat waveform to a (t_steps, N_FEATURES) observation
    matrix suitable for the HMM.

    Parameters
    ----------
    beat      : 1-D array, shape (BEAT_LENGTH,)
    t_steps   : number of temporal steps (columns = time axis for HMM)
    normalise : if True, z-score the amplitude before feature extraction

    Returns
    -------
    obs : np.ndarray, shape (t_steps, N_FEATURES)
    """
    if normalise:
        beat = normalize_beat(beat.astype(float))
    else:
        beat = beat.astype(float)

    # -------- raw features on full resolution --------
    amp  = beat
    d1   = np.gradient(amp)
    d2   = np.gradient(d1)
    env  = np.abs(amp)
    re   = _rolling_energy(amp, window=max(1, BEAT_LENGTH // 20))
    ce   = np.cumsum(amp**2) / (np.arange(1, len(amp) + 1))

    full = np.column_stack([amp, d1, d2, env, re, ce])   # (BEAT_LENGTH, 6)

    # -------- down-sample to t_steps using block-mean --------
    block = BEAT_LENGTH // t_steps
    obs   = np.zeros((t_steps, N_FEATURES), dtype=np.float32)
    for i in range(t_steps):
        start      = i * block
        end        = start + block if i < t_steps - 1 else BEAT_LENGTH
        obs[i]     = full[start:end].mean(axis=0)

    return obs


def extract_dataset(beats: list,
                    t_steps: int = T_STEPS,
                    normalise: bool = True) -> list:
    """
    Apply extract_sequence to every beat in a list.

    Returns a list of (t_steps, N_FEATURES) arrays.
    """
    return [extract_sequence(b, t_steps=t_steps, normalise=normalise)
            for b in beats]


def feature_info() -> dict:
    return {
        "t_steps"    : T_STEPS,
        "n_features" : N_FEATURES,
        "feature_names": [
            "Amplitude",
            "1st Derivative",
            "2nd Derivative",
            "Envelope (|amp|)",
            "Rolling Energy",
            "Cumulative Energy",
        ],
    }