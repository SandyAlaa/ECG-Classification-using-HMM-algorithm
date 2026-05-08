"""
Synthetic ECG heartbeat generator.

Generates realistic-looking heartbeat waveforms for 5 classes
(modelled after the MIT-BIH annotation scheme):

  N  – Normal sinus rhythm
  S  – Supraventricular ectopic (APC)
  V  – Ventricular ectopic (PVC)
  F  – Fusion beat
  Q  – Pacemaker / unknown

Each beat is 180 samples long (≈ 0.5 s at 360 Hz).
Waveforms are built as a superposition of Gaussian "bumps", then
corrupted with realistic baseline wander and additive noise.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple


BEAT_LENGTH  = 180          # samples per beat
CLASSES      = ["N", "S", "V", "F", "Q"]
CLASS_NAMES  = {
    "N": "Normal",
    "S": "Supraventricular (APC)",
    "V": "Ventricular (PVC)",
    "F": "Fusion Beat",
    "Q": "Pacemaker",
}
CLASS_COLORS = {
    "N": "#2ecc71",
    "S": "#3498db",
    "V": "#e74c3c",
    "F": "#f39c12",
    "Q": "#9b59b6",
}


# --------------------------------------------------------------------------- #
#  Low-level helpers
# --------------------------------------------------------------------------- #
def _gauss(t: np.ndarray, centre: float, height: float, width: float
           ) -> np.ndarray:
    """Gaussian bump centred at `centre` (fraction of beat length)."""
    c = centre * BEAT_LENGTH
    return height * np.exp(-0.5 * ((t - c) / (width * BEAT_LENGTH))**2)


def _make_beat(components: list, rng: np.random.RandomState,
               noise_std: float = 0.04, bw_amp: float = 0.05) -> np.ndarray:
    """
    Synthesise one beat from a list of (centre, height, width) Gaussian
    components, then add baseline wander + Gaussian noise.
    """
    t    = np.arange(BEAT_LENGTH, dtype=float)
    beat = sum(_gauss(t, c, h, w) for c, h, w in components)
    # baseline wander (low-frequency sine)
    phase  = rng.uniform(0, 2 * np.pi)
    freq   = rng.uniform(0.5, 1.5) / BEAT_LENGTH
    bw     = bw_amp * rng.uniform(0.5, 1.5) * np.sin(2 * np.pi * freq * t + phase)
    beat  += bw + rng.normal(0, noise_std, BEAT_LENGTH)
    return beat.astype(np.float32)


# --------------------------------------------------------------------------- #
#  Per-class templates (centre, height, width)
# --------------------------------------------------------------------------- #
def _normal(rng: np.random.RandomState) -> np.ndarray:
    """Normal PQRST morphology."""
    r_ht = rng.uniform(0.85, 1.15)
    components = [
        (0.14, rng.uniform(0.18, 0.28), 0.045),   # P wave
        (0.30, rng.uniform(-0.12, -0.06), 0.020),  # Q wave
        (0.35, r_ht, 0.022),                        # R wave
        (0.41, rng.uniform(-0.25, -0.12), 0.020),  # S wave
        (0.60, rng.uniform(0.25, 0.45), 0.080),    # T wave
    ]
    return _make_beat(components, rng, noise_std=0.035, bw_amp=0.04)


def _apc(rng: np.random.RandomState) -> np.ndarray:
    """Supraventricular ectopic — early beat, abnormal P, narrow QRS."""
    r_ht = rng.uniform(0.70, 1.00)
    components = [
        (0.10, rng.uniform(0.08, 0.18), 0.055),   # abnormal P (wider, earlier)
        (0.28, rng.uniform(-0.08, -0.03), 0.018),  # Q
        (0.33, r_ht, 0.022),                        # R (slightly narrower)
        (0.38, rng.uniform(-0.18, -0.08), 0.018),  # S
        (0.55, rng.uniform(0.20, 0.38), 0.090),    # T (slightly prolonged)
    ]
    return _make_beat(components, rng, noise_std=0.040, bw_amp=0.05)


def _pvc(rng: np.random.RandomState) -> np.ndarray:
    """Ventricular ectopic — no P wave, wide QRS, inverted T."""
    sign = rng.choice([-1, 1])   # monophasic up or down
    r_ht = sign * rng.uniform(1.0, 1.4)
    components = [
        # No P wave
        (0.33, r_ht, 0.055),                              # wide R (or S)
        (0.46, -sign * rng.uniform(0.20, 0.40), 0.045),   # compensatory dip
        (0.66, -sign * rng.uniform(0.30, 0.55), 0.090),   # inverted T
    ]
    return _make_beat(components, rng, noise_std=0.045, bw_amp=0.06)


def _fusion(rng: np.random.RandomState) -> np.ndarray:
    """Fusion beat — mix of Normal and PVC morphology."""
    # Blend a normal beat and a PVC beat
    n = _normal(rng)
    v = _pvc(rng)
    alpha = rng.uniform(0.35, 0.65)
    return (alpha * n + (1 - alpha) * v).astype(np.float32)


def _pacemaker(rng: np.random.RandomState) -> np.ndarray:
    """Pacemaker beat — pacing spike + wide QRS, no natural P."""
    t = np.arange(BEAT_LENGTH, dtype=float)
    beat = np.zeros(BEAT_LENGTH, dtype=float)

    # Pacing spike (very narrow, tall)
    spike_pos = int(rng.uniform(0.20, 0.28) * BEAT_LENGTH)
    beat[spike_pos] = rng.uniform(1.8, 2.5)

    # Wide paced QRS
    components = [
        (0.36, rng.uniform(0.80, 1.10), 0.060),
        (0.50, rng.uniform(-0.25, -0.10), 0.050),
        (0.65, rng.uniform(0.20, 0.40), 0.085),
    ]
    beat += sum(_gauss(t, c, h, w) for c, h, w in components)

    phase = rng.uniform(0, 2 * np.pi)
    beat += 0.04 * np.sin(2 * np.pi * 0.8 / BEAT_LENGTH * t + phase)
    beat += rng.normal(0, 0.04, BEAT_LENGTH)
    return beat.astype(np.float32)


_GENERATORS = {
    "N": _normal,
    "S": _apc,
    "V": _pvc,
    "F": _fusion,
    "Q": _pacemaker,
}


# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #
def generate_beat(beat_type: str, rng: np.random.RandomState = None
                  ) -> np.ndarray:
    """Generate a single synthetic heartbeat of the given class."""
    if rng is None:
        rng = np.random.RandomState()
    assert beat_type in _GENERATORS, f"Unknown beat type '{beat_type}'"
    return _GENERATORS[beat_type](rng)


def generate_dataset(
    n_per_class: int = 200,
    classes: List[str] | None = None,
    seed: int = 0,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Generate a balanced synthetic dataset.

    Returns
    -------
    beats  : list of np.ndarray, shape (BEAT_LENGTH,) each
    labels : np.ndarray of string class labels
    """
    if classes is None:
        classes = CLASSES
    rng    = np.random.RandomState(seed)
    beats  = []
    labels = []
    for cls in classes:
        for _ in range(n_per_class):
            beats.append(generate_beat(cls, rng))
            labels.append(cls)
    # Shuffle
    idx    = rng.permutation(len(beats))
    beats  = [beats[i] for i in idx]
    labels = np.array(labels)[idx]
    return beats, labels


def normalize_beat(beat: np.ndarray) -> np.ndarray:
    """Z-score normalisation."""
    std = beat.std()
    if std < 1e-8:
        return beat - beat.mean()
    return (beat - beat.mean()) / std


def get_class_info() -> Dict:
    return {
        "classes": CLASSES,
        "names"  : CLASS_NAMES,
        "colors" : CLASS_COLORS,
        "beat_length": BEAT_LENGTH,
    }