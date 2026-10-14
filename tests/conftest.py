import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def make_tone(
    f0: float,
    sr: int = 44100,
    duration: float = 1.5,
    harmonics=(1.0, 0.4, 0.15),
    vibrato_depth: float = 0.01,
    vibrato_hz: float = 5.0,
) -> np.ndarray:
    """A synthetic harmonic tone with a short fade in/out, standing in for
    a sustained sung note.

    A tiny (1%) vibrato is included by default: WORLD's Harvest F0 tracker
    is tuned for natural voices and largely fails to mark a perfectly
    static, jitter-free sine as voiced (it falls back to ~0% voicing),
    which would make these fixtures useless for exercising the pipeline.
    Real sung notes always carry this much pitch movement, so this doesn't
    make the fixture less representative -- it makes it more so.
    """
    n = int(sr * duration)
    t = np.arange(n) / sr
    vibrato = 1 + vibrato_depth * np.sin(2 * np.pi * vibrato_hz * t)
    phase = 2 * np.pi * np.cumsum(f0 * vibrato) / sr
    y = np.zeros(n)
    for k, amp in enumerate(harmonics, start=1):
        y += amp * np.sin(k * phase)
    y *= 0.3 / np.max(np.abs(y))
    fade = min(1000, n // 10)
    window = np.ones(n)
    window[:fade] = np.linspace(0, 1, fade)
    window[-fade:] = np.linspace(1, 0, fade)
    return y * window


def estimate_f0_autocorr(x: np.ndarray, sr: int, f0_min: float = 50.0, f0_max: float = 1200.0) -> float:
    """A minimal autocorrelation F0 estimator used only to independently
    verify the pipeline's pitch output in tests (not part of the pipeline
    itself). Parabolic interpolation around the correlation peak gives
    sub-sample lag resolution -- plain integer-lag picking is only accurate
    to a few percent, too coarse to check a "few cents" tolerance."""
    n = len(x)
    seg = x[int(n * 0.3) : int(n * 0.7)].astype(np.float64)
    seg = seg - seg.mean()
    corr = np.correlate(seg, seg, mode="full")[len(seg) - 1 :]
    min_lag = max(int(sr / f0_max), 1)
    max_lag = min(int(sr / f0_min), len(corr) - 2)
    window = corr[min_lag:max_lag]
    peak_lag = min_lag + int(np.argmax(window))

    # Octave-error guard: a harmonic-rich tone can autocorrelate just as
    # strongly (or more) at an integer multiple of the true period as at
    # the true period itself. If a comparably strong peak exists at an
    # integer submultiple of the detected lag, prefer the shorter (higher
    # frequency, true fundamental) one.
    for divisor in (2, 3, 4):
        candidate = peak_lag // divisor
        if candidate < min_lag:
            break
        probe = corr[max(candidate - 2, 0) : candidate + 3]
        if probe.size and probe.max() >= 0.85 * corr[peak_lag]:
            peak_lag = candidate - 2 + int(np.argmax(probe))

    peak_lag = max(min(peak_lag, len(corr) - 2), 1)
    y0, y1, y2 = corr[peak_lag - 1], corr[peak_lag], corr[peak_lag + 1]
    denom = y0 - 2 * y1 + y2
    delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    refined_lag = peak_lag + delta
    return sr / refined_lag


@pytest.fixture
def tone_220():
    return make_tone(220.0), 44100
