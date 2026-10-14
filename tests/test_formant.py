import numpy as np
import pytest

from vocal_shift import FORMANT_RATIO_RANGE, shift_formant


def _impulse_envelope(n_bins=513, peak_bin=50, width=3):
    sp = np.full((1, n_bins), 1e-6)
    bins = np.arange(n_bins)
    sp[0] += np.exp(-0.5 * ((bins - peak_bin) / width) ** 2)
    return sp


def test_formant_warp_moves_peak():
    n_bins = 513
    fs = 44100
    peak_bin = 50
    sp = _impulse_envelope(n_bins, peak_bin)
    sp_new = shift_formant(sp, alpha=2.0, fs=fs, clamp=False)
    new_peak = int(np.argmax(sp_new[0]))
    assert abs(new_peak - 2 * peak_bin) <= 1


def test_formant_warp_down_moves_peak():
    n_bins = 513
    fs = 44100
    peak_bin = 120
    sp = _impulse_envelope(n_bins, peak_bin)
    sp_new = shift_formant(sp, alpha=0.5, fs=fs, clamp=False)
    new_peak = int(np.argmax(sp_new[0]))
    assert abs(new_peak - peak_bin // 2) <= 1


def test_formant_taper_attenuates_above_cutoff():
    n_bins = 513
    fs = 44100
    sp = np.full((4, n_bins), 1.0)
    sp_new = shift_formant(sp, alpha=2.0, fs=fs, clamp=False)
    nyquist = fs / 2.0
    bin_hz = nyquist / (n_bins - 1)
    cutoff_hz = min(nyquist * 0.9 / 2.0, nyquist)
    cutoff_bin = int(cutoff_hz / bin_hz)
    low_energy = sp_new[:, : cutoff_bin - 20].mean()
    high_energy = sp_new[:, cutoff_bin + 20 :].mean()
    assert high_energy < low_energy * 0.1


def test_formant_identity_at_ratio_one():
    sp = _impulse_envelope()
    sp_new = shift_formant(sp, alpha=1.0, fs=44100)
    assert np.allclose(sp, sp_new)


def test_formant_ratio_out_of_range_is_clamped_with_warning():
    sp = _impulse_envelope()
    with pytest.warns(UserWarning):
        shift_formant(sp, alpha=5.0, fs=44100, clamp=True)
    lo, hi = FORMANT_RATIO_RANGE
    assert lo == 0.7 and hi == 1.4
