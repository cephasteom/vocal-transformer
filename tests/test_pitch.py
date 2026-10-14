import numpy as np

from vocal_shift import ratio_from_semitones, shift_pitch


def test_shift_pitch_octave_up():
    f0 = np.array([0.0, 220.0, 220.0, 0.0])
    ratio = ratio_from_semitones(12)
    out = shift_pitch(f0, ratio)
    assert out[0] == 0.0
    assert out[3] == 0.0
    assert np.allclose(out[1:3], 440.0, atol=1e-6)


def test_shift_pitch_cents_only():
    ratio = ratio_from_semitones(0, cents=100)
    assert np.isclose(ratio, 2 ** (1 / 12))


def test_shift_pitch_leaves_unvoiced_untouched():
    f0 = np.zeros(10)
    out = shift_pitch(f0, ratio_from_semitones(7))
    assert np.all(out == 0.0)
