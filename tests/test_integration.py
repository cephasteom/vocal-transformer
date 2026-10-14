import numpy as np
from conftest import estimate_f0_autocorr, make_tone

from vocal_shift import (
    analyse_world,
    autotune,
    choose_working_rate,
    hz_to_cents,
    lowpass_before_downsample,
    parse_key,
    peak_normalise_if_clipping,
    ratio_from_semitones,
    resample,
    shift_formant,
    shift_pitch,
    synthesise_world,
)

SR = 44100
FRAME_PERIOD = 5.0


def _analyse(x, sr=SR, f0_min=60.0, f0_max=1100.0):
    return analyse_world(x, sr, FRAME_PERIOD, f0_min, f0_max)


# ---- round trip -----------------------------------------------------------


def test_round_trip_preserves_dominant_frequency():
    x = make_tone(220.0, sr=SR)
    frames = _analyse(x)
    y = synthesise_world(frames)
    f0_in = estimate_f0_autocorr(x, SR)
    f0_out = estimate_f0_autocorr(y, SR)
    cents_diff = abs(hz_to_cents(np.array([f0_out]), reference=f0_in)[0])
    assert cents_diff < 20  # WORLD resynthesis is not bit-exact, pitch should still hold


# ---- pitch shift ------------------------------------------------------------


def test_pitch_up_matches_target_within_a_few_cents():
    x = make_tone(220.0, sr=SR)
    frames = _analyse(x)
    ratio = ratio_from_semitones(7)  # perfect fifth up
    f0_new = shift_pitch(frames.f0, ratio)
    frames.f0 = f0_new
    y = synthesise_world(frames)

    target_hz = 220.0 * ratio
    measured = estimate_f0_autocorr(y, SR)
    cents_error = abs(1200 * np.log2(measured / target_hz))
    assert cents_error < 10


def test_pitch_down_matches_target_within_a_few_cents():
    x = make_tone(330.0, sr=SR)
    frames = _analyse(x)
    ratio = ratio_from_semitones(-5)
    frames.f0 = shift_pitch(frames.f0, ratio)
    y = synthesise_world(frames)

    target_hz = 330.0 * ratio
    measured = estimate_f0_autocorr(y, SR)
    cents_error = abs(1200 * np.log2(measured / target_hz))
    assert cents_error < 10


# ---- auto-tune --------------------------------------------------------------


def test_autotune_corrects_detuned_tone_within_five_cents():
    # A4 (440 Hz) detuned sharp by +30 cents.
    detuned_hz = 440.0 * (2 ** (30 / 1200))
    x = make_tone(detuned_hz, sr=SR)
    frames = _analyse(x, f0_min=200.0, f0_max=800.0)

    key = parse_key("C")
    f0_new, stats = autotune(
        frames.f0,
        FRAME_PERIOD,
        key=key,
        scale="major",
        strength=1.0,
        retune_speed_ms=5.0,
        vibrato_keep=0.0,
        min_note_ms=80.0,
        a4=440.0,
    )
    frames.f0 = f0_new
    y = synthesise_world(frames)

    measured = estimate_f0_autocorr(y, SR, f0_min=200.0, f0_max=800.0)
    cents_error = abs(1200 * np.log2(measured / 440.0))
    assert cents_error < 5
    assert stats.n_notes >= 1


def test_autotune_chromatic_default_when_no_key():
    detuned_hz = 220.0 * (2 ** (40 / 1200))
    x = make_tone(detuned_hz, sr=SR)
    frames = _analyse(x)
    f0_new, stats = autotune(
        frames.f0, FRAME_PERIOD, key=None, scale=None, strength=1.0,
        retune_speed_ms=5.0, vibrato_keep=0.0, min_note_ms=80.0, a4=440.0,
    )
    assert stats.scale_label == "chromatic"
    voiced = f0_new > 0
    cents = hz_to_cents(f0_new[voiced], reference=440.0)
    # 220 Hz (A3) is itself an equal-tempered semitone, one octave below the
    # A4=440 Hz reference, i.e. exactly -1200 cents.
    assert np.median(np.abs(cents - (-1200.0))) < 5


# ---- clipping / level integrity ---------------------------------------------


def test_peak_normalise_only_acts_when_clipping():
    quiet = np.array([0.1, -0.2, 0.05])
    out, n_clipped = peak_normalise_if_clipping(quiet)
    assert n_clipped == 0
    assert np.allclose(out, quiet)  # untouched: no unconditional gain change

    loud = np.array([0.5, -0.3, 1.5, -1.2])
    out, n_clipped = peak_normalise_if_clipping(loud)
    assert n_clipped == 2
    assert np.max(np.abs(out)) <= 1.0 + 1e-9


# ---- anti-aliasing ------------------------------------------------------


def test_lowpass_before_downsample_removes_content_above_new_nyquist():
    fs_working = 44100
    fs_out = 16000
    n = fs_working * 2
    t = np.arange(n) / fs_working
    # Energy both inside and outside the target Nyquist (8 kHz).
    y = np.sin(2 * np.pi * 2000 * t) + np.sin(2 * np.pi * 12000 * t)
    filtered = lowpass_before_downsample(y, fs_working, fs_out)

    spec = np.abs(np.fft.rfft(filtered))
    freqs = np.fft.rfftfreq(len(filtered), d=1 / fs_working)
    low_band = spec[(freqs > 1500) & (freqs < 2500)].max()
    high_band = spec[(freqs > 11500) & (freqs < 12500)].max()
    assert high_band < low_band * 0.01


def test_lowpass_before_downsample_is_noop_when_not_downsampling():
    y = np.random.randn(1000)
    out = lowpass_before_downsample(y, 44100, 48000)
    assert np.array_equal(out, y)


def test_full_pipeline_no_energy_above_new_nyquist_after_upward_shifts(tmp_path):
    sr_in = 16000  # forces an internal upsample to 44100 and back down
    x = make_tone(150.0, sr=sr_in, harmonics=(1.0, 0.6, 0.4, 0.3, 0.2))

    fs_working = choose_working_rate(sr_in)
    assert fs_working == 44100
    x_work = resample(x, sr_in, fs_working)
    frames = _analyse(x_work, sr=fs_working)

    frames.f0 = shift_pitch(frames.f0, ratio_from_semitones(12))
    frames.sp = shift_formant(frames.sp, 1.35, fs_working)
    y = synthesise_world(frames)
    y = lowpass_before_downsample(y, fs_working, sr_in)
    y_out = resample(y, fs_working, sr_in)

    spec = np.abs(np.fft.rfft(y_out))
    freqs = np.fft.rfftfreq(len(y_out), d=1 / sr_in)
    nyquist = sr_in / 2
    total_energy = (spec ** 2).sum()
    edge_energy = (spec[freqs > 0.95 * nyquist] ** 2).sum()
    assert edge_energy < 0.01 * total_energy


# ---- WAV I/O round trip through disk ----------------------------------------


def test_write_and_load_wav_preserves_samples(tmp_path):
    from vocal_shift import load_wav, write_wav

    x = make_tone(200.0, sr=SR).reshape(-1, 1)
    path = tmp_path / "t.wav"
    write_wav(path, x, SR, bit_depth="float32")
    data, sr = load_wav(path)
    assert sr == SR
    assert data.shape == x.shape
    assert np.allclose(data, x, atol=1e-4)


def test_load_wav_rejects_non_wav(tmp_path):
    import pytest

    path = tmp_path / "not_a_wav.txt"
    path.write_text("hello")
    from vocal_shift import load_wav

    with pytest.raises(ValueError):
        load_wav(path)
