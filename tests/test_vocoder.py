import numpy as np
import pytest
from conftest import estimate_f0_autocorr, make_tone

from vocal_shift import (
    add_breathiness,
    analyse_world,
    apply_vocoder,
    build_arg_parser,
    note_name_to_hz,
    parse_pitch_or_note,
    smooth_envelope,
    synthesise_world,
    validate_args,
)

SR = 44100


# ---- note / pitch parsing ---------------------------------------------


def test_note_name_to_hz_known_values():
    assert np.isclose(note_name_to_hz("A4"), 440.0)
    assert np.isclose(note_name_to_hz("A3"), 220.0)
    assert np.isclose(note_name_to_hz("C4"), 440.0 * 2 ** (-9 / 12))


def test_note_name_to_hz_accepts_sharps_and_flats():
    assert np.isclose(note_name_to_hz("C#4"), note_name_to_hz("Db4"))
    assert np.isclose(note_name_to_hz("Bb3"), note_name_to_hz("A#3"))


def test_note_name_to_hz_rejects_garbage():
    with pytest.raises(ValueError):
        note_name_to_hz("notanote")


def test_parse_pitch_or_note_accepts_hz_or_note():
    assert np.isclose(parse_pitch_or_note("110"), 110.0)
    assert np.isclose(parse_pitch_or_note("A2"), 110.0)


def test_parse_pitch_or_note_rejects_non_positive_hz():
    with pytest.raises(ValueError):
        parse_pitch_or_note("-10")
    with pytest.raises(ValueError):
        parse_pitch_or_note("0")


# ---- smooth_envelope -----------------------------------------------------


def _formant_like_envelope(n_frames=3, n_bins=513):
    sp = np.full((n_frames, n_bins), 1e-3)
    for peak_bin in (40, 120, 250):
        bins = np.arange(n_bins)
        sp += np.exp(-0.5 * ((bins - peak_bin) / 2.0) ** 2)
    return sp


def test_smooth_envelope_zero_is_identity():
    sp = _formant_like_envelope()
    out = smooth_envelope(sp, 0.0)
    assert np.array_equal(out, sp)


def test_smooth_envelope_flattens_peaks_but_preserves_energy():
    sp = _formant_like_envelope()
    out = smooth_envelope(sp, 1.0)
    # Peak-to-mean ratio drops a lot once formants are blurred together.
    orig_ratio = sp.max(axis=1) / sp.mean(axis=1)
    new_ratio = out.max(axis=1) / out.mean(axis=1)
    assert np.all(new_ratio < orig_ratio * 0.5)
    assert np.allclose(out.sum(axis=1), sp.sum(axis=1), rtol=1e-6)


# ---- add_breathiness ------------------------------------------------------


def test_add_breathiness_zero_is_identity():
    ap = np.array([[0.1, 0.5, 0.9]])
    assert np.array_equal(add_breathiness(ap, 0.0), ap)


def test_add_breathiness_blends_toward_one():
    ap = np.array([[0.0, 0.5, 1.0]])
    out = add_breathiness(ap, 0.5)
    assert np.allclose(out, [[0.5, 0.75, 1.0]])


def test_add_breathiness_full_amount_saturates_at_one():
    ap = np.array([[0.0, 0.3]])
    out = add_breathiness(ap, 1.0)
    assert np.allclose(out, [[1.0, 1.0]])


# ---- apply_vocoder ---------------------------------------------------------


def test_apply_vocoder_forces_constant_pitch_on_voiced_frames_only():
    f0 = np.array([0.0, 220.0, 225.0, 0.0, 230.0])
    sp = np.ones((5, 8))
    ap = np.zeros((5, 8))
    f0_new, sp_new, ap_new = apply_vocoder(f0, sp, ap, pitch_hz=110.0, smooth=0.0, breath=0.0)
    assert f0_new[0] == 0.0 and f0_new[3] == 0.0
    assert np.all(f0_new[[1, 2, 4]] == 110.0)
    assert np.array_equal(sp_new, sp)
    assert np.array_equal(ap_new, ap)


def test_apply_vocoder_no_pitch_keeps_contour():
    f0 = np.array([0.0, 220.0, 225.0])
    sp = np.ones((3, 8))
    ap = np.zeros((3, 8))
    f0_new, _, _ = apply_vocoder(f0, sp, ap, pitch_hz=None, smooth=0.0, breath=0.0)
    assert np.array_equal(f0_new, f0)


# ---- CLI validation --------------------------------------------------------


def _parse(tmp_path, args):
    input_wav = tmp_path / "in.wav"
    input_wav.write_bytes(b"RIFF....WAVEfmt ")
    parser = build_arg_parser()
    return parser, parser.parse_args([str(input_wav)] + args)


def test_vocoder_only_option_without_vocoder_is_rejected(tmp_path, capsys):
    parser, args = _parse(tmp_path, ["out.wav", "--vocoder-smooth", "0.5"])
    with pytest.raises(SystemExit):
        validate_args(args, parser)
    assert "require --vocoder" in capsys.readouterr().err


def test_vocoder_smooth_out_of_range_is_rejected(tmp_path, capsys):
    parser, args = _parse(tmp_path, ["out.wav", "--vocoder", "--vocoder-smooth", "1.5"])
    with pytest.raises(SystemExit):
        validate_args(args, parser)
    assert "--vocoder-smooth must be between 0 and 1" in capsys.readouterr().err


def test_vocoder_breath_out_of_range_is_rejected(tmp_path, capsys):
    parser, args = _parse(tmp_path, ["out.wav", "--vocoder", "--vocoder-breath", "-0.1"])
    with pytest.raises(SystemExit):
        validate_args(args, parser)
    assert "--vocoder-breath must be between 0 and 1" in capsys.readouterr().err


def test_vocoder_bad_pitch_string_is_rejected(tmp_path, capsys):
    parser, args = _parse(tmp_path, ["out.wav", "--vocoder", "--vocoder-pitch", "notanote"])
    with pytest.raises(SystemExit):
        validate_args(args, parser)
    assert "unrecognised note" in capsys.readouterr().err


def test_valid_vocoder_args_resolve_pitch_hz(tmp_path):
    parser, args = _parse(tmp_path, ["out.wav", "--vocoder", "--vocoder-pitch", "A2"])
    validate_args(args, parser)
    assert np.isclose(args.vocoder_pitch_hz, 110.0)


def test_vocoder_without_pitch_leaves_pitch_hz_none(tmp_path):
    parser, args = _parse(tmp_path, ["out.wav", "--vocoder", "--vocoder-smooth", "0.4"])
    validate_args(args, parser)
    assert args.vocoder_pitch_hz is None


# ---- integration: forced carrier pitch resynthesizes correctly -----------


def test_vocoder_carrier_pitch_matches_target_within_a_few_cents():
    x = make_tone(220.0, sr=SR)
    frames = analyse_world(x, SR, 5.0, 60.0, 1100.0)
    target_hz = 110.0
    frames.f0, frames.sp, frames.ap = apply_vocoder(
        frames.f0, frames.sp, frames.ap, pitch_hz=target_hz, smooth=0.0, breath=0.0
    )
    y = synthesise_world(frames)
    measured = estimate_f0_autocorr(y, SR, f0_min=60.0, f0_max=300.0)
    cents_error = abs(1200 * np.log2(measured / target_hz))
    assert cents_error < 10
