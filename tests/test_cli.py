import pytest

from vocal_shift import build_arg_parser, validate_args


def _parse(tmp_path, args):
    input_wav = tmp_path / "in.wav"
    input_wav.write_bytes(b"RIFF....WAVEfmt ")  # existence is all validate_args checks
    parser = build_arg_parser()
    return parser, parser.parse_args([str(input_wav)] + args)


def test_key_without_scale_is_rejected(tmp_path, capsys):
    parser, args = _parse(tmp_path, ["out.wav", "--key", "C"])
    with pytest.raises(SystemExit):
        validate_args(args, parser)
    assert "--key and --scale must be given together" in capsys.readouterr().err


def test_scale_without_key_is_rejected(tmp_path, capsys):
    parser, args = _parse(tmp_path, ["out.wav", "--scale", "major"])
    with pytest.raises(SystemExit):
        validate_args(args, parser)
    assert "--key and --scale must be given together" in capsys.readouterr().err


def test_formant_and_formant_semitones_mutually_exclusive(tmp_path, capsys):
    parser, args = _parse(tmp_path, ["out.wav", "--formant", "1.2", "--formant-semitones", "3"])
    with pytest.raises(SystemExit):
        validate_args(args, parser)
    assert "mutually exclusive" in capsys.readouterr().err


def test_autotune_only_option_without_autotune_is_rejected(tmp_path, capsys):
    parser, args = _parse(tmp_path, ["out.wav", "--strength", "0.5"])
    with pytest.raises(SystemExit):
        validate_args(args, parser)
    assert "require --autotune" in capsys.readouterr().err


def test_f0_min_must_be_less_than_f0_max(tmp_path, capsys):
    parser, args = _parse(tmp_path, ["out.wav", "--f0-min", "1000", "--f0-max", "500"])
    with pytest.raises(SystemExit):
        validate_args(args, parser)
    assert "--f0-min must be less than --f0-max" in capsys.readouterr().err


def test_output_required_unless_analyze(tmp_path, capsys):
    parser, args = _parse(tmp_path, [])
    with pytest.raises(SystemExit):
        validate_args(args, parser)
    assert "OUTPUT is required" in capsys.readouterr().err


def test_analyze_without_output_is_allowed(tmp_path):
    parser, args = _parse(tmp_path, ["--analyze"])
    validate_args(args, parser)  # must not raise


def test_missing_input_is_rejected(tmp_path, capsys):
    parser = build_arg_parser()
    args = parser.parse_args([str(tmp_path / "missing.wav"), "out.wav"])
    with pytest.raises(SystemExit):
        validate_args(args, parser)
    assert "input not found" in capsys.readouterr().err


def test_valid_autotune_args_pass(tmp_path):
    parser, args = _parse(
        tmp_path, ["out.wav", "--autotune", "--key", "D", "--scale", "minor", "--strength", "0.8"]
    )
    validate_args(args, parser)  # must not raise
