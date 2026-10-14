import numpy as np
import pytest

from vocal_shift import (
    apply_strength,
    parse_key,
    segment_notes,
    snap_chromatic,
    snap_scale,
    snap_to_targets,
)

# ---- segment_notes -----------------------------------------------------


def _two_notes_with_passing_tone(frame_period_ms=5.0):
    # Note A: 0 cents for 200ms, passing tone: +70 cents for 20ms (a blip,
    # well under the 80ms min-note-ms default), Note B: +400 cents for 200ms.
    n_a = int(200 / frame_period_ms)
    n_blip = int(20 / frame_period_ms)
    n_b = int(200 / frame_period_ms)
    cents = np.concatenate([np.zeros(n_a), np.full(n_blip, 70.0), np.full(n_b, 400.0)])
    voiced = np.ones(len(cents), dtype=bool)
    return cents, voiced, frame_period_ms


def test_segment_notes_absorbs_short_passing_tone():
    cents, voiced, fp = _two_notes_with_passing_tone()
    notes = segment_notes(cents, voiced, fp, split_threshold_cents=50.0, min_note_ms=80.0)
    assert len(notes) == 2


def test_segment_notes_keeps_two_sustained_notes_separate():
    fp = 5.0
    n = int(200 / fp)
    cents = np.concatenate([np.zeros(n), np.full(n, 300.0)])
    voiced = np.ones(len(cents), dtype=bool)
    notes = segment_notes(cents, voiced, fp, split_threshold_cents=50.0, min_note_ms=80.0)
    assert len(notes) == 2
    assert notes[0].end == notes[1].start


def test_segment_notes_ignores_unvoiced_frames():
    fp = 5.0
    cents = np.zeros(40)
    voiced = np.zeros(40, dtype=bool)
    notes = segment_notes(cents, voiced, fp)
    assert notes == []


# ---- snap_to_targets ----------------------------------------------------


def test_snap_chromatic_known_values():
    assert np.isclose(snap_chromatic(0.0), 0.0)
    assert np.isclose(snap_chromatic(30.0), 0.0)
    assert np.isclose(snap_chromatic(-470.0), -500.0)


def test_snap_chromatic_exact_tie_breaks_low():
    # Exactly half-way between 0 and 100 cents.
    assert np.isclose(snap_chromatic(50.0), 0.0)
    assert np.isclose(snap_chromatic(-50.0), -100.0)


def test_snap_scale_constrains_to_scale_members():
    key = parse_key("C")
    # D (200 cents above A4's... ) -- use a cents value near D5/E5 boundary
    # and check the result's pitch class is in C major.
    target = snap_scale(250.0, key, "major")
    midi = round(69 + target / 100.0)
    assert midi % 12 in {0, 2, 4, 5, 7, 9, 11}


def test_snap_scale_tie_breaks_low():
    # C major contains C(0) and D(2); a cents value exactly between two
    # scale tones should resolve to the lower one.
    key = parse_key("C")
    # 0 cents = A4 (midi 69, pitch class 9, in C major). Build an exact
    # half-way point between two adjacent scale members: B(11) and C(12->0).
    # B4 is midi 71 -> cents (71-69)*100 = 200. C5 is midi 72 -> 300 cents.
    midpoint_cents = 250.0
    target = snap_scale(midpoint_cents, key, "major")
    assert target == 200.0


def test_snap_to_targets_chromatic_when_no_key():
    assert snap_to_targets(30.0) == snap_chromatic(30.0)


def test_parse_key_accepts_flats_and_sharps():
    assert parse_key("C#") == parse_key("Db")
    assert parse_key("C") == 0


def test_parse_key_rejects_unknown():
    with pytest.raises(ValueError):
        parse_key("H")


# ---- apply_strength ------------------------------------------------------


def test_apply_strength_endpoints_and_midpoint():
    orig = np.array([10.0, -20.0])
    target = np.array([0.0, 0.0])
    assert np.allclose(apply_strength(orig, target, 0.0), orig)
    assert np.allclose(apply_strength(orig, target, 1.0), target)
    assert np.allclose(apply_strength(orig, target, 0.5), (orig + target) / 2)
