# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "numpy>=1.24",
#   "scipy>=1.10",
#   "soundfile>=0.12",
#   "soxr>=0.3",
#   "pyworld>=0.3.4",
#   "setuptools<81",
# ]
# ///
"""vocal_shift.py -- offline vocal pitch/formant/time transformer and auto-tune.

Usage
-----
    uv run vocal_shift.py in.wav out.wav --pitch -2 --formant 1.04
    uv run vocal_shift.py in.wav out.wav --autotune --key D --scale minor --strength 0.8
    uv run vocal_shift.py in.wav --analyze

Build requirements
------------------
`pyworld` has no prebuilt wheel on most platforms and is compiled from its C++
sources on first `uv run`. A C++ compiler toolchain must be available:

  - macOS: Xcode Command Line Tools (`xcode-select --install`).
  - Linux: `build-essential` (Debian/Ubuntu) or an equivalent `gcc`/`g++` package.

`requires-python` is capped below 3.13 because `pyworld` (and the system
headers needed to build it) lag behind new CPython releases; capping lets
`uv` download a managed interpreter it can actually build against instead of
failing against whatever very new Python happens to be first on PATH.
`setuptools<81` is pinned because `pyworld` imports the now-removed
`pkg_resources` module at import time; setuptools 81+ dropped it.
"""

from __future__ import annotations

import argparse
import dataclasses
import re
import sys
import warnings
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# Music-theory constants
# ----------------------------------------------------------------------------

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

_KEY_ALIASES = {
    "C": 0, "B#": 0,
    "C#": 1, "DB": 1,
    "D": 2,
    "D#": 3, "EB": 3,
    "E": 4, "FB": 4,
    "F": 5, "E#": 5,
    "F#": 6, "GB": 6,
    "G": 7,
    "G#": 8, "AB": 8,
    "A": 9,
    "A#": 10, "BB": 10,
    "B": 11, "CB": 11,
}

SCALE_INTERVALS = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "chromatic": list(range(12)),
}


def parse_key(name: str) -> int:
    """Return the key's semitone offset from C (0-11)."""
    key = name.strip().upper()
    if key not in _KEY_ALIASES:
        raise ValueError(
            f"unrecognised key {name!r}; expected a note name such as C, F#, Bb"
        )
    return _KEY_ALIASES[key]


# ----------------------------------------------------------------------------
# Pitch <-> cents helpers
# ----------------------------------------------------------------------------


def ratio_from_semitones(semitones: float, cents: float = 0.0) -> float:
    return 2.0 ** ((semitones + cents / 100.0) / 12.0)


def hz_to_cents(f0: np.ndarray, reference: float = 440.0) -> np.ndarray:
    """Cents relative to `reference` (A4 by default). Unvoiced (<=0) frames -> 0."""
    f0 = np.asarray(f0, dtype=np.float64)
    voiced = f0 > 0
    cents = np.zeros_like(f0)
    cents[voiced] = 1200.0 * np.log2(f0[voiced] / reference)
    return cents


def cents_to_hz(cents: np.ndarray, reference: float = 440.0) -> np.ndarray:
    return reference * (2.0 ** (np.asarray(cents, dtype=np.float64) / 1200.0))


def hz_to_note_name(freq: float, a4: float = 440.0) -> str:
    if freq <= 0:
        return "-"
    midi = 69 + 12 * np.log2(freq / a4)
    midi_round = int(np.round(midi))
    octave = midi_round // 12 - 1
    name = NOTE_NAMES[midi_round % 12]
    return f"{name}{octave}"


_NOTE_OCTAVE_RE = re.compile(r"^([A-Ga-g])([#b]?)(-?\d+)$")


def note_name_to_hz(name: str, a4: float = 440.0) -> float:
    """Parse a note+octave string (e.g. "A2", "C#4", "Bb3") to Hz, using
    scientific pitch notation (C4 = MIDI 60)."""
    m = _NOTE_OCTAVE_RE.match(name.strip())
    if not m:
        raise ValueError(f"unrecognised note {name!r}; expected e.g. A2, C#4, Bb3")
    letter, accidental, octave = m.group(1).upper(), m.group(2), int(m.group(3))
    pitch_class = parse_key(letter + accidental)
    midi = (octave + 1) * 12 + pitch_class
    return a4 * (2.0 ** ((midi - 69) / 12.0))


def parse_pitch_or_note(value: str, a4: float = 440.0) -> float:
    """Accept either a plain Hz value ("110") or a note name ("A2")."""
    try:
        hz = float(value)
    except ValueError:
        return note_name_to_hz(value, a4)
    if hz <= 0:
        raise ValueError(f"pitch must be positive Hz, got {value!r}")
    return hz


# ----------------------------------------------------------------------------
# WAV I/O and resampling
# ----------------------------------------------------------------------------


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    """Load a WAV file as float64, shape (n_samples, n_channels)."""
    import soundfile as sf

    try:
        info = sf.info(str(path))
    except Exception as exc:
        raise ValueError(f"{path} is not a readable WAV file: {exc}") from exc
    if info.format != "WAV":
        raise ValueError(
            f"{path} is not a WAV file (detected format: {info.format}); "
            "only WAV input is supported"
        )
    data, sr = sf.read(str(path), dtype="float64", always_2d=True)
    return data, sr


def write_wav(path: Path, data: np.ndarray, sr: int, bit_depth: str = "float32") -> None:
    import soundfile as sf

    subtype = {"float32": "FLOAT", "24": "PCM_24"}[bit_depth]
    sf.write(str(path), data, sr, subtype=subtype)


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """High-quality resample of a 1-D signal. No-op if rates already match."""
    if sr_in == sr_out:
        return x
    import soxr

    return soxr.resample(x, sr_in, sr_out, quality="HQ")


def choose_working_rate(sr: int) -> int:
    """Use the input rate directly if it's already 44.1k/48k, else 44.1k."""
    return sr if sr in (44100, 48000) else 44100


# ----------------------------------------------------------------------------
# WORLD analysis / synthesis
# ----------------------------------------------------------------------------


@dataclasses.dataclass
class WorldFrames:
    f0: np.ndarray
    temporal_positions: np.ndarray
    sp: np.ndarray
    ap: np.ndarray
    fs: int
    frame_period: float
    fft_size: int


def analyse_world(
    x: np.ndarray,
    fs: int,
    frame_period: float,
    f0_floor: float,
    f0_ceil: float,
) -> WorldFrames:
    import pyworld

    x = np.ascontiguousarray(x, dtype=np.float64)
    f0, t = pyworld.harvest(
        x, fs, f0_floor=f0_floor, f0_ceil=f0_ceil, frame_period=frame_period
    )
    f0 = pyworld.stonemask(x, f0, t, fs)
    fft_size = pyworld.get_cheaptrick_fft_size(fs, f0_floor)
    sp = pyworld.cheaptrick(x, f0, t, fs, f0_floor=f0_floor, fft_size=fft_size)
    ap = pyworld.d4c(x, f0, t, fs, fft_size=fft_size)
    return WorldFrames(f0, t, sp, ap, fs, frame_period, fft_size)


def synthesise_world(frames: WorldFrames) -> np.ndarray:
    import pyworld

    return pyworld.synthesize(
        np.ascontiguousarray(frames.f0, dtype=np.float64),
        np.ascontiguousarray(frames.sp, dtype=np.float64),
        np.ascontiguousarray(frames.ap, dtype=np.float64),
        frames.fs,
        frames.frame_period,
    )


# ----------------------------------------------------------------------------
# Pitch shift
# ----------------------------------------------------------------------------


def shift_pitch(f0: np.ndarray, ratio: float) -> np.ndarray:
    """Scale F0 by `ratio` on voiced frames only; unvoiced (0) frames stay 0."""
    f0 = np.asarray(f0, dtype=np.float64)
    out = f0.copy()
    voiced = f0 > 0
    out[voiced] = f0[voiced] * ratio
    return out


# ----------------------------------------------------------------------------
# Formant shift
# ----------------------------------------------------------------------------

FORMANT_RATIO_RANGE = (0.7, 1.4)


def _normalise_envelope_energy(sp_orig: np.ndarray, sp_new: np.ndarray) -> np.ndarray:
    """Rescale each frame of `sp_new` so its total energy matches `sp_orig`
    (sp is squared-magnitude power), so reshaping the envelope doesn't by
    itself cause an audible level jump."""
    orig_energy = sp_orig.sum(axis=1)
    new_energy = sp_new.sum(axis=1)
    safe = new_energy > 1e-20
    gain = np.ones(sp_orig.shape[0])
    gain[safe] = orig_energy[safe] / new_energy[safe]
    return sp_new * gain[:, np.newaxis]


def shift_formant(
    sp: np.ndarray,
    alpha: float,
    fs: int,
    clamp: bool = True,
) -> np.ndarray:
    """Warp the spectral envelope's frequency axis by `alpha` (ratio).

    `sp_new[k] = sp[k / alpha]`, linearly interpolated, with a cosine taper
    above a cutoff near Nyquist so an upward warp cannot push envelope energy
    past Nyquist (which would alias on resynthesis). Per-frame energy is
    preserved so formant shifts do not cause audible level jumps.
    """
    lo, hi = FORMANT_RATIO_RANGE
    if clamp and not (lo <= alpha <= hi):
        warnings.warn(
            f"formant ratio {alpha:.3f} outside the recommended range "
            f"[{lo}, {hi}]; clamping. Large warps sound unnatural.",
            stacklevel=2,
        )
        alpha = float(np.clip(alpha, lo, hi))

    if alpha == 1.0:
        return sp.copy()

    n_bins = sp.shape[1]
    nyquist = fs / 2.0
    bin_hz = nyquist / (n_bins - 1)

    src_bins = np.arange(n_bins) / alpha
    src_bins = np.clip(src_bins, 0, n_bins - 1)
    bin_idx = np.arange(n_bins)

    # Vectorised linear interpolation along the frequency axis for all frames.
    lo_idx = np.floor(src_bins).astype(int)
    hi_idx = np.minimum(lo_idx + 1, n_bins - 1)
    frac = src_bins - lo_idx
    sp_new = sp[:, lo_idx] * (1 - frac) + sp[:, hi_idx] * frac

    cutoff_hz = min(nyquist * 0.9 / max(alpha, 1.0), nyquist)
    cutoff_bin = cutoff_hz / bin_hz
    # A narrow transition band (a fraction of the spectrum, capped), not the
    # whole remaining range up to Nyquist -- otherwise the raised-cosine
    # fade is so gradual it never gets close to the floor and barely
    # attenuates anything before Nyquist.
    taper_width = max(min(n_bins - cutoff_bin, 0.15 * n_bins), 4.0)
    taper = np.ones(n_bins)
    above = bin_idx > cutoff_bin
    # Raised-cosine fade from 1 down to a small floor, not a hard cutoff,
    # so tapering doesn't itself ring or click.
    fade_pos = np.clip((bin_idx - cutoff_bin) / taper_width, 0, 1)
    taper[above] = 0.5 * (1 + np.cos(np.pi * fade_pos[above]))
    taper = np.maximum(taper, 1e-6)
    sp_new = sp_new * taper[np.newaxis, :]

    return _normalise_envelope_energy(sp, sp_new)


# ----------------------------------------------------------------------------
# Vocoder (synthetic/robotic timbre)
# ----------------------------------------------------------------------------

VOCODER_SMOOTH_MAX_SIGMA_FRACTION = 0.12


def smooth_envelope(sp: np.ndarray, amount: float) -> np.ndarray:
    """Blur the spectral envelope along the frequency axis, in log domain,
    reducing formant detail towards a flat/buzzy timbre. `amount` is 0
    (unchanged) to 1 (heavily blurred). Per-frame energy is preserved.

    Blurring happens in log (dB-like) domain rather than on the raw power
    values so a few very loud bins (near a formant peak) don't dominate the
    average and leave everything else untouched -- that would barely sound
    different from the unblurred envelope.
    """
    if amount <= 0:
        return sp.copy()
    from scipy.ndimage import gaussian_filter1d

    n_bins = sp.shape[1]
    sigma = amount * VOCODER_SMOOTH_MAX_SIGMA_FRACTION * n_bins
    if sigma <= 0:
        return sp.copy()
    log_sp = np.log(np.maximum(sp, 1e-12))
    blurred = gaussian_filter1d(log_sp, sigma=sigma, axis=1, mode="nearest")
    sp_new = np.exp(blurred)
    return _normalise_envelope_energy(sp, sp_new)


def add_breathiness(ap: np.ndarray, amount: float) -> np.ndarray:
    """Blend aperiodicity towards 1.0 (fully noise-driven) by `amount`
    (0 = untouched, 1 = fully breathy/whispered)."""
    if amount <= 0:
        return ap.copy()
    return np.clip(ap + amount * (1.0 - ap), 0.0, 1.0)


def apply_vocoder(
    f0: np.ndarray,
    sp: np.ndarray,
    ap: np.ndarray,
    pitch_hz: float | None,
    smooth: float,
    breath: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classic vocoder effect built from the same F0/envelope/aperiodicity
    decomposition as the rest of the pipeline: `pitch_hz` replaces the
    natural pitch contour with a constant carrier tone (monotone/robotic),
    `smooth` flattens the formants, and `breath` adds noise."""
    f0_new = f0
    if pitch_hz is not None:
        f0_new = np.where(f0 > 0, pitch_hz, 0.0)
    sp_new = smooth_envelope(sp, smooth)
    ap_new = add_breathiness(ap, breath)
    return f0_new, sp_new, ap_new


# ----------------------------------------------------------------------------
# Time stretch
# ----------------------------------------------------------------------------


def time_stretch_frames(
    f0: np.ndarray, sp: np.ndarray, ap: np.ndarray, factor: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample the frame axis by `factor` (>1 = longer). F0 values are kept
    (not scaled), so pitch is unaffected by the duration change."""
    if factor == 1.0:
        return f0, sp, ap
    n_frames = len(f0)
    n_new = max(round(n_frames * factor), 1)
    src_pos = np.linspace(0, n_frames - 1, n_new)
    lo_idx = np.floor(src_pos).astype(int)
    hi_idx = np.minimum(lo_idx + 1, n_frames - 1)
    frac = (src_pos - lo_idx)[:, np.newaxis]

    f0_new = f0[lo_idx] * (1 - frac[:, 0]) + f0[hi_idx] * frac[:, 0]
    # Unvoiced frames should stay exactly unvoiced rather than interpolate
    # towards a neighbouring voiced frame's frequency.
    unvoiced = (f0[lo_idx] <= 0) | (f0[hi_idx] <= 0)
    f0_new[unvoiced] = 0.0

    sp_new = sp[lo_idx] * (1 - frac) + sp[hi_idx] * frac
    ap_new = ap[lo_idx] * (1 - frac) + ap[hi_idx] * frac
    return f0_new, sp_new, ap_new


# ----------------------------------------------------------------------------
# Auto-tune
# ----------------------------------------------------------------------------


def bridge_short_gaps(
    f0: np.ndarray, frame_period_ms: float, max_gap_ms: float = 30.0
) -> tuple[np.ndarray, np.ndarray]:
    """Fill unvoiced gaps shorter than `max_gap_ms` by log-domain interpolation
    between the surrounding voiced frames, so brief dropouts don't split a
    note. Returns (f0_bridged, voiced_mask)."""
    f0 = np.asarray(f0, dtype=np.float64).copy()
    n = len(f0)
    voiced = f0 > 0
    max_gap_frames = max(round(max_gap_ms / frame_period_ms), 0)

    i = 0
    while i < n:
        if voiced[i]:
            i += 1
            continue
        j = i
        while j < n and not voiced[j]:
            j += 1
        gap_len = j - i
        if 0 < i and j < n and gap_len <= max_gap_frames:
            left, right = f0[i - 1], f0[j]
            t = (np.arange(i, j) - (i - 1)) / (j - (i - 1))
            f0[i:j] = np.exp(np.log(left) + t * (np.log(right) - np.log(left)))
            voiced[i:j] = True
        i = j
    return f0, voiced


def median_smooth_f0(f0: np.ndarray, voiced: np.ndarray, kernel: int = 5) -> np.ndarray:
    """Median-filter F0 within each contiguous voiced run (runs are not
    smeared into neighbouring silence)."""
    from scipy.signal import medfilt

    out = f0.copy()
    n = len(f0)
    i = 0
    while i < n:
        if not voiced[i]:
            i += 1
            continue
        j = i
        while j < n and voiced[j]:
            j += 1
        run = f0[i:j]
        if len(run) >= 3:
            k = min(kernel, len(run) if len(run) % 2 == 1 else len(run) - 1)
            k = max(k, 1)
            out[i:j] = medfilt(run, kernel_size=k)
        i = j
    return out


@dataclasses.dataclass
class Note:
    start: int
    end: int  # exclusive


def segment_notes(
    cents: np.ndarray,
    voiced: np.ndarray,
    frame_period_ms: float,
    split_threshold_cents: float = 50.0,
    min_note_ms: float = 80.0,
    debounce_ms: float = 20.0,
) -> list[Note]:
    """Group contiguous voiced frames into notes.

    A candidate split only takes effect once the new pitch has persisted for
    `debounce_ms` (so a single-frame transient doesn't start a new note --
    this is the "hysteresis" step). Notes shorter than `min_note_ms` are
    merged into a neighbour, so passing tones and consonants don't become
    their own (mis-tuned) targets.
    """
    n = len(cents)
    debounce_frames = max(round(debounce_ms / frame_period_ms), 1)
    min_note_frames = max(round(min_note_ms / frame_period_ms), 1)

    final: list[Note] = []
    i = 0
    while i < n:
        if not voiced[i]:
            i += 1
            continue
        run_start = i
        j = i
        while j < n and voiced[j]:
            j += 1
        run_end = j

        run_notes: list[Note] = []
        note_start = run_start
        ref = cents[run_start]
        k = run_start + 1
        while k < run_end:
            if abs(cents[k] - ref) > split_threshold_cents:
                # Candidate boundary: require it to persist for debounce_frames.
                persists = k + debounce_frames <= run_end and np.all(
                    np.abs(cents[k : k + debounce_frames] - cents[k]) <= split_threshold_cents
                )
                if persists:
                    run_notes.append(Note(note_start, k))
                    note_start = k
                    ref = cents[k]
                else:
                    k += 1
                    continue
            else:
                ref = cents[k]
            k += 1
        run_notes.append(Note(note_start, run_end))

        # Merge notes shorter than min_note_ms into a neighbour, within this
        # voiced run only (never across a silence gap into another run).
        # Short notes merge backward where possible; a short note that opens
        # the run (no predecessor yet) merges forward instead.
        merged: list[Note] = []
        for note in run_notes:
            if (note.end - note.start) < min_note_frames and merged:
                merged[-1] = Note(merged[-1].start, note.end)
            else:
                merged.append(note)
        while len(merged) > 1 and (merged[0].end - merged[0].start) < min_note_frames:
            merged[1] = Note(merged[0].start, merged[1].end)
            merged.pop(0)

        final.extend(merged)
        i = run_end

    return final


def snap_chromatic(cents: float, a4: float = 440.0) -> float:
    """Nearest equal-tempered semitone (multiples of 100 cents from A4).
    Exact ties (precisely 50 cents from both neighbours) resolve to the
    lower pitch, for determinism."""
    semitones = cents / 100.0
    floor_s = np.floor(semitones)
    dist_floor = semitones - floor_s
    dist_ceil = (floor_s + 1) - semitones
    target_s = floor_s if dist_floor <= dist_ceil else floor_s + 1
    return target_s * 100.0


def snap_scale(cents: float, key: int, scale: str) -> float:
    """Nearest note (any octave) whose pitch class is in `key`'s `scale`.
    Exact ties resolve to the lower pitch, for determinism."""
    allowed = {(key + iv) % 12 for iv in SCALE_INTERVALS[scale]}
    midi_real = 69 + cents / 100.0
    center = round(midi_real)
    best = None
    best_dist = None
    for candidate in range(center - 13, center + 14):
        if candidate % 12 not in allowed:
            continue
        dist = abs(candidate - midi_real)
        if best is None or dist < best_dist or (dist == best_dist and candidate < best):
            best = candidate
            best_dist = dist
    target_cents = (best - 69) * 100.0
    return target_cents


def snap_to_targets(
    cents: float, key: int | None = None, scale: str | None = None
) -> float:
    if key is None or scale is None:
        return snap_chromatic(cents)
    return snap_scale(cents, key, scale)


def apply_strength(orig_cents: np.ndarray, target_cents: np.ndarray, strength: float) -> np.ndarray:
    return orig_cents + strength * (target_cents - orig_cents)


def glide_filter(target: np.ndarray, frame_period_ms: float, tau_ms: float) -> np.ndarray:
    """One-pole low-pass filter towards `target`, time constant `tau_ms`.
    `tau_ms <= 0` means an instant snap (no smoothing)."""
    out = np.asarray(target, dtype=np.float64).copy()
    if tau_ms <= 0 or len(out) == 0:
        return out
    coeff = float(np.exp(-frame_period_ms / tau_ms))
    for i in range(1, len(out)):
        out[i] = out[i - 1] + (1 - coeff) * (target[i] - out[i - 1])
    return out


@dataclasses.dataclass
class AutotuneStats:
    n_notes: int
    mean_abs_correction_cents: float
    max_correction_cents: float
    scale_label: str


def autotune(
    f0: np.ndarray,
    frame_period_ms: float,
    key: int | None,
    scale: str | None,
    strength: float,
    retune_speed_ms: float,
    vibrato_keep: float,
    min_note_ms: float,
    a4: float,
    split_threshold_cents: float = 50.0,
    gap_bridge_ms: float = 30.0,
) -> tuple[np.ndarray, AutotuneStats]:
    f0_bridged, voiced = bridge_short_gaps(f0, frame_period_ms, gap_bridge_ms)
    f0_smooth = median_smooth_f0(f0_bridged, voiced, kernel=5)
    cents = hz_to_cents(f0_smooth, a4)

    notes = segment_notes(
        cents, voiced, frame_period_ms,
        split_threshold_cents=split_threshold_cents,
        min_note_ms=min_note_ms,
    )

    note_median = np.zeros_like(cents)
    note_target = np.zeros_like(cents)
    corrections = []
    for note in notes:
        seg = cents[note.start : note.end]
        median_c = float(np.median(seg))
        target_c = snap_to_targets(median_c, key, scale)
        note_median[note.start : note.end] = median_c
        note_target[note.start : note.end] = target_c
        corrections.append(target_c - median_c)

    strength_target = apply_strength(note_median, note_target, strength)
    glide = glide_filter(strength_target, frame_period_ms, retune_speed_ms)

    residual = cents - note_median
    final_cents = glide + vibrato_keep * residual

    # Crossfade at note onset/offset so the correction fades in/out rather
    # than switching abruptly at a voiced-run boundary.
    fade_frames = max(round(min(20.0, min_note_ms / 2) / frame_period_ms), 1)
    out_cents = cents.copy()
    for note in notes:
        length = note.end - note.start
        fade = min(fade_frames, length // 2) if length > 1 else 0
        seg_final = final_cents[note.start : note.end]
        seg_orig = cents[note.start : note.end]
        blended = seg_final.copy()
        if fade > 0:
            ramp_in = np.linspace(0, 1, fade)
            ramp_out = np.linspace(1, 0, fade)
            blended[:fade] = seg_orig[:fade] * (1 - ramp_in) + seg_final[:fade] * ramp_in
            blended[-fade:] = seg_orig[-fade:] * (1 - ramp_out) + seg_final[-fade:] * ramp_out
        out_cents[note.start : note.end] = blended

    f0_new = np.zeros_like(f0_bridged)
    f0_new[voiced] = cents_to_hz(out_cents[voiced], a4)

    if corrections:
        abs_corr = np.abs(corrections)
        stats = AutotuneStats(
            n_notes=len(notes),
            mean_abs_correction_cents=float(np.mean(abs_corr)),
            max_correction_cents=float(np.max(abs_corr)),
            scale_label=f"{scale}" if scale else "chromatic",
        )
    else:
        stats = AutotuneStats(0, 0.0, 0.0, f"{scale}" if scale else "chromatic")
    return f0_new, stats


# ----------------------------------------------------------------------------
# Post-synthesis processing
# ----------------------------------------------------------------------------


def lowpass_before_downsample(y: np.ndarray, fs_working: int, fs_out: int) -> np.ndarray:
    """Explicit anti-aliasing low-pass before resampling down. soxr's HQ
    resampler already filters internally; this is a defence-in-depth safety
    net per the design's anti-aliasing requirement, and is a no-op unless
    actually downsampling."""
    if fs_out >= fs_working:
        return y
    from scipy.signal import butter, sosfiltfilt

    cutoff = min(0.45 * fs_out, fs_out / 2.0 * 0.98)
    sos = butter(8, cutoff, btype="low", fs=fs_working, output="sos")
    return sosfiltfilt(sos, y)


def peak_normalise_if_clipping(y: np.ndarray) -> tuple[np.ndarray, int]:
    """Only scales the signal down if it would clip; never raises level.
    Returns (signal, n_clipped_samples_before_correction)."""
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    n_clipped = int(np.sum(np.abs(y) > 1.0))
    if peak > 1.0:
        y = y / peak
    return y, n_clipped


# ----------------------------------------------------------------------------
# Analysis report
# ----------------------------------------------------------------------------


@dataclasses.dataclass
class AnalysisReport:
    duration_s: float
    sample_rate: int
    channels: int
    f0_min_hz: float
    f0_max_hz: float
    voicing_ratio: float


def analyse_report(data: np.ndarray, sr: int, frames_per_channel: list[WorldFrames]) -> AnalysisReport:
    all_f0 = np.concatenate([f.f0 for f in frames_per_channel])
    voiced = all_f0 > 0
    f0_voiced = all_f0[voiced]
    return AnalysisReport(
        duration_s=data.shape[0] / sr,
        sample_rate=sr,
        channels=data.shape[1],
        f0_min_hz=float(f0_voiced.min()) if f0_voiced.size else 0.0,
        f0_max_hz=float(f0_voiced.max()) if f0_voiced.size else 0.0,
        voicing_ratio=float(voiced.mean()) if voiced.size else 0.0,
    )


def print_report(
    report: AnalysisReport,
    args: argparse.Namespace | None = None,
    autotune_stats: AutotuneStats | None = None,
    n_clipped: int | None = None,
    a4: float = 440.0,
) -> None:
    print(f"Duration:        {report.duration_s:.2f} s")
    print(f"Sample rate:     {report.sample_rate} Hz")
    print(f"Channels:        {report.channels}")
    if report.voicing_ratio > 0:
        print(
            f"F0 range:        {report.f0_min_hz:.1f}-{report.f0_max_hz:.1f} Hz "
            f"({hz_to_note_name(report.f0_min_hz, a4)} - {hz_to_note_name(report.f0_max_hz, a4)})"
        )
    else:
        print("F0 range:        no voiced frames detected")
    print(f"Voicing ratio:   {report.voicing_ratio * 100:.1f}%")

    if args is not None:
        print(f"Pitch shift:     {args.pitch:+g} semitones, {args.cents:+g} cents")
        formant_desc = (
            f"{args.formant:.4f} ratio"
            if args.formant_semitones is None
            else f"{args.formant_semitones:+g} semitones"
        )
        print(f"Formant shift:   {formant_desc}")
        print(f"Time stretch:    {args.time_stretch:.3f}x")
        if args.vocoder:
            pitch_desc = (
                f"{args.vocoder_pitch_hz:.2f} Hz carrier"
                if args.vocoder_pitch_hz is not None
                else "natural pitch contour"
            )
            print(
                f"Vocoder:         {pitch_desc}, "
                f"smooth {args.vocoder_smooth:.2f}, breath {args.vocoder_breath:.2f}"
            )

    if autotune_stats is not None:
        print(
            f"Auto-tune:       {autotune_stats.n_notes} notes, "
            f"mean |correction| {autotune_stats.mean_abs_correction_cents:.1f} cents, "
            f"max {autotune_stats.max_correction_cents:.1f} cents, "
            f"scale: {autotune_stats.scale_label}"
        )

    if n_clipped is not None:
        print(f"Clipping:        {n_clipped} sample(s) clipped before output")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vocal_shift.py",
        description="Offline pitch/formant/time transformer and auto-tune for sung vocals (WAV only).",
    )
    p.add_argument("input", type=Path, help="Input WAV file (or directory with --batch)")
    p.add_argument("output", type=Path, nargs="?", help="Output WAV file (or directory with --batch)")

    p.add_argument("--pitch", type=float, default=0.0, help="Pitch shift in semitones (default: 0)")
    p.add_argument("--cents", type=float, default=0.0, help="Additional fine pitch shift in cents (default: 0)")
    p.add_argument("--formant", type=float, default=1.0, help="Formant ratio, 1.0 = unchanged (default: 1.0)")
    p.add_argument("--formant-semitones", type=float, default=None, help="Formant shift in semitones (mutually exclusive with --formant)")
    p.add_argument("--time-stretch", type=float, default=1.0, help="Duration factor (default: 1.0)")

    p.add_argument("--autotune", action="store_true", help="Enable intonation correction")
    p.add_argument("--key", type=str, default=None, help="Key for scale-constrained auto-tune (requires --scale)")
    p.add_argument("--scale", type=str, default=None, choices=sorted(SCALE_INTERVALS), help="Scale for auto-tune (requires --key)")
    p.add_argument("--strength", type=float, default=1.0, help="Auto-tune correction amount, 0-1 (default: 1.0)")
    p.add_argument("--retune-speed", type=float, default=20.0, help="Auto-tune glide time constant in ms (default: 20)")
    p.add_argument("--vibrato-keep", type=float, default=0.0, help="Fraction of original vibrato kept, 0-1 (default: 0)")
    p.add_argument("--min-note-ms", type=float, default=80.0, help="Minimum note duration for segmentation (default: 80)")

    p.add_argument("--vocoder", action="store_true", help="Enable synthetic/robotic vocoder effect")
    p.add_argument("--vocoder-pitch", type=str, default=None, metavar="HZ|NOTE", help="Force a constant carrier pitch, e.g. 110 or A2 (requires --vocoder; omit to keep the natural pitch contour)")
    p.add_argument("--vocoder-smooth", type=float, default=0.0, help="Spectral envelope smoothing, 0-1 (default: 0, requires --vocoder)")
    p.add_argument("--vocoder-breath", type=float, default=0.0, help="Blend toward fully breathy/noisy, 0-1 (default: 0, requires --vocoder)")

    p.add_argument("--a4", type=float, default=440.0, help="Reference tuning in Hz (default: 440)")
    p.add_argument("--f0-min", type=float, default=60.0, help="Lowest expected F0 in Hz (default: 60)")
    p.add_argument("--f0-max", type=float, default=1100.0, help="Highest expected F0 in Hz (default: 1100)")
    p.add_argument("--quality", type=str, default="high", choices=["fast", "high"], help="Analysis quality (default: high)")
    p.add_argument("--bit-depth", type=str, default="float32", choices=["float32", "24"], help="Output bit depth (default: float32)")

    p.add_argument("--analyze", action="store_true", help="Print analysis report only, write no output file")
    p.add_argument("--preview", type=float, default=None, metavar="SECONDS", help="Render only the first N seconds")
    p.add_argument("--batch", action="store_true", help="Treat INPUT as a directory of WAVs, OUTPUT as an output directory")

    return p


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if (args.key is None) != (args.scale is None):
        parser.error("--key and --scale must be given together")
    if args.formant != 1.0 and args.formant_semitones is not None:
        parser.error("--formant and --formant-semitones are mutually exclusive")
    if not args.autotune:
        autotune_only = {
            "--key": args.key,
            "--strength": None if args.strength == 1.0 else args.strength,
            "--retune-speed": None if args.retune_speed == 20.0 else args.retune_speed,
            "--vibrato-keep": None if args.vibrato_keep == 0.0 else args.vibrato_keep,
            "--min-note-ms": None if args.min_note_ms == 80.0 else args.min_note_ms,
        }
        used = [name for name, val in autotune_only.items() if val not in (None, False)]
        if used:
            parser.error(
                f"the following options require --autotune to be set: {', '.join(used)}"
            )
    if not args.vocoder:
        vocoder_only = {
            "--vocoder-pitch": args.vocoder_pitch,
            "--vocoder-smooth": None if args.vocoder_smooth == 0.0 else args.vocoder_smooth,
            "--vocoder-breath": None if args.vocoder_breath == 0.0 else args.vocoder_breath,
        }
        used = [name for name, val in vocoder_only.items() if val not in (None, False)]
        if used:
            parser.error(
                f"the following options require --vocoder to be set: {', '.join(used)}"
            )
    if not 0.0 <= args.vocoder_smooth <= 1.0:
        parser.error("--vocoder-smooth must be between 0 and 1")
    if not 0.0 <= args.vocoder_breath <= 1.0:
        parser.error("--vocoder-breath must be between 0 and 1")
    args.vocoder_pitch_hz = None
    if args.vocoder_pitch is not None:
        try:
            args.vocoder_pitch_hz = parse_pitch_or_note(args.vocoder_pitch, args.a4)
        except ValueError as exc:
            parser.error(str(exc))
    if args.f0_min >= args.f0_max:
        parser.error("--f0-min must be less than --f0-max")
    if not args.analyze and not args.batch and args.output is None:
        parser.error("OUTPUT is required unless --analyze is set")
    if not args.input.exists():
        parser.error(f"input not found: {args.input}")


# ----------------------------------------------------------------------------
# Core pipeline (single file)
# ----------------------------------------------------------------------------


def process_file(input_path: Path, output_path: Path | None, args: argparse.Namespace) -> None:
    data, sr_in = load_wav(input_path)
    if args.preview is not None:
        n = int(args.preview * sr_in)
        data = data[:n]

    fs_working = choose_working_rate(sr_in)
    frame_period = 5.0 if args.quality == "high" else 10.0

    n_channels = data.shape[1]
    channel_frames: list[WorldFrames] = []
    channel_out: list[np.ndarray] = []
    autotune_stats: AutotuneStats | None = None

    pitch_ratio = ratio_from_semitones(args.pitch, args.cents)
    if args.formant_semitones is not None:
        formant_ratio = ratio_from_semitones(args.formant_semitones)
    else:
        formant_ratio = args.formant

    key_offset = parse_key(args.key) if args.key else None

    for ch in range(n_channels):
        x = data[:, ch]
        x_work = resample(x, sr_in, fs_working)
        frames = analyse_world(x_work, fs_working, frame_period, args.f0_min, args.f0_max)
        channel_frames.append(frames)

        if args.analyze:
            continue

        f0 = frames.f0
        if args.autotune:
            f0, autotune_stats = autotune(
                f0,
                frame_period,
                key_offset,
                args.scale,
                args.strength,
                args.retune_speed,
                args.vibrato_keep,
                args.min_note_ms,
                args.a4,
            )

        f0 = shift_pitch(f0, pitch_ratio)
        sp = shift_formant(frames.sp, formant_ratio, fs_working) if formant_ratio != 1.0 else frames.sp
        ap = frames.ap

        if args.vocoder:
            f0, sp, ap = apply_vocoder(
                f0, sp, ap, args.vocoder_pitch_hz, args.vocoder_smooth, args.vocoder_breath
            )

        if args.time_stretch != 1.0:
            f0, sp, ap = time_stretch_frames(f0, sp, ap, args.time_stretch)

        out_frames = WorldFrames(f0, frames.temporal_positions, sp, ap, fs_working, frame_period, frames.fft_size)
        y = synthesise_world(out_frames)
        y = lowpass_before_downsample(y, fs_working, sr_in)
        y = resample(y, fs_working, sr_in)
        channel_out.append(y)

    report = analyse_report(data, sr_in, channel_frames)

    if args.analyze:
        print(f"--- Analysis: {input_path} ---")
        print_report(report, args=None, a4=args.a4)
        return

    max_len = max(len(y) for y in channel_out)
    out = np.zeros((max_len, n_channels), dtype=np.float64)
    for ch, y in enumerate(channel_out):
        out[: len(y), ch] = y

    out, n_clipped = peak_normalise_if_clipping(out)
    write_wav(output_path, out, sr_in, args.bit_depth)

    print(f"--- {input_path} -> {output_path} ---")
    print_report(report, args=args, autotune_stats=autotune_stats, n_clipped=n_clipped, a4=args.a4)


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    validate_args(args, parser)

    if args.batch:
        if not args.input.is_dir():
            parser.error("--batch requires INPUT to be a directory")
        if args.output is None and not args.analyze:
            parser.error("--batch requires OUTPUT to be a directory")
        if args.output is not None:
            args.output.mkdir(parents=True, exist_ok=True)
        wavs = sorted(args.input.glob("*.wav"))
        if not wavs:
            print(f"no .wav files found in {args.input}", file=sys.stderr)
            return
        for wav in wavs:
            out_path = None if args.analyze else args.output / wav.name
            process_file(wav, out_path, args)
    else:
        process_file(args.input, args.output, args)


if __name__ == "__main__":
    main()
