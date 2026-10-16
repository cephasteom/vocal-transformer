# vocal_shift.py

An offline, command-line vocal transformer for WAV files, focused on singing.
It changes pitch, formants and timing independently, and can correct
imprecise intonation (auto-tune). It ships as a single self-contained script
run with [uv](https://docs.astral.sh/uv/).

Audio quality (no audible artifacts, no aliasing) is the first priority,
then correctness, then speed. The engine is the
[WORLD vocoder](https://github.com/mmorise/World) (via `pyworld`), which
decomposes a voice into F0 (pitch), spectral envelope (formants/timbre) and
aperiodicity (breathiness) so each can be changed independently, instead of
the "chipmunk" pitch/formant coupling you get from naive resampling.

## Requirements

- Python, managed automatically by `uv` (the script pins `>=3.10,<3.13`).
- A C/C++ compiler toolchain, because `pyworld` has no prebuilt wheel on
  most platforms and is compiled from source on first run:
  - **macOS**: `xcode-select --install`
  - **Linux**: `build-essential` (Debian/Ubuntu) or your distro's `gcc`/`g++`
    package.

The Python version is capped below 3.13 and `setuptools<81` is pinned in
the script's dependencies. Both work around `pyworld` lagging behind very
recent CPython/setuptools releases (see the script's module docstring for
details) -- `uv run` handles this automatically, you shouldn't need to do
anything.

## Usage

```bash
uv run vocal_shift.py in.wav out.wav --pitch -2 --formant 1.04
uv run vocal_shift.py in.wav out.wav --autotune --key D --scale minor --strength 0.8 --retune-speed 40
uv run vocal_shift.py in.wav --analyze
```

### Common examples

```bash
# Pitch down 2 semitones, no formant change (natural-sounding pitch shift)
uv run vocal_shift.py verse.wav verse_down2.wav --pitch -2

# Raise formants (brighter/smaller-sounding voice) without changing pitch
uv run vocal_shift.py verse.wav verse_bright.wav --formant 1.08

# Chromatic auto-tune, moderate strength, natural-sounding glide
uv run vocal_shift.py verse.wav verse_tuned.wav --autotune --strength 0.6 --retune-speed 35

# Hard-tuned to a specific key/scale, classic "T-Pain" effect
uv run vocal_shift.py verse.wav verse_hard.wav --autotune --key C --scale major --strength 1.0 --retune-speed 10

# Keep some natural vibrato while still correcting the underlying pitch
uv run vocal_shift.py verse.wav verse_tuned.wav --autotune --key A --scale minor --vibrato-keep 0.4

# Slow a phrase down 15% without changing its pitch
uv run vocal_shift.py verse.wav verse_slow.wav --time-stretch 1.15

# Just inspect a file: F0 range, voicing, note names -- no file written
uv run vocal_shift.py verse.wav --analyze

# Quick 10-second A/B render while dialing in settings
uv run vocal_shift.py verse.wav preview.wav --autotune --key D --scale minor --preview 10

# Process every WAV in a folder the same way
uv run vocal_shift.py takes/ takes_tuned/ --batch --autotune --key E --scale major

# Classic robot/vocoder voice: flat carrier pitch, flattened formants, breathy noise
uv run vocal_shift.py verse.wav verse_robot.wav --vocoder --vocoder-pitch A2 --vocoder-smooth 0.7 --vocoder-breath 0.3

# Just flatten the formants a bit, keep the natural pitch contour
uv run vocal_shift.py verse.wav verse_buzzy.wav --vocoder --vocoder-smooth 0.4
```

Run `uv run vocal_shift.py --help` for the full option list.

## How the options fit together

- `--pitch` / `--cents`: transpose the whole performance (semitones / fine
  cents), independent of formants.
- `--formant` (ratio) or `--formant-semitones`: change the voice's
  perceived size/brightness without changing pitch. Kept inside
  `0.7`-`1.4` by default (outside that range sounds unnatural; the script
  warns and clamps).
- `--time-stretch`: change duration without affecting pitch.
- `--autotune` plus `--key`/`--scale`: correct intonation. Leave `--key`
  and `--scale` unset for chromatic (nearest-semitone) correction; set both
  for scale-constrained correction. `--strength` blends original vs.
  corrected pitch (0 = untouched, 1 = fully corrected); `--retune-speed`
  controls how fast the pitch glides onto the corrected note (lower =
  snappier/more obviously "autotuned", higher = smoother); `--vibrato-keep`
  reintroduces a fraction of the singer's original vibrato on top of the
  correction so sustained notes don't sound flat.
- `--vocoder` plus `--vocoder-pitch`/`--vocoder-smooth`/`--vocoder-breath`:
  a synthetic/robotic timbre effect, built from the same F0/envelope/
  aperiodicity decomposition as everything else (see `apply_vocoder` in the
  script). `--vocoder-pitch` replaces the natural pitch contour with a
  constant carrier tone -- accepts either Hz (`110`) or a note name
  (`A2`) -- and is optional (omit it to keep the natural pitch and just
  apply smoothing/breath). `--vocoder-smooth` (0-1) blurs the formants
  towards a flat, buzzy timbre. `--vocoder-breath` (0-1) blends the
  aperiodicity towards fully noise-driven (breathy/whispered). All three
  combine with pitch/formant shift and auto-tune if you want layered
  effects.
- `--analyze`: just print F0 range, voicing ratio etc. and exit -- useful
  for sanity-checking a take, or seeing how much correction a later
  `--autotune` run would need to do.
- `--preview SECONDS`: render only the first N seconds, for fast A/B
  iteration on long files.
- `--batch`: treat `INPUT`/`OUTPUT` as directories and process every `.wav`
  in `INPUT`.

## Report output

Every run (except `--analyze`, which prints the same analysis block and
nothing else) prints a short summary: input duration/rate/channels,
detected F0 range and voicing ratio, the transform(s) applied, auto-tune
statistics (notes detected, mean/max correction in cents), and how many
samples would have clipped before output (the signal is peak-normalised
only when that number is non-zero -- gain is never changed unconditionally,
and clipping is always reported, never silently fixed by lowering level
you didn't ask for).

## Scope

In scope: WAV in/out (mono or stereo, processed per channel), pitch shift,
formant shift, time stretch, chromatic and scale-constrained auto-tune,
analysis, batch processing, short previews.

Out of scope: other audio formats (MP3, FLAC, ...), real-time/streaming use,
a GUI, separating multiple simultaneous singers, and non-vocal material
(instruments may work but are not the target and aren't tuned for).

## Testing

```bash
uv run --group dev pytest
```

Unit tests cover the pitch/formant/auto-tune math directly (no audio I/O).
Integration tests synthesize tones in code, run them through the real
WORLD analysis/synthesis pipeline, and check the result with an
independent (test-only) autocorrelation pitch estimator -- there is no
`librosa` or other heavyweight dependency involved.

One thing the test suite works around, worth knowing if you write more
synthetic test audio: WORLD's Harvest F0 tracker is tuned for natural
voices and can fail almost completely on a perfectly static, jitter-free
sine (it's tuned to flag such a signal as unvoiced). The test tone
generator (`tests/conftest.py::make_tone`) adds a small (1%) vibrato by
default, which is realistic for actual singing anyway and is enough for
Harvest to track reliably.

### Listening checks (manual)

Automated tests check the math, not how a result actually sounds. Before
trusting a new setting (especially auto-tune `--strength`/`--retune-speed`
combinations) on real material, run it on a short sung clip and listen:

- Compare `--autotune --strength 1.0 --retune-speed 10` (hard/obvious) against
  `--strength 0.5-0.7 --retune-speed 30-50` (gentler, less noticeable) on the
  same take.
- Expressive, highly ornamented singing (slides, grace notes) is more prone
  to sounding over-corrected with chromatic auto-tune at high strength;
  scale-constrained auto-tune at a lower strength tends to sound more
  natural on that kind of material.
- Large formant shifts (near the edges of the `0.7`-`1.4` range) are more
  likely to sound unnatural than large pitch shifts -- if a result sounds
  off, try a smaller `--formant` value before suspecting a bug.
- Watch the "Clipping" line in the report. WORLD's resynthesis is
  generative, not a literal waveform reconstruction, so its output level
  can differ from the input's even with no transform applied; occasional
  clipping on an unusually hot input is expected and handled, not a sign
  that something is broken.
