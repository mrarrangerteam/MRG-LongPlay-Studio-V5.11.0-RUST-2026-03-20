# LongPlay Studio V5 — CLAUDE.md

## Key Files
- **gui.py** — Standalone monolith (LongPlayStudioV4, 13k LOC). Run directly: `python gui.py`
- **gui/main.py** — Refactored package version (7k LOC). Used by: `from gui import LongPlayStudioV4`
- **gui/__init__.py** — Re-exports from gui/main.py and gui/ subpackages
- **modules/master/ui_panel.py** — Mastering panel (MasterPanel, MetersPanel, all widgets, 6k+ LOC)
- **modules/master/chain.py** — Audio processing chain (MasterChain, real DSP pipeline)
- **modules/master/maximizer.py** — IRC Maximizer (gain push, ceiling, IRC modes 1-5)
- **modules/master/equalizer.py** — 8-band parametric EQ
- **modules/master/dynamics.py** — Compressor (single + multiband)
- **modules/master/imager.py** — Stereo width (multiband)
- **modules/master/loudness.py** — LUFS/True Peak measurement (ITU-R BS.1770-4)
- **modules/master/limiter.py** — Look-ahead True Peak brickwall limiter
- **modules/master/ai_assist.py** — AI recommendation engine
- **modules/master/genre_profiles.py** — Genre presets, platform targets, IRC modes, tone presets
- **gui/widgets/** — Reusable QPainter widgets (rotary knob, vectorscope, transfer curve, etc.)
- **rust/crates/longplay-rt/** — Real-time audio engine (cpal + lock-free DSP)
- **rust/crates/longplay-rt/src/params.rs** — AtomicF32 parameter store (Python → audio thread)
- **rust/crates/longplay-rt/src/stream.rs** — cpal audio callback + DSP chain per-block
- **rust/crates/longplay-python/src/rt_engine.rs** — PyRtEngine PyO3 bindings

## Dual Entry Points (IMPORTANT)
- `python gui.py` — runs the standalone monolith (used by launch.sh)
- `from gui import LongPlayStudioV4` — imports the refactored version (gui/main.py via gui/__init__.py)
- **Both define LongPlayStudioV4 but they are separate implementations**
- gui/main.py has Phase 2/4 features (undo, themes, text clips, speed ramp, pipeline)
- gui.py has all helper classes inline; gui/main.py imports from gui/ subpackages
- **When fixing bugs, apply to BOTH files if the code exists in both**

## NEVER Do
- **NEVER delete gui.py or gui/main.py** — both are needed (standalone vs package mode)
- **NEVER delete or replace existing working code** — extend it
- **NEVER use fake/random data for meters** — always use real audio samples from chain callbacks

## Development Loop (RALP)
1. **R**ead — Read existing code before modifying
2. **A**dd — Add new code, extend existing
3. **L**int — Test with `python3 -c 'from gui import LongPlayStudioV4; print("OK")'`
4. **P**ush — Commit with `feat(STORY-ID): description`

## Architecture
- **Hybrid** (V5.10+): Offline rendering + Real-time preview via `longplay-rt` (Rust + cpal)
- **Offline**: chain.py renders full audio for export (Export-Parity)
- **Real-time**: `longplay-rt` plays audio through DSP with lock-free atomic parameters (knob → instant)
- **Playback**: PyRtEngine (cpal) for mastering preview; QMediaPlayer for timeline playback
- **Meter data**: RT engine → crossbeam channel → Python poll 30Hz; chain._send_meter() for offline
- **Signal flow**: Input → EQ → Dynamics → Imager → Maximizer → Loudness Norm → True Peak Limit → Output
- **RT signal flow**: File → Memory → [EQ → Imager → Maximizer → Limiter] → cpal output (512 samples/block)

## Tech Stack
- Python 3.14, PyQt6, NumPy, SciPy, pedalboard, pyloudnorm, soundfile
- Rust backend (longplay-dsp, longplay-chain, etc.) compiled to .so via PyO3
- QPainter for all custom widgets (meters, knobs, curves, vectorscope)
