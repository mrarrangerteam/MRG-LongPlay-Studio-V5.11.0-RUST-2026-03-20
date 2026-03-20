# PRD Phase 5 — Ozone 12 Clone + WLM Plus Clone

**Goal**: Transform LongPlay Studio into a professional mastering suite that visually and functionally matches iZotope Ozone 12 and Waves WLM Plus.

**Version**: v5.6.0
**Stories**: 24 total (B-4 already complete)

---

## EPIC A — Ozone Maximizer (6 stories)

### A-1: IRC Sub-Modes Dropdown
**Status**: TODO
**Priority**: High

Add IRC sub-mode selector to the Maximizer panel. When user selects IRC III, show a dropdown with sub-modes: Pumping, Balanced, Crisp, Clipping. When IRC IV is selected, show: Classic, Modern, Transient. IRC I, II, V have no sub-modes (hide dropdown).

**Acceptance Criteria**:
- QComboBox appears/hides dynamically based on IRC mode selection
- Sub-mode selection is passed to `Maximizer.set_irc_sub_mode()`
- Chain processes audio differently based on sub-mode
- Persisted in save/load settings

### A-2: OzoneRotaryKnob QPainter Widget
**Status**: TODO
**Priority**: High (blocks A-3, A-4, A-5, A-6)

Create `gui/widgets/rotary_knob.py` with `OzoneRotaryKnob(QWidget)`:
- Dark circle background (#1A1A1E)
- 270° value arc (gap at bottom), teal color
- Center value text + unit label below
- Parameter name above
- Mouse drag (vertical) to adjust, scroll wheel for fine, double-click to reset default
- Shift+drag for fine mode (0.1x)
- Signals: `valueChanged(float)`
- Constructor: `OzoneRotaryKnob(name, min_val, max_val, default, unit, decimals)`

**File**: `gui/widgets/rotary_knob.py`

### A-3: Maximizer Panel Rebuild
**Status**: TODO
**Priority**: High

Rebuild `_build_maximizer_view()` in ui_panel.py to show ALL Ozone 12 Maximizer controls using OzoneRotaryKnob widgets:
- **Gain** knob (0–20 dB, large size)
- **Ceiling** knob (-3.0 to 0.0 dBTP)
- **Character** knob (0–10)
- **Upward Compress** knob (0–12 dB)
- **Soft Clip** knob (0–100%)
- **Transient Emphasis** selector (High/Medium/Low)
- **Stereo Independence** checkbox
- **IRC Mode** selector (I–V) with sub-mode dropdown (from A-1)
- **Tone Preset** selector
- GR History widget + LUFS readout (existing, re-integrated)

### A-4: Dynamics Panel Rebuild
**Status**: TODO
**Priority**: Medium

Rebuild `_build_dynamics_view()` in ui_panel.py:
- Multiband crossover frequency display (graph showing low/mid/high split)
- Per-band controls: Threshold, Ratio, Attack, Release, Knee (using OzoneRotaryKnob)
- Transfer Curve widget (from B-2) showing compression curve
- **Parallel Mix** knob (0–100%)
- **Detection** mode selector: Peak / Envelope / RMS
- Band solo/mute buttons
- Dynamics preset selector (existing)

### A-5: Imager Panel Rebuild
**Status**: TODO
**Priority**: Medium

Rebuild `_build_imager_view()` in ui_panel.py:
- 4-band width sliders (Low, Low-Mid, High-Mid, High) with range 0–200%
- **Stereoize I** and **Stereoize II** toggle buttons
- Vectorscope widget (from B-1) showing live stereo field
- **Correlation Meter** (-1 to +1 horizontal bar)
- Mono Bass frequency knob
- Balance knob (-100L to +100R)
- Imager preset selector (existing)

### A-6: EQ Panel Rebuild
**Status**: TODO
**Priority**: Medium

Rebuild `_build_eq_view()` in ui_panel.py:
- Interactive EQ curve (QPainter): draggable band points on frequency response
- 8 bands with: Frequency, Gain, Q, Band Type (Peak/LowShelf/HighShelf/LP/HP)
- **FFT Spectrum Overlay**: live spectrum analysis behind EQ curve (teal filled, semi-transparent)
- **Analog/Digital mode** toggle (Analog adds subtle harmonic coloring)
- Band enable/disable buttons
- EQ tone preset selector (existing, 13 presets)
- Matched visual style to Ozone 12 EQ

---

## EPIC B — Visualization (4 stories, B-4 done)

### B-1: Vectorscope Widget
**Status**: TODO
**Priority**: High

Create `gui/widgets/vectorscope.py` with `VectorscopeWidget(QWidget)`:
- Half-circle (180°) or full Lissajous display
- X-axis: L-R (stereo difference), Y-axis: L+R (stereo sum)
- Phosphor-style green dots with alpha decay (85% per frame)
- Downsample input to ~4096 points for performance
- Correlation bar below: -1 (out of phase) to +1 (mono)
- Size: 200x200 px
- Method: `set_audio_data(left_channel, right_channel)`

**File**: `gui/widgets/vectorscope.py`

### B-2: Transfer Curve Widget
**Status**: TODO
**Priority**: High

Create `gui/widgets/transfer_curve.py` with `TransferCurveWidget(QWidget)`:
- Square plot: input dB (X-axis) vs output dB (Y-axis)
- Range: -60 to 0 dB both axes
- 1:1 reference line (diagonal, dim gray dashed)
- Compression curve based on threshold/ratio/knee parameters
- Soft-knee region with smooth quadratic interpolation
- Method: `set_params(threshold, ratio, knee, makeup)`
- Size: 200x200 px

**File**: `gui/widgets/transfer_curve.py`

### B-3: Spectrum Analyzer Upgrade
**Status**: TODO
**Priority**: Medium

Create `gui/widgets/spectrum_analyzer.py` with `SpectrumAnalyzerWidget(QWidget)`:
- 4096-point FFT with Hann window
- Logarithmic frequency axis: 20 Hz – 20 kHz
- dB scale: -80 to 0 dB
- Teal filled polygon with darker outline
- Peak hold trace (white, 2s decay)
- Frequency labels: 20, 50, 100, 200, 500, 1k, 2k, 5k, 10k, 20k
- Method: `set_audio_data(samples, sample_rate)`
- Designed to overlay on EQ curve in A-6

**File**: `gui/widgets/spectrum_analyzer.py`

### B-4: Logic Channel Strip BEFORE/AFTER Meters
**Status**: ✅ DONE
**Commit**: feat(B-4)

---

## EPIC C — WLM Plus Clone (4 stories)

### C-1: WLM Plus Meter Panel
**Status**: TODO
**Priority**: High

Enhance `WavesWLMMeter` in ui_panel.py (or replace with new `WLMPlusMeterPanel`):
- 3 large numeric displays stacked vertically:
  - **Short-term LUFS** (largest font, primary reading)
  - **Integrated LUFS** (large font)
  - **Loudness Range** in LU (medium font)
- Color-coded backgrounds:
  - Green: within target ±2 LU
  - Yellow: within target ±5 LU
  - Red: outside ±5 LU
- Target LUFS reference line

### C-2: LED-Segment Horizontal Bars
**Status**: TODO
**Priority**: Medium

Add LED-segment horizontal bar meters to MetersPanel:
- **Momentary LUFS** bar: -60 to 0 LUFS, green→yellow→red
- **True Peak** bar: -60 to +3 dBTP, green→yellow→red with clip indicator
- **Gain Reduction** bar: 0 to -24 dB, teal color, right-to-left fill
- Segmented appearance (2px segments, 1px gaps)
- Numeric readout at end of each bar

### C-3: Loudness History Timeline
**Status**: TODO
**Priority**: Medium

Add `LoudnessHistoryWidget(QWidget)` to MetersPanel:
- Scrolling time-series graph of LUFS over time
- Y-axis: -60 to 0 LUFS
- X-axis: time (auto-scaling, last 30 seconds visible)
- Three traces: Momentary (thin), Short-term (medium), Integrated (thick dashed)
- Target LUFS line (horizontal dashed teal)
- QPainter rendered, updated at 10 Hz
- Method: `append_levels(momentary, short_term, integrated)`

### C-4: WLM Gain Trim + TP Limiter + CSV Log
**Status**: TODO
**Priority**: Low

Add to MetersPanel or WLM section:
- **Gain Trim** knob: ±18 dB (using OzoneRotaryKnob)
- **True Peak Limiter** toggle (checkbox, enables ceiling enforcement)
- **CSV Log Export** button: exports all captured loudness data to CSV file
  - Columns: timestamp, momentary_lufs, short_term_lufs, integrated_lufs, true_peak_l, true_peak_r, gain_reduction

---

## EPIC D — Presets + AI (3 stories)

### D-1: Expand Genre Presets to 50+
**Status**: TODO
**Priority**: Medium

Add genres to `GENRE_PROFILES` in genre_profiles.py:
- Shoegaze, Synthwave, Phonk, Hyperpop, Drill, Lo-Fi Hip Hop
- Thai Pop, Thai Luk Thung, Thai Mor Lam, Thai Isan
- K-Pop, J-Pop, Bollywood, Afrobeats, Reggaeton, Dembow
- Ambient, Drone, Post-Rock, Math Rock, Emo, Screamo
- Cumbia, Bossa Nova, Samba, Tango
- Each with appropriate EQ, dynamics, stereo, loudness settings

### D-2: 11 Platform Presets
**Status**: TODO
**Priority**: Medium

Add to `PLATFORM_TARGETS` in genre_profiles.py:
- **Deezer**: -15.0 LUFS, -1.0 dBTP
- **Podcasts**: -16.0 LUFS, -1.5 dBTP
- **Vinyl**: -12.0 LUFS, -0.5 dBTP
(Existing 8 platforms remain unchanged)

### D-3: AI Master Assistant
**Status**: TODO
**Priority**: High

Enhance `_build_ai_view()` in ui_panel.py:
- **Listen** button: analyze first 5 seconds of loaded audio
- **Auto-detect genre**: use spectral analysis + BPM to guess genre
- **Auto-set chain**: apply detected genre preset to all modules
- **Strength** slider (0–100%): blend between current settings and AI suggestion
- **Before/After Preview** button: preview with/without AI settings
- Visual: show detected genre, confidence %, recommended settings diff

---

## EPIC E — Workflow (5 stories)

### E-1: A/B Compare Toggle
**Status**: TODO
**Priority**: High

Add A/B compare functionality:
- **A/B Toggle** button in action bar (existing placeholder → make functional)
- Keyboard shortcut: **B** key toggles between Original and Mastered
- Label shows "ORIGINAL" (amber) or "MASTERED" (teal) prominently
- When "ORIGINAL": bypass chain, play unprocessed audio via QMediaPlayer
- When "MASTERED": play processed preview
- Meters update to reflect current playback source

### E-2: Match EQ
**Status**: TODO
**Priority**: Medium

Add Match EQ feature to EQ panel:
- **Load Reference** button: select reference WAV file
- Analyze reference FFT spectrum vs current track FFT spectrum
- Compute correction curve (difference between reference and source)
- **Strength** slider (0–100%): blend correction amount
- **Apply** button: apply correction curve as EQ band adjustments
- Visual: overlay reference spectrum (orange) on EQ curve

### E-3: Undo/Redo System
**Status**: TODO
**Priority**: High

Implement command history:
- `CommandHistory` class with undo/redo stacks (max 50 entries)
- Each parameter change creates a `Command(module, param, old_val, new_val)`
- **Ctrl+Z** → undo, **Ctrl+Shift+Z** → redo
- Toolbar buttons: Undo / Redo with greyed-out state when empty
- Status bar shows "Undo: [last action]" on undo

### E-4: AutoSave
**Status**: TODO
**Priority**: Medium

Implement automatic session saving:
- QTimer fires every 60 seconds
- Saves current chain settings to `~/.longplay_studio/autosave/`
- File format: `autosave_YYYYMMDD_HHMMSS.json`
- Keep last 10 autosaves, delete older ones
- On launch: check for autosave, offer "Recover last session?" dialog
- Manual save/load also uses `~/.longplay_studio/sessions/`

### E-5: Production Pipeline Wizard
**Status**: TODO
**Priority**: Low

Create `ProductionPipelineDialog(QDialog)`:
- 7-step wizard with Next/Back navigation:
  1. **Import**: Select audio files (drag & drop or file picker)
  2. **AI DJ**: Auto-order tracks (use existing AIDJ)
  3. **Compile**: Arrange tracks into album order with crossfades
  4. **Master**: Apply mastering chain (batch mode)
  5. **Hook Extract (Before)**: Extract hooks from original tracks
  6. **Hook Extract (After)**: Extract hooks from mastered tracks
  7. **Video + Export**: Generate video preview, export final files
- Progress bar per step
- Skip buttons for optional steps (Hook Extract, Video)

---

## EPIC F — Tests (2 stories)

### F-1: Unit Test Suite
**Status**: TODO
**Priority**: High

Create `tests/` directory with comprehensive tests:
- `tests/test_chain.py` — MasterChain init, process, save/load settings
- `tests/test_maximizer.py` — IRC modes, gain push, ceiling enforcement
- `tests/test_equalizer.py` — Band creation, frequency response, tone presets
- `tests/test_dynamics.py` — Compression ratio, threshold, multiband
- `tests/test_imager.py` — Stereo width, mono bass, balance
- `tests/test_loudness.py` — LUFS measurement accuracy, True Peak detection
- `tests/test_limiter.py` — Brickwall behavior, ceiling enforcement
- `tests/test_genre_profiles.py` — All presets load, platform targets valid
- `tests/test_ai_assist.py` — Recommendation generation
- `tests/test_widgets.py` — Widget instantiation (no display needed)
- **Target: 50+ test functions**, all passing

### F-2: Integration Verification Script
**Status**: TODO
**Priority**: Medium

Create `verify_longplay.sh`:
- Check Python version ≥ 3.10
- Check all dependencies installed (PyQt6, numpy, scipy, soundfile, pedalboard, pyloudnorm)
- Import test: `python3 -c 'from gui import LongPlayStudioV4; print("OK")'`
- Run pytest: `python3 -m pytest tests/ -q`
- Check Rust .so compiled: `ls *.so`
- Report pass/fail summary
- Exit code 0 if all pass, 1 if any fail

---

## Execution Order

B-4 is already done. Execute remaining 23 stories in this order:

```
A-2 → A-3 → A-1 → B-1 → B-2 → B-3 → C-1 → A-4 → A-5 → A-6 →
C-2 → C-3 → C-4 → D-1 → D-2 → E-1 → E-2 → E-3 → E-4 → E-5 →
D-3 → F-1 → F-2
```

**Rationale**: A-2 (rotary knob) is a dependency for A-3/A-4/A-5/A-6. B-1/B-2 are dependencies for A-5/A-4. Visualization widgets come before panel rebuilds that use them.

## Release
- Tag: `v5.6.0`
- Push to origin main after all stories complete
