# PRD Phase 3 — Ozone 12 Clone + Logic Pro X Metering + Production Flow

**Version:** 3.0
**Status:** NEW — Deep feature implementation based on real Ozone 12 / Logic Pro X reference screenshots
**Goal:** Make mastering panel match real Ozone 12 at pixel-level detail + implement correct production workflow

---

## REFERENCE ANALYSIS (from screenshots)

### Ozone 12 Maximizer (Screenshots 1-2)
- IRC dropdown: IRC 1, IRC 2, IRC 3 (→ Pumping/Balanced/Crisp/Clipping), IRC 4 (→ Classic/Modern/Transient), IRC 5, IRC Low Latency
- Gain knob: large rotary knob showing "+6.4 dB"
- Output Level: -1.00 dBTP with True Peak toggle (blue)
- Character slider: Smooth ↔ Aggressive (0.00 to 10.00)
- Upward Compress: 0.0 dB
- Soft Clip: 0%
- Transient Emphasis: 0% with H/M/L selector
- Stereo Independence: Transient 0% / Sustain 0%
- Spectrum display: teal waveform showing limiting activity
- Learn Input Gain button showing "-11.0 LUFS"

### Logic Pro X Channel Strip (Screenshot 3)
- Vertical level meter (green/yellow/red gradient)
- Peak hold indicators
- Real-time VU showing signal as it passes through chain
- Numbers: 0.0, -1.0 at bottom of fader
- M/S/R buttons (Mute, Solo, Record)
- Ozone 12 as insert plugin
- Loudness Meter window: M/S/I columns, LU Range, Integrated, Pause/Reset buttons

### Ozone 12 Imager (Screenshot 4)
- 4 band sliders (0 to -20 dB range)
- Stereoize I / II toggle
- Crossover points: 60, 100, 300, 600, 1k, 3k, 6k, 10k Hz
- Amount slider (cyan bar)
- Learn button
- Recover Sides: 0.0 dB knob
- Vectorscope (half-circle) showing stereo image
- Per-band: (i) info and (S) solo buttons
- (i) and (S) buttons with color indicators (pink = active)

### Ozone 12 Dynamics (Screenshot 5)  
- Multiband frequency crossover graph at top
- 4 band crossover points adjustable
- Per-band: Threshold, Ratio, Attack, Release, Knee
- Compressor section: Ratio 10.0:1, Attack 20ms, Release 100ms, Knee 10.0
- Limiter section: Ratio 1.5:1, Attack 30ms, Release 200ms, Knee 5.0
- Detection: Peak / Env / RMS selector
- Parallel: Dry ↔ Wet slider (0-100, showing 55)
- Band 1 Gain: 0.0 dB
- Transfer curve display (compression knee visualization)
- Link Bands toggle
- A/L (Above/Below) frequency band selector

### Metering Panel (Screenshot 3 - Loudness Meter)
- Columns: M (Momentary), S (Short-term), I (Integrated)
- Values: -7.3, -8.3, -9.0 LUFS
- LED-bar meters: green/yellow/red gradient
- Scale: 0 to -40 LUFS
- LU Range: 2.9
- Integrated: -9.0
- Pause / Reset buttons

---

## ALGORITHM & LANGUAGE ANALYSIS

### What Ozone 12 Uses (Reference)
- **Core DSP:** C++ with SIMD (SSE/AVX/NEON) for real-time processing
- **IRC Limiter:** Custom look-ahead limiter with spectral weighting (not simple peak limiter)
- **True Peak:** ITU-R BS.1770-4 compliant 4x oversampling
- **Multiband:** Linkwitz-Riley crossover filters (linear phase)
- **Loudness:** ITU-R BS.1770-4 K-weighted loudness measurement
- **GUI:** JUCE framework (C++ cross-platform audio GUI)

### What We Should Use (Matching Architecture)
- **Core DSP:** Rust (equivalent to C++ but memory-safe) — already have longplay-dsp crate
- **IRC Limiter:** Rust longplay-dsp/irc_limiter.rs — already written, needs optimization
- **True Peak:** Rust with 4x oversampling via polyphase — already in limiter.py
- **Multiband:** Rust longplay-dsp with Linkwitz-Riley crossover — already in chain.py
- **Loudness:** pyloudnorm (Python) + future Rust port
- **GUI:** PyQt6 with QPainter custom widgets (equivalent to JUCE Canvas)

---

## PRODUCTION FLOW DESIGN

```
┌─────────────────────────────────────────────────────────────┐
│                 PRODUCTION PIPELINE                          │
│                                                              │
│  ┌──────────┐   ┌──────────┐   ┌───────────┐               │
│  │ 1. SONGS │ → │ 2. AI DJ │ → │ 3. COMPILE│               │
│  │ Import   │   │ Order &  │   │ Crossfade │               │
│  │ Tracks   │   │ Arrange  │   │ Concat    │               │
│  └──────────┘   └──────────┘   └─────┬─────┘               │
│                                       │                      │
│                                       ▼                      │
│  ┌──────────────────────────────────────────┐               │
│  │ 4. MASTER (Ozone 12 Quality)             │               │
│  │ EQ → Dynamics → Imager → Maximizer       │               │
│  │ → Loudness Norm → True Peak Limit        │               │
│  └────────────────────┬─────────────────────┘               │
│                        │                                     │
│              ┌─────────┼──────────┐                          │
│              ▼         ▼          ▼                          │
│  ┌────────────┐ ┌──────────┐ ┌──────────┐                   │
│  │ 5a. HOOK   │ │ 5b. FULL │ │ 5c. HOOK │                   │
│  │ EXTRACT    │ │ MASTERED │ │ BEFORE   │                   │
│  │ (after     │ │ EXPORT   │ │ MASTER   │                   │
│  │  master)   │ │          │ │ (option) │                   │
│  └─────┬──────┘ └──────────┘ └──────────┘                   │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────────────────────────────────┐               │
│  │ 6. VIDEO ASSEMBLY                         │               │
│  │ Audio (mastered) + Video/Lip-sync         │               │
│  │ + Text overlay + Transitions + Effects    │               │
│  └────────────────────┬─────────────────────┘               │
│                        │                                     │
│                        ▼                                     │
│  ┌──────────────────────────────────────────┐               │
│  │ 7. EXPORT                                 │               │
│  │ YouTube / Spotify / TikTok presets        │               │
│  │ + Loudness Report (CSV/PDF)               │               │
│  └──────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────┘
```

### Hook Extract: Before vs After Master (User Choice)
- **After Master (Recommended):** Hook has final mastered sound, consistent loudness
- **Before Master:** Raw hook for re-processing or different mastering treatment
- **UI:** Radio button in Hook Extractor dialog: "Extract from: ○ Original ○ Mastered"

---

## STORIES

### Story P3-1: Real-time Level Meter (Logic Pro X Style) — P0

**What:** Add vertical level meter strip (like Logic Pro X channel strip) showing real-time signal level as audio passes through each stage of mastering chain.

**Visual Reference:** Screenshot 3 left side — tall green/yellow/red gradient bar with peak hold

**Implementation:**
- Custom QPainter widget: `LogicLevelMeter`
- Gradient: green (-inf to -12 dB) → yellow (-12 to -3 dB) → red (-3 to 0 dB)
- Peak hold line (white horizontal) with 2-second decay
- Numeric peak display at top (e.g., "-0.5")
- Show BEFORE and AFTER meters side by side
- Clip indicator (red dot) when True Peak > ceiling
- Stereo: L/R separate bars
- Update rate: 30fps from chain meter callback

**Algorithm:**
- Peak measurement: `20 * log10(max(abs(samples)))` per 512-sample block
- RMS measurement: `20 * log10(sqrt(mean(samples^2)))` per block  
- Peak hold: track max over 2-second window, decay at 20 dB/sec
- True Peak: 4x oversampled via `scipy.signal.resample_poly(block, 4, 1)`

**Language:** Python (QPainter) for visual, Rust for DSP measurement (via PyO3 callback)

**File to edit:** `modules/master/ui_panel.py` — add `LogicLevelMeter` widget class, wire into meters section

**Acceptance Criteria:**
- L/R green/yellow/red bars visible in mastering panel
- Peak hold lines with numeric readout
- Clip indicator when > ceiling
- BEFORE/AFTER comparison mode
- Updates at 30fps during processing

---

### Story P3-2: Ozone 12 Maximizer Panel (Full Clone) — P0

**What:** Rebuild the Maximizer section in ui_panel.py to match Ozone 12 exactly.

**Visual Reference:** Screenshots 1-2

**Implementation:**
- Large rotary Gain knob (QPainter drawn, shows "+6.4 dB")
- IRC dropdown with sub-mode flyout menu (IRC 3 → Pumping/Balanced/Crisp/Clipping, IRC 4 → Classic/Modern/Transient)
- Output Level spinbox: -1.00 dBTP
- True Peak toggle (blue switch)
- Character slider: Smooth ↔ Aggressive (0.00-10.00)
- Upward Compress knob: 0.0-12.0 dB
- Soft Clip knob: 0-100%
- Transient Emphasis: 0-100% with H/M/L radio buttons
- Stereo Independence: Transient 0-100% / Sustain 0-100%
- Spectrum waveform display (teal) showing limiter activity
- Learn Input Gain button → analyze and display measured LUFS

**Algorithm:** All parameters already exist in `modules/master/maximizer.py` — this story is pure UI wiring

**File to edit:** `modules/master/ui_panel.py` — rebuild `_build_maximizer_view()`

---

### Story P3-3: Ozone 12 Imager Panel (Full Clone) — P1

**What:** Rebuild the Imager section to match Ozone 12 exactly.

**Visual Reference:** Screenshot 4

**Implementation:**
- 4-band vertical sliders (width: 0 to 200%, display as -20 to +20 or 0-200)
- Frequency crossover display with draggable crossover points
- Stereoize I/II toggle per band
- Recover Sides knob
- Amount slider (horizontal cyan bar)
- Vectorscope display (half-circle QPainter) showing M/S correlation
- Per-band Solo (S) and Info (i) buttons with color indicators
- Learn button

**Algorithm:**
- Mid/Side processing: `mid = (L+R)/2, side = (L-R)/2`
- Width: `side_scaled = side * (width/100)`, then `L = mid + side_scaled, R = mid - side_scaled`
- Vectorscope: plot (L-R) vs (L+R) in polar coordinates, QPainter drawPath
- Crossover: Linkwitz-Riley 4th order (already in chain.py `_CrossoverFilter`)

**Language:** Rust (DSP) + Python (QPainter UI)

**File to edit:** `modules/master/ui_panel.py` — rebuild `_build_imager_view()`

---

### Story P3-4: Ozone 12 Dynamics Panel (Full Clone) — P1

**What:** Rebuild the Dynamics section to match Ozone 12 exactly.

**Visual Reference:** Screenshot 5

**Implementation:**
- Multiband frequency crossover graph (top) with draggable split points
- Per-band controls: Threshold, Ratio, Attack, Release, Knee (all knobs)
- Compressor section with Threshold knob (large, shows "-30.0 dB")
- Limiter section with separate Ratio/Attack/Release/Knee
- Detection mode selector: Peak / Env / RMS
- Parallel mix slider: Dry ↔ Wet (0-100)
- Band Gain knob per band
- Transfer curve display (QPainter: input dB on X, output dB on Y, showing knee)
- Link Bands toggle
- A/L (Above threshold / Below threshold) selector

**Algorithm:**
- Compressor: envelope-follower → gain computer → smooth gain → apply
- Envelope: `env[n] = attack * env[n-1] + (1-attack) * abs(input[n])` (peak mode)
- Gain computer: `if env_dB > threshold: gain = threshold + (env_dB - threshold) / ratio`
- Knee: soft-knee interpolation around threshold ± knee/2
- Parallel: `output = dry * input + wet * compressed`
- All already in `modules/master/dynamics.py` and Rust `longplay-dsp/dynamics.rs`

**Language:** Rust (DSP) + Python (QPainter UI)

**File to edit:** `modules/master/ui_panel.py` — rebuild `_build_dynamics_view()`

---

### Story P3-5: Loudness Meter Panel (Ozone + Logic Style) — P1

**What:** Combine Waves WLM Plus meter with Logic Pro X loudness display.

**Visual Reference:** Screenshot 3 — Loudness Meter window

**Implementation:**
- 3 columns: M (Momentary), S (Short-term), I (Integrated)
- LED-bar meters: green → yellow → red gradient
- Scale: 0 to -40 LUFS with markings
- LU Range display
- Integrated display (large number)
- Pause / Reset buttons
- Peak/RMS numeric readout like Logic (top of meter panel)
- True Peak L/R bars
- History graph (timeline of LUFS over time)

**Algorithm:**
- LUFS: ITU-R BS.1770-4 K-weighted filter → 400ms blocks (momentary), 3s blocks (short-term), gated integration
- Already using `pyloudnorm` — just need to wire to UI at 30fps refresh

**File to edit:** `modules/master/ui_panel.py` — rebuild meters section

---

### Story P3-6: Production Flow Pipeline — P0

**What:** Implement the correct production workflow in the main app.

**Flow:**
1. Import songs → 2. AI DJ order → 3. Compile (crossfade/concat) → 4. Master → 5. Hook extract (before/after choice) → 6. Video assembly → 7. Export

**Implementation:**
- Add "Production Pipeline" tab or wizard in main window
- Step-by-step flow with progress indicator
- At Hook Extract step: radio button "Extract from: ○ Original ○ Mastered"
- At Video step: option for "Video" or "Lip-sync Video"
- Each step can be skipped or re-done
- Final export with loudness report

**File to edit:** `gui/main.py` — add production pipeline panel/wizard

---

### Story P3-7: Rotary Knob Widget (Ozone Style) — P1

**What:** Create a reusable rotary knob widget that matches Ozone 12's knobs.

**Visual Reference:** All screenshots — the circular knobs with value display

**Implementation:**
- QPainter drawn: dark circle with arc indicator
- Value display in center (e.g., "+6.4 dB")
- Mouse drag to adjust (vertical drag = value change)
- Double-click to type exact value
- Range configurable (min, max, default, step)
- Label below knob
- Optional: unit suffix (dB, %, ms)

**Algorithm:** Purely UI — `angle = (value - min) / (max - min) * 270` degrees arc

**File to create:** `gui/widgets/rotary_knob.py`

---

### Story P3-8: Vectorscope Widget — P1

**What:** Create vectorscope display for Imager showing stereo field.

**Visual Reference:** Screenshot 4 — half-circle at bottom of Imager

**Implementation:**
- Half-circle QPainter display
- L-R on X axis, L+R on Y axis
- Points rendered as fading dots (phosphor style)
- Color: teal/cyan gradient
- Center line = mono, spread = stereo width
- Correlation meter below (-1 to +1)

**Algorithm:**
- For each sample pair (L,R): `x = (L-R) * 0.707, y = (L+R) * 0.707`
- Plot in polar: if mostly vertical = mono, if spread = stereo
- Correlation: `sum(L*R) / sqrt(sum(L^2) * sum(R^2))`

**File to create:** `gui/widgets/vectorscope.py`

---

### Story P3-9: Transfer Curve Widget — P1

**What:** Create transfer curve display for Dynamics showing compression knee.

**Visual Reference:** Screenshot 5 — bottom right shows the compression curve

**Implementation:**
- QPainter: input dB on X axis, output dB on Y axis
- 1:1 line (no compression) as reference
- Actual curve showing threshold, ratio, knee
- Above threshold: curve bends according to ratio
- Soft knee: smooth transition around threshold

**Algorithm:**
- Below threshold: `output = input` (1:1 line)
- Above threshold: `output = threshold + (input - threshold) / ratio`
- Knee: smooth interpolation using `2 * (input - threshold + knee/2)^2 / (2 * knee)` in knee region

**File to create:** `gui/widgets/transfer_curve.py`

---

### Story P3-10: Final Integration + Visual Polish — P0

**What:** Wire all P3 stories into the running app, ensure everything looks and works like Ozone 12.

**Acceptance Criteria:**
- Mastering panel visually matches Ozone 12 screenshots
- All knobs are rotary (not sliders) for Gain, Character, Upward, etc.
- Level meters show real-time signal during processing
- Vectorscope shows stereo field
- Transfer curve shows dynamics response  
- Production pipeline flow works end-to-end
- Export report includes all loudness data

---

## EXECUTION ORDER

```
P3-7  → Rotary Knob widget (needed by P3-2, P3-3, P3-4)
P3-8  → Vectorscope widget (needed by P3-3)
P3-9  → Transfer Curve widget (needed by P3-4)
P3-1  → Logic Level Meter
P3-2  → Maximizer Panel (full Ozone clone)
P3-3  → Imager Panel (full Ozone clone)
P3-4  → Dynamics Panel (full Ozone clone)
P3-5  → Loudness Meter Panel
P3-6  → Production Flow Pipeline
P3-10 → Final Integration
```

---

## CLAUDE CODE COMMAND

```
Read CLAUDE.md and docs/prd-phase3-ozone-clone.md then execute ALL 10 stories 
(P3-1 through P3-10) using RALP loop. Rules: 
1) DO NOT ask questions — make decisions yourself 
2) Stories P3-7, P3-8, P3-9 CREATE new widget files in gui/widgets/ 
3) All other stories EDIT modules/master/ui_panel.py and gui/main.py 
4) Use QPainter for all custom visual widgets (knobs, meters, vectorscope, curves)
5) Match Ozone 12 visual style: dark background (#1a1a2e), teal accents (#00d4aa), 
   white text, gradient meters green→yellow→red
6) Test after every story 
7) Self-review: run wiring report before push 
8) Do not stop until P3-10 is complete
```
