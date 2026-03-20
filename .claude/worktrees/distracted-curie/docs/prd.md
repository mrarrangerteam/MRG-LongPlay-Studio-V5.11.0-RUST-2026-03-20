# Product Requirements Document — MRG LongPlay Studio V5.5

**Version:** 1.0
**Last Updated:** 2026-03-13
**Document Owner:** mrarrangerteam
**Status:** Approved

---

## 1. Introduction

This PRD defines the complete feature set and technical requirements for MRG LongPlay Studio V5.5 — a professional desktop application combining CapCut-class video editing, Logic Pro X + iZotope Ozone 12 mastering, and Waves WLM Plus metering.

**Reference Documents:**
- `docs/brief.md` — Project Brief
- `README.md` — Setup & Architecture Overview
- BMAD Audit Report (March 2026) — Bug fixes and gap analysis

---

## 2. Product Overview

### Problem Statement

Music producers using AI tools (Suno, Udio) need to master, assemble, and publish content across YouTube, Spotify, etc. Current workflow requires 5+ separate tools (DAW, mastering plugin, video editor, metering tool, upload tool). LongPlay Studio consolidates this into one professional-grade application.

### Value Proposition

One application that does: AI DJ ordering → Professional Mastering (Ozone-quality) → Video Assembly (CapCut-quality) → Loudness Verification (WLM Plus-quality) → Export/Upload — all with native Rust performance.

---

## 3. Functional Requirements

### FR-001: Rust Native Backend Compilation (P0)

**User Story:** As a developer, I want the Rust DSP engine compiled into a Python-importable module, so that mastering runs 10-100x faster than Python fallback.

**Acceptance Criteria:**
- `cd rust && maturin develop --release` succeeds on macOS ARM64
- `import longplay` works in Python
- `longplay.PyMasterChain` processes audio correctly
- Benchmark: 5-minute WAV file masters in < 5 seconds (Rust) vs ~60 seconds (Python)
- Python fallback still works if Rust module not available

**Technical Notes:**
- 11 Rust crates, 13,191 LOC already written
- PyO3 0.28 bindings in `longplay-python` crate
- Build system: maturin with `pyproject.toml`
- Test with: `rust/crates/longplay-chain/tests/integration_test.rs`

---

### FR-002: gui.py Modular Refactor (P0)

**User Story:** As a developer, I want gui.py split into logical modules, so that each component can be developed and tested independently.

**Acceptance Criteria:**
- gui.py split from 12,978 lines into ≤ 500 lines (main entry + imports)
- New structure:
  ```
  gui/
  ├── __init__.py
  ├── main.py              # Entry point, QMainWindow
  ├── styles.py             # Colors, themes, stylesheets
  ├── audio_player.py       # AudioPlayerWidget
  ├── timeline/
  │   ├── canvas.py         # TimelineCanvas
  │   ├── capcut_timeline.py # CapCutTimeline
  │   └── track_list.py     # TrackListItem, TrackControlsPanel
  ├── dialogs/
  │   ├── ai_dj.py          # AIDJDialog
  │   ├── ai_video.py       # AIVideoDialog
  │   ├── youtube_gen.py    # YouTubeGeneratorDialog
  │   ├── hook_extractor.py # HookExtractorDialog
  │   ├── video_prompt.py   # VideoPromptDialog
  │   └── timestamp.py      # TimestampDialog
  ├── video/
  │   ├── preview.py        # VideoPreviewCard, VideoThread
  │   ├── detached.py       # DetachedVideoWindow
  │   └── export.py         # Export engine
  ├── widgets/
  │   ├── meter.py          # RealTimeMeter, LUFSDisplay
  │   ├── waveform.py       # WaveformCache, ThumbnailCache
  │   ├── drop_zone.py      # DropZoneListWidget
  │   └── collapsible.py    # CollapsibleSection
  └── utils/
      ├── ffmpeg.py          # FFmpeg helpers, HW detection
      └── temp_dir.py        # Smart temp directory
  ```
- All 28 existing classes preserved with same API
- Application launches and all features work identically after refactor
- No circular imports

---

### FR-003: Multi-Track Timeline (P1)

**User Story:** As a video editor, I want a multi-track timeline with separate video, audio, text, and effects layers, so that I can compose complex projects like in CapCut.

**Acceptance Criteria:**
- Minimum 4 track types: Video, Audio, Text, Effects
- Unlimited tracks per type
- Drag-drop clips between tracks
- Clip trimming (drag edges)
- Clip splitting (razor tool)
- Track mute/solo/lock controls
- Vertical zoom (track height)
- Horizontal zoom (time scale)
- Playhead with snapping

**Dependencies:** FR-002 (refactored GUI)

---

### FR-004: Undo/Redo System (P1)

**User Story:** As an editor, I want to undo/redo any action, so that I can experiment without fear of losing work.

**Acceptance Criteria:**
- Command Pattern implementation with history stack
- Unlimited undo depth (limited by memory)
- Ctrl+Z / Cmd+Z for undo, Ctrl+Shift+Z / Cmd+Shift+Z for redo
- Actions covered: clip move, trim, split, delete, add, property change, mastering parameter change
- History panel showing action names

---

### FR-005: Keyframe Animation System (P1)

**User Story:** As a video editor, I want to animate any parameter over time with keyframes, so that I can create dynamic effects like CapCut.

**Acceptance Criteria:**
- Keyframe support for: position, scale, rotation, opacity, volume, EQ bands, any numeric parameter
- Linear, ease-in, ease-out, bezier interpolation
- Visual keyframe editor on timeline
- Diamond markers on track clips
- Copy/paste keyframes between clips

---

### FR-006: Text/Title Overlays (P1)

**User Story:** As a content creator, I want to add animated text overlays to my videos, so that I can create professional titles and captions.

**Acceptance Criteria:**
- Add text layer on timeline
- Font selection (system fonts)
- Color, size, alignment, outline, shadow controls
- Text animation presets (fade in, typewriter, slide, bounce) — minimum 10
- Position drag on video preview
- Export text burned into video via FFmpeg drawtext

---

### FR-007: Video Effects & Transitions Library (P2)

**User Story:** As a video editor, I want a library of effects and transitions, so that I can enhance my videos without external tools.

**Acceptance Criteria:**
- Transitions: Crossfade, Dissolve, Wipe (L/R/U/D), Zoom, Slide — minimum 10 types
- Transition duration adjustable (0.1s to 5.0s)
- Drag transitions between clips on timeline
- Effects: Brightness, Contrast, Saturation, Blur, Sharpen, Vignette — minimum 10
- Effect parameters adjustable with keyframes (FR-005)
- All implemented via FFmpeg filtergraph

---

### FR-008: Real-time Spectrum Analyzer (P2)

**User Story:** As a mastering engineer, I want a real-time spectrum analyzer overlaid on the EQ display, so that I can see the frequency content while adjusting EQ bands.

**Acceptance Criteria:**
- FFT-based spectrum display (4096-point)
- Updates at 30fps during preview playback
- Logarithmic frequency axis (20 Hz - 20 kHz)
- dB scale (-60 to 0 dB)
- Color gradient (cool blue → warm orange)
- Overlaid on EQ band display in mastering panel
- Pre/Post EQ toggle

---

### FR-009: Waves WLM Plus Exact Clone (P2)

**User Story:** As a mastering engineer, I want the loudness meter to look and function exactly like Waves WLM Plus, so that I get a familiar professional metering experience.

**Acceptance Criteria:**
- **Color scheme:** Teal-to-dark gradient background, green/yellow/red LED-segment meters
- **Layout:** Identical to WLM Plus — Momentary (large), Short-term (medium), Integrated (large), True Peak L/R bars, LRA display
- **Loudness Histogram:** Distribution graph showing LUFS values over time
- **Preset targets:** ITU-R BS.1770, EBU R128, ATSC A/85, custom
- **Logging:** Start/Stop/Reset controls
- **Export:** CSV and PDF loudness report
- **Numeric displays:** LED-style 7-segment font for all values
- **Gain Reduction:** Timeline history graph with peak hold

---

### FR-010: Real-time Audio Monitoring via Rust (P2)

**User Story:** As a mastering engineer, I want to hear the mastering chain applied in real-time during playback, so that I can tweak parameters and hear results instantly.

**Acceptance Criteria:**
- Rust audio thread using `cpal` crate for system audio output
- Audio routed through mastering chain in real-time
- Latency < 20ms (acceptable for monitoring)
- Bypass toggle (A/B compare) works in real-time
- Meter data streamed to GUI at 30fps
- No audio glitches when changing parameters

**Technical Notes:**
- Current architecture: offline render → play WAV file
- Target: real-time processing in Rust thread → stream to audio output
- Fallback: keep offline render + play for systems without Rust backend

---

### FR-011: Match EQ (P3)

**User Story:** As a mastering engineer, I want to match the frequency profile of a reference track, so that my masters have a consistent tonal balance.

**Acceptance Criteria:**
- Load reference audio file
- Analyze reference spectrum (FFT average)
- Analyze current audio spectrum
- Calculate difference curve
- Apply correction curve to EQ
- Adjustable match strength (0-100%)
- Visual overlay: reference spectrum vs current

---

### FR-012: Speed Ramp with Curve Editor (P2)

**User Story:** As a video editor, I want variable speed control with a bezier curve editor, so that I can create smooth speed ramps like CapCut.

**Acceptance Criteria:**
- Speed curve editor (0.1x to 10x)
- Bezier curve control points
- Preset curves: ease-in, ease-out, flash, montage
- Audio pitch correction during speed changes (optional)
- Visual speed indicator on timeline clip

---

### FR-013: GPU-Accelerated Video Preview (P3)

**User Story:** As a video editor, I want GPU-accelerated preview rendering, so that I can see multi-track compositions in real-time.

**Acceptance Criteria:**
- Metal (macOS) or Vulkan/OpenGL (cross-platform) rendering
- Composite multiple video tracks + text + effects at 30fps
- Hardware decode of video files
- Scrubbing at interactive frame rate
- Proxy file support for 4K editing

---

## 4. Non-Functional Requirements

### NFR-001: Performance
- Mastering (Rust): 5-min WAV in < 5 seconds
- Video export: Faster than real-time with HW acceleration
- GUI: 60fps UI rendering, no freezes during processing
- Memory: < 2GB RAM for typical 20-track project

### NFR-002: Reliability
- No crashes on any valid input file
- Graceful handling of corrupt files with user-friendly error messages
- Auto-save project state every 60 seconds
- Crash recovery from auto-save

### NFR-003: Compatibility
- macOS 13+ (Apple Silicon native, Intel via Rosetta)
- Windows 10+ (future)
- Linux (future, community)
- Audio: WAV, FLAC, MP3, AAC, OGG
- Video: MP4 (H.264/H.265), MOV, AVI, WebM

### NFR-004: Audio Quality
- True Peak accuracy: ±0.1 dBTP vs reference tool
- LUFS accuracy: ±0.5 LU vs reference tool
- Bit depth: 24-bit internal processing, 32-bit float intermediates
- Sample rate: Support 44.1, 48, 88.2, 96, 176.4, 192 kHz

---

## 5. Epics & User Stories

### Epic 1: Foundation — Compile & Refactor (Week 1-2)

| Story | Title | Priority | Points | Dependencies |
|-------|-------|----------|--------|--------------|
| 1.1 | Compile Rust backend to .so/.dylib via maturin | P0 | 5 | None |
| 1.2 | Verify Rust backend with integration tests | P0 | 3 | 1.1 |
| 1.3 | Refactor gui.py into modular structure | P0 | 8 | None |
| 1.4 | Add Python unit tests for mastering chain | P1 | 5 | 1.1 |
| 1.5 | Fix Rust/Python platform key normalization | P1 | 2 | 1.1 |
| 1.6 | Wire up Rust progress callbacks via PyO3 | P1 | 3 | 1.1 |

### Epic 2: CapCut Timeline — Multi-Track Foundation (Week 2-3)

| Story | Title | Priority | Points | Dependencies |
|-------|-------|----------|--------|--------------|
| 2.1 | Create Track model and multi-track data structure | P1 | 5 | 1.3 |
| 2.2 | Build multi-track timeline widget (video + audio layers) | P1 | 8 | 2.1 |
| 2.3 | Implement clip drag-drop between tracks | P1 | 5 | 2.2 |
| 2.4 | Implement clip trim and split | P1 | 5 | 2.2 |
| 2.5 | Implement Undo/Redo command pattern | P1 | 5 | 2.1 |
| 2.6 | Connect multi-track to video export pipeline | P1 | 5 | 2.2 |

### Epic 3: CapCut Features — Text, Effects, Keyframes (Week 3-5)

| Story | Title | Priority | Points | Dependencies |
|-------|-------|----------|--------|--------------|
| 3.1 | Implement keyframe animation system | P1 | 8 | 2.2 |
| 3.2 | Add text/title overlay layer with templates | P1 | 5 | 2.2, 3.1 |
| 3.3 | Build transitions library (10+ types via FFmpeg) | P2 | 5 | 2.2 |
| 3.4 | Build effects library (brightness, contrast, blur, etc.) | P2 | 5 | 2.2, 3.1 |
| 3.5 | Implement speed ramp with curve editor | P2 | 5 | 3.1 |
| 3.6 | Add export format presets (resolution, codec, platform) | P2 | 3 | 2.6 |

### Epic 4: Mastering — Ozone Pro Features (Week 4-6)

| Story | Title | Priority | Points | Dependencies |
|-------|-------|----------|--------|--------------|
| 4.1 | Real-time spectrum analyzer overlay on EQ | P2 | 5 | 1.1 |
| 4.2 | Waves WLM Plus exact clone (color, layout, histogram) | P2 | 8 | 1.3 |
| 4.3 | Match EQ feature (reference track matching) | P3 | 5 | 4.1 |
| 4.4 | Real-time audio monitoring via Rust (cpal) | P2 | 8 | 1.1, 1.6 |
| 4.5 | A/B comparison with instant toggle | P2 | 3 | 4.4 |
| 4.6 | Loudness report export (CSV/PDF) | P2 | 3 | 4.2 |

### Epic 5: Polish & Production (Week 6-8)

| Story | Title | Priority | Points | Dependencies |
|-------|-------|----------|--------|--------------|
| 5.1 | GPU-accelerated video preview (Metal/wgpu) | P3 | 13 | 2.2 |
| 5.2 | symphonia integration for native audio I/O | P3 | 5 | 1.1 |
| 5.3 | Auto-save and crash recovery | P2 | 5 | 2.5 |
| 5.4 | Vintage hardware UI polish (final pass) | P2 | 5 | All |
| 5.5 | Performance optimization and profiling | P2 | 5 | All |
| 5.6 | PyInstaller build for macOS .app distribution | P2 | 5 | All |

---

## 6. Timeline & Milestones

| Milestone | Target | Epic | Key Deliverable |
|-----------|--------|------|-----------------|
| M1: Rust Engine Live | Week 2 | Epic 1 | `import longplay` works, 10x faster mastering |
| M2: Multi-Track Timeline | Week 3 | Epic 2 | CapCut-style multi-track editing works |
| M3: Full Editor | Week 5 | Epic 3 | Text, effects, keyframes, transitions |
| M4: Pro Mastering | Week 6 | Epic 4 | Spectrum analyzer, WLM Plus clone, real-time monitoring |
| M5: V5.5 Release | Week 8 | Epic 5 | Production-ready, packaged .app |

---

## 7. Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Rust compilation fails on macOS | Medium | High | Python fallback always available; fix build issues incrementally |
| PyQt6 performance for multi-track | Medium | Medium | Use QGraphicsView for timeline; consider QML for future |
| Real-time audio latency too high | Low | Medium | Use Rust thread + cpal; fallback to offline preview |
| FFmpeg filtergraph complexity for effects | Medium | Low | Pre-tested filter templates; test each effect individually |
| gui.py refactor breaks existing features | Medium | High | Comprehensive test suite before refactor; git branch per story |

---

## 8. Technical Preferences (for Claude Code)

```yaml
# .claude/technical-preferences.md equivalent
language_primary: Rust (DSP, audio I/O, analysis)
language_secondary: Python (GUI, AI logic, orchestration)
gui_framework: PyQt6 (with PySide6 fallback)
build_rust: maturin (PyO3)
build_python: pip + requirements.txt
audio_processing: Rust (longplay-dsp) > Python (pedalboard+scipy) > FFmpeg (last resort)
video_processing: FFmpeg (subprocess for now, future: ffmpeg-next Rust crate)
testing: pytest (Python), cargo test (Rust)
style:
  python: PEP 8, type hints, docstrings
  rust: rustfmt, clippy clean
  commits: Conventional Commits (feat:, fix:, refactor:)
branching: feature branches from main, PR per story
```

---

## 9. How to Use This PRD with Claude Code

### Starting a Story

```bash
# In Claude Code, say:
# "Read docs/prd.md and implement Story 1.1: Compile Rust backend"
# Claude Code will:
# 1. Read this PRD for context
# 2. Read the relevant crate code
# 3. Implement the story
# 4. Run tests
# 5. Commit with conventional commit message
```

### Story Workflow

1. **Pick a story** from the Epics section above
2. **Tell Claude Code:** "Implement Story X.Y from docs/prd.md"
3. **Claude Code reads** PRD → brief.md → relevant source files
4. **Implements** the story following Technical Preferences
5. **Tests** with acceptance criteria from this PRD
6. **Commits** with `feat(scope): description` format
7. **Move to next story** in dependency order

### Recommended Story Order

```
1.1 → 1.2 → 1.3 → 1.4 → 1.5 → 1.6  (Foundation)
  ↓
2.1 → 2.2 → 2.3 → 2.4 → 2.5 → 2.6  (Timeline)
  ↓
3.1 → 3.2 → 3.3 → 3.4 → 3.5 → 3.6  (Features)
  ↓
4.1 → 4.2 → 4.3 → 4.4 → 4.5 → 4.6  (Mastering)
  ↓
5.1 → 5.2 → 5.3 → 5.4 → 5.5 → 5.6  (Polish)
```

---

## Appendix A: Existing Codebase Reference

### Python Files (27,022 LOC)

| File | Lines | Role |
|------|-------|------|
| gui.py | 12,978 | Main app — 28 classes, needs refactor (Story 1.3) |
| modules/master/ui_panel.py | 6,250 | Mastering GUI — Waves-inspired, production quality |
| modules/master/chain.py | 1,537 | Master chain — real DSP processing |
| modules/master/genre_profiles.py | 1,242 | 30+ genre profiles, IRC modes, platform targets |
| ai_dj.py | 709 | AI DJ playlist ordering |
| modules/master/maximizer.py | 519 | Ozone 12-style maximizer |
| hook_extractor.py | 510 | Audio hook detection |
| video_prompt_generator.py | 528 | AI video prompt generation |
| modules/master/analyzer.py | 476 | Audio analysis (spectral, dynamic, stereo) |
| modules/master/limiter.py | 373 | Look-ahead true peak limiter |
| modules/master/dynamics.py | 306 | Multiband dynamics/compressor |
| modules/master/equalizer.py | 288 | 8-band parametric EQ |
| modules/master/rust_chain.py | 289 | Rust/C++/Python fallback bridge |
| modules/master/loudness.py | 269 | LUFS measurement |
| modules/master/imager.py | 267 | Stereo imager |
| modules/master/ai_assist.py | 241 | AI recommendation engine (FIXED in v5.5) |
| license_manager.py | 225 | License validation |

### Rust Crates (13,191 LOC)

| Crate | Lines | Role |
|-------|-------|------|
| longplay-dsp | 4,284 | EQ, Dynamics, Imager, Maximizer, IRC Limiter |
| longplay-analysis | 1,725 | FFT, LUFS, spectral, stereo analysis |
| longplay-profiles | 1,535 | Genre profiles, IRC modes, tone presets |
| longplay-chain | 1,463 | Master chain orchestrator |
| longplay-aidj | 871 | AI DJ (Rust) |
| longplay-python | 727 | PyO3 bindings |
| longplay-hooks | 562 | Hook extraction (Rust) |
| longplay-io | 462 | WAV I/O via hound |
| longplay-cli | 390 | CLI interface |
| longplay-core | 312 | Type definitions |
| longplay-ai | 297 | AI recommendation |

### Signal Flow (Mastering Chain)

```
Input → Pre-gain (-3dB) → EQ → Dynamics → Imager → Maximizer → Loudness Norm → True Peak Limit → Output
```
