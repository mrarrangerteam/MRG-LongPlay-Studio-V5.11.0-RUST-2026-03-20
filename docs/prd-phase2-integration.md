# PRD Phase 2 — INTEGRATION: Wire Everything Into Working App

**Version:** 2.0  
**Status:** URGENT — Phase 1 modules exist but NONE are connected to running app  
**Goal:** Make every module from Phase 1 actually visible and usable in the GUI

---

## THE PROBLEM

Phase 1 created 46 new Python files (19,600 LOC) but **ZERO are wired into the running application.**

Evidence — every new module has 0 references in gui/main.py:
- MultiTrackTimeline: **0** — app still uses old single-track timeline
- KeyframeEditor: **0** — no way to add keyframes
- TextOverlay: **0** — no text tool button
- TransitionLibrary: **0** — no transitions panel
- EffectsLibrary: **0** — no effects panel
- SpeedRamp: **0** — no speed curve editor
- SpectrumAnalyzer: **0** — not overlaid on EQ
- WLM Plus Meter: **0 in mastering panel** — not replacing old meter
- MatchEQ: **0** — no Match EQ button
- ABComparison: **0** — no A/B toggle
- AutoSave: **0** — no auto-save running
- VintageTheme: **0** — old theme still active
- UndoRedo: **0** — Ctrl+Z does nothing
- LoudnessReport: **0** — no export button
- RealtimeMonitor: **0** — still offline render
- GPUPreview: **0** — still OpenCV
- ExportPresets: **0** — no preset selector

The app opens and looks IDENTICAL to before Phase 1. User sees zero new features.

---

## ARCHITECTURE RULE

**DO NOT create new files.** All work in this phase is EDITING EXISTING FILES to import and instantiate the modules that already exist.

Primary files to edit:
- `gui/main.py` (6,663 lines) — main window, toolbar, panels
- `modules/master/ui_panel.py` (6,266 lines) — mastering panel
- `gui.py` (12,981 lines) — original entry point (may need updates)

Secondary files that may need small edits:
- `gui/__init__.py` — re-exports
- `gui/styles.py` — if vintage theme needs to hook in
- Existing modules — if API adapters are needed

---

## CRITICAL INSTRUCTION FOR CLAUDE CODE

```
READ THIS CAREFULLY:

1. DO NOT create new .py files. All modules already exist.
2. Your job is ONLY to edit gui/main.py and modules/master/ui_panel.py 
   to import and use the existing modules.
3. Every change must be testable — after each story, the app must still 
   launch without crash.
4. Use try/except ImportError for every new import so the app doesn't 
   crash if a module has issues.
5. Test after EVERY story: python3 -c "from gui import LongPlayStudioV4"
6. Run full tests after every story: python3 -m pytest tests/ -v
```

---

## STORIES — Execute in EXACT order

### Story I-1: Wire Undo/Redo System (P0 — Everything depends on this)

**File to edit:** `gui/main.py`

**What to do:**
1. Import `UndoRedoManager` from `gui.models.commands`
2. Create `self.undo_manager = UndoRedoManager()` in `__init__`
3. Add Ctrl+Z / Cmd+Z shortcut → `self.undo_manager.undo()`
4. Add Ctrl+Shift+Z / Cmd+Shift+Z → `self.undo_manager.redo()`
5. Add Undo/Redo buttons to toolbar
6. Connect undo_manager to status bar for action name display

**Acceptance Criteria:**
- Ctrl+Z triggers undo (verify with print statement)
- Ctrl+Shift+Z triggers redo
- Buttons visible in toolbar
- App does not crash

**Guard pattern:**
```python
try:
    from gui.models.commands import UndoRedoManager
    _HAS_UNDO = True
except ImportError:
    _HAS_UNDO = False
```

---

### Story I-2: Wire AutoSave System (P0 — Prevents data loss)

**File to edit:** `gui/main.py`

**What to do:**
1. Import `AutoSaveManager` from `gui.models.autosave`
2. Create `self.auto_save = AutoSaveManager()` in `__init__`
3. Start auto-save timer: `self.auto_save.start(interval_sec=60)`
4. On app close, call `self.auto_save.stop()`
5. On app launch, check for recovery: `self.auto_save.check_recovery()`
6. If recovery available, show dialog asking user if they want to restore

**Acceptance Criteria:**
- Auto-save file created in `~/.longplay_studio/autosave/` after 60 seconds
- Recovery dialog shows on next launch if previous session crashed
- Clean exit cleans up auto-save

---

### Story I-3: Apply Vintage Theme System (P0 — Visual identity)

**File to edit:** `gui/main.py`

**What to do:**
1. Import `VintageTheme, get_theme, THEME_NAMES` from `gui.styles_vintage`
2. At app startup, apply theme: `theme = get_theme("Waves SSL"); theme.apply(self)`
3. Add Theme selector in View menu or Settings: combo box with THEME_NAMES
4. When theme changes, call `theme.apply(self)` to re-style entire app

**Acceptance Criteria:**
- App launches with dark vintage theme (gunmetal/amber/teal)
- Theme selector visible in menu
- Switching themes changes all widgets immediately
- All panels (mastering, timeline, video) use consistent theme

---

### Story I-4: Wire Spectrum Analyzer into Mastering Panel (P1)

**File to edit:** `modules/master/ui_panel.py`

**What to do:**
1. Import `SpectrumAnalyzerWidget` from `gui.widgets.spectrum_analyzer`
2. In `MasterPanel.__init__`, create spectrum widget
3. Add it as overlay on top of the EQ band display (use QStackedWidget or overlay layout)
4. Connect to audio data: when preview plays, feed samples to analyzer
5. Add Pre/Post EQ toggle button
6. Start 30fps update timer when playing, stop when paused

**Acceptance Criteria:**
- Spectrum visible as colored overlay on EQ section
- Updates during preview playback
- Pre/Post toggle switches measurement point
- Logarithmic frequency axis, 20Hz-20kHz

---

### Story I-5: Wire WLM Plus Meter into Mastering Panel (P1)

**File to edit:** `modules/master/ui_panel.py`

**What to do:**
1. Import `WavesWLMPlusMeter` from `gui.widgets.wlm_meter`
2. Replace existing `WavesWLMMeter` widget with `WavesWLMPlusMeter`
3. Connect meter data from chain processing to new meter
4. Ensure LED-segment style, teal gradient, histogram all visible
5. Add Start/Stop/Reset logging controls
6. Wire gain reduction history display

**Acceptance Criteria:**
- Meter panel looks like Waves WLM Plus (teal background, LED segments)
- Momentary/Short-term/Integrated/TP L/R all updating
- Histogram visible
- Gain reduction history graph visible

---

### Story I-6: Wire Match EQ into Mastering Panel (P1)

**File to edit:** `modules/master/ui_panel.py`

**What to do:**
1. Import `MatchEQ` from `modules.master.match_eq`
2. Add "Match EQ" tab or button in EQ section
3. Add "Load Reference" file picker button
4. When reference loaded, show spectrum comparison overlay
5. Add strength slider (0-100%)
6. Add "Apply" button that writes correction curve to chain EQ

**Acceptance Criteria:**
- Load reference audio file via file picker
- See reference vs current spectrum overlay
- Adjust match strength
- Apply correction to EQ bands

---

### Story I-7: Wire A/B Compare into Mastering Panel (P1)

**File to edit:** `modules/master/ui_panel.py`

**What to do:**
1. Import `ABComparison` from `modules.master.ab_compare`
2. Add A/B toggle button in transport bar (big, visible button)
3. Connect to audio playback: when toggled, bypass all processing
4. Show "ORIGINAL" / "MASTERED" label on toggle state
5. Add keyboard shortcut (Space or B key)
6. Wire loudness-matched mode checkbox

**Acceptance Criteria:**
- A/B button visible in mastering transport
- Toggle switches between processed/unprocessed instantly
- Label shows current mode
- Loudness-matched option available

---

### Story I-8: Wire Loudness Report Export (P1)

**File to edit:** `modules/master/ui_panel.py`

**What to do:**
1. Import `export_csv, export_pdf, LoudnessReportData` from `modules.master.loudness_report`
2. Add "Export Report" button in meters section
3. On click, show save dialog with CSV/PDF format choice
4. Collect current loudness data from chain and meters
5. Generate report and save to chosen path
6. Show success message

**Acceptance Criteria:**
- Export Report button visible
- Saves CSV with all loudness data
- Saves PDF with formatted report
- File contains correct values from current session

---

### Story I-9: Replace Timeline with Multi-Track Timeline (P1 — Biggest visual change)

**File to edit:** `gui/main.py`

**What to do:**
1. Import `MultiTrackTimeline` from `gui.timeline.multi_track_timeline`
2. Find where old `CapCutTimeline` is created in `__init__`
3. Replace with `MultiTrackTimeline` — keep same parent layout position
4. Connect existing audio/video file lists to new timeline's track model
5. Wire playback controls (play, pause, seek) to new timeline
6. Wire export button to new multi-track export pipeline
7. Keep old timeline available as fallback via try/except

**Acceptance Criteria:**
- App opens showing multi-track timeline (Video, Audio, Text, Effects layers)
- Existing audio files appear on audio track
- Existing video files appear on video track
- Playback controls work
- Export still produces video output

---

### Story I-10: Wire Clip Drag-Drop, Trim, Split (P1)

**File to edit:** `gui/main.py` + `gui/timeline/multi_track_timeline.py`

**What to do:**
1. Import `ClipDragHandler` from `gui.timeline.clip_drag`
2. Import `ClipTrimSplitHandler` from `gui.timeline.clip_trim`
3. Attach drag handler to multi-track timeline widget
4. Attach trim/split handler to multi-track timeline widget
5. Add razor tool button to timeline toolbar
6. Connect undo_manager from Story I-1 to all clip operations

**Acceptance Criteria:**
- Drag clips between tracks
- Trim clip edges by dragging
- Split clip with razor tool
- All operations are undoable (Ctrl+Z)

---

### Story I-11: Wire Keyframe Editor (P2)

**File to edit:** `gui/main.py` + `gui/timeline/multi_track_timeline.py`

**What to do:**
1. Import `KeyframeEditorPanel` from `gui.timeline.keyframe_editor`
2. Add keyframe panel below timeline (collapsible)
3. When clip selected, show its keyframes in editor
4. Double-click on timeline to add keyframe
5. Drag keyframe diamond to adjust time/value
6. Wire interpolation type selector (linear, ease-in, ease-out, bezier)

**Acceptance Criteria:**
- Keyframe panel visible below timeline
- Add/remove/drag keyframes
- Diamond markers on clips
- Property selector (opacity, volume, position, scale)

---

### Story I-12: Wire Text Overlay Tool (P2)

**File to edit:** `gui/main.py`

**What to do:**
1. Import `TextOverlay, TextAnimPreset` from `gui.timeline.text_layer`
2. Add "Text" tool button in timeline toolbar (T icon)
3. On click, create new text clip on Text track
4. Show text properties panel: font, size, color, alignment, animation preset
5. Wire text rendering into export pipeline (FFmpeg drawtext filter)
6. Show text preview on video preview area

**Acceptance Criteria:**
- "T" button adds text clip to timeline
- Edit text content, font, size, color
- Select animation preset (fade in, typewriter, etc.)
- Text visible on video preview
- Text exported in final video

---

### Story I-13: Wire Transitions Panel (P2)

**File to edit:** `gui/main.py`

**What to do:**
1. Import `TransitionLibrary, TransitionType` from `gui.models.transitions`
2. Add transitions panel (sidebar or popup) showing available transitions
3. Drag transition between two clips on timeline to apply
4. Show transition duration handle on timeline
5. Wire FFmpeg xfade filter generation for export
6. Show transition preview on hover

**Acceptance Criteria:**
- Transitions panel shows 12+ transition types with icons
- Drag transition between clips
- Transition duration adjustable
- Transitions render in export

---

### Story I-14: Wire Effects Panel (P2)

**File to edit:** `gui/main.py`

**What to do:**
1. Import `EffectsLibrary, EffectType` from `gui.models.effects`
2. Add effects panel (sidebar or popup)
3. Drag effect onto clip to apply
4. Show effect parameters when clip selected
5. Wire parameters to keyframe system (from Story I-11)
6. Wire FFmpeg filter generation for export

**Acceptance Criteria:**
- Effects panel shows 10+ effect types
- Apply effect to clip by drag or button
- Adjust effect parameters
- Effects visible in preview and export

---

### Story I-15: Wire Speed Ramp (P2)

**File to edit:** `gui/main.py` + `gui/timeline/multi_track_timeline.py`

**What to do:**
1. Import `SpeedRampEditor` from `gui.timeline.speed_ramp`
2. Add speed ramp editor panel (opens when clip right-click → "Speed")
3. Show bezier curve editor for speed profile
4. Add preset buttons (ease-in, flash, montage)
5. Wire to FFmpeg setpts filter for export
6. Update clip visual length on timeline when speed changes

**Acceptance Criteria:**
- Right-click clip → "Speed" opens editor
- Bezier curve visible and draggable
- Presets apply instantly
- Export uses correct speed curve

---

### Story I-16: Wire Export Presets (P2)

**File to edit:** `gui/main.py`

**What to do:**
1. Import `ExportPresetManager, ExportPreset` from `gui.models.export_presets`
2. Replace existing export dialog with preset-aware version
3. Add platform preset selector (YouTube 4K, Instagram Reels, TikTok, etc.)
4. Show resolution, codec, bitrate auto-populated from preset
5. Allow custom overrides
6. Wire to export pipeline

**Acceptance Criteria:**
- Export dialog shows platform presets
- Selecting preset auto-fills settings
- Custom overrides work
- Export produces correct format/resolution

---

### Story I-17: Wire Real-time Monitor (P2)

**File to edit:** `modules/master/ui_panel.py`

**What to do:**
1. Import `RealtimeMonitor` from `modules.master.realtime_monitor`
2. Replace offline-render-then-play with realtime monitor
3. Add "Monitor" toggle button in transport bar
4. When enabled, audio plays through mastering chain in real-time
5. Meter data streams to WLM Plus meter at 30fps
6. Fallback to offline render if Rust/cpal not available

**Acceptance Criteria:**
- Monitor toggle visible
- When ON, hear audio through mastering chain live
- Meters update in real-time
- When OFF, fallback to old offline method

---

### Story I-18: Wire GPU Preview (P3)

**File to edit:** `gui/main.py` + `gui/video/preview.py`

**What to do:**
1. Import `GPUPreviewCompositor, FrameCache` from `gui.video.gpu_preview`
2. Replace OpenCV frame-by-frame preview with GPU compositor
3. Use FrameCache for decoded frames
4. Composite video + text + effects layers at 30fps
5. Fallback to OpenCV if GPU not available
6. Add proxy file support toggle in settings

**Acceptance Criteria:**
- Video preview renders via GPU pipeline when available
- Scrubbing is smooth (30fps target)
- Multi-layer compositing (video + text overlay)
- Fallback to OpenCV works

---

### Story I-19: Final Integration Test & Polish (P0)

**File to edit:** Multiple — fix any issues found

**What to do:**
1. Launch app and test EVERY new feature end-to-end
2. Fix any crashes, visual glitches, or broken connections
3. Verify all keyboard shortcuts work
4. Verify all menu items work
5. Run full test suite — must be 261+ tests passing
6. Test with real audio/video files:
   - Load 5 audio files → AI DJ order → Master → Export
   - Load video + audio → Add text overlay → Add transition → Export
   - Master with spectrum analyzer visible → Export loudness report

**Acceptance Criteria:**
- Zero crashes on any feature path
- All 19 stories visually confirmed working
- Full test suite passes
- Exported video plays correctly in VLC
- Exported audio passes loudness spec (True Peak ≤ -1.0 dBTP)

---

## EXECUTION ORDER

```
CRITICAL PATH (do first, in order):
I-1  → Undo/Redo (everything needs this)
I-2  → AutoSave (safety net)
I-3  → Vintage Theme (visual foundation)
I-9  → Multi-Track Timeline (biggest change)
I-10 → Clip Drag/Trim/Split (timeline usability)

MASTERING PANEL (can parallel with timeline):
I-4  → Spectrum Analyzer
I-5  → WLM Plus Meter
I-6  → Match EQ
I-7  → A/B Compare
I-8  → Loudness Report
I-17 → Real-time Monitor

CAPCUT FEATURES (after timeline works):
I-11 → Keyframes
I-12 → Text Overlay
I-13 → Transitions
I-14 → Effects
I-15 → Speed Ramp
I-16 → Export Presets

FINAL:
I-18 → GPU Preview
I-19 → Integration Test
```

---

## VALIDATION AFTER EVERY STORY

```bash
# MUST pass after every single story:
python3 -c "from gui import LongPlayStudioV4; print('OK')"
python3 -m pytest tests/ -v --tb=short
# App must launch:
python3 gui.py  # (visually verify new feature appears)
```

---

## WHAT "DONE" LOOKS LIKE

When all 19 stories are complete, a user opening the app will see:

1. **Dark vintage theme** (Waves/SSL style) applied everywhere
2. **Multi-track timeline** at bottom with Video/Audio/Text/Effects tracks
3. **Toolbar** with Text tool, Razor tool, Undo/Redo buttons
4. **Transitions panel** with 12+ types
5. **Effects panel** with 10+ types  
6. **Mastering panel** with spectrum analyzer overlay on EQ, WLM Plus meter, Match EQ tab, A/B toggle, loudness report export
7. **Real-time monitoring** of audio through mastering chain
8. **Speed ramp editor** on right-click
9. **Export dialog** with platform presets
10. **Auto-save** running in background
11. **Ctrl+Z** undoing any action
12. **GPU preview** rendering smooth composited output

Every feature is actually accessible, actually functional, actually visible.
