# PRD Phase 4 — FINAL COMPLETION: Real-time Wiring + End-to-End + Polish

**Version:** 4.0  
**Status:** FINAL PHASE — Make everything actually work with real data in real-time  
**Goal:** Every widget shows live data, every export produces real output, every feature is testable end-to-end

---

## THE PROBLEM

Phase 1 created modules. Phase 2 wired them into GUI. Phase 3 added Ozone widgets.  
But widgets are **empty shells** — they don't receive real audio data during processing.

- LogicLevelMeter: bars don't move (no data feed)  
- Vectorscope: empty circle (no audio samples)  
- Spectrum Analyzer: blank (no FFT feed)  
- Transfer Curve: static (doesn't update when user changes Dynamics params)  
- Production Pipeline: just a message box, not a real wizard  
- Video export with text/transitions: untested with real FFmpeg  
- Hook Extractor: no before/after master option  

---

## CRITICAL RULES FOR CLAUDE CODE

```
1. EVERY story must end with a REAL TEST using actual audio/video data
2. "It imports OK" is NOT enough — the widget must SHOW DATA
3. After each story, run: python3 gui.py and VISUALLY VERIFY the feature
4. Use the test WAV: ffmpeg -f lavfi -i "sine=frequency=440:duration=10" -ar 44100 -ac 2 /tmp/test.wav
5. Do not stop until verify_longplay.sh passes ALL checks
```

---

## STORIES

### Story F-1: Wire Level Meters to Chain Processing (P0)

**Problem:** LogicLevelMeter L/R bars never move — chain._send_meter() doesn't update them.

**What to do in `modules/master/ui_panel.py`:**
1. In `MasterPanel.__init__`, store reference to MetersPanel's logic meters
2. In the meter callback (`update_live_levels` or `_send_meter` callback), extract `left_peak_db` and `right_peak_db`
3. Call `self.meters.logic_before.set_levels(left_db, right_db)` for input levels
4. Call `self.meters.logic_after.set_levels(left_db, right_db)` for output levels
5. During render/preview, update at each chain stage

**Verification:** Load a WAV file → click Preview → L/R green bars move up and down in real-time.

---

### Story F-2: Wire Spectrum Analyzer to Audio Data (P0)

**Problem:** Spectrum Analyzer widget is blank — no audio samples fed to FFT engine.

**What to do in `modules/master/ui_panel.py`:**
1. In the chain's meter callback, get the raw audio samples (last 4096 samples)
2. Feed to spectrum analyzer: `self._spectrum_analyzer.feed_samples(samples)`
3. Start a 30fps QTimer that calls `self._spectrum_analyzer.update()` during playback
4. Stop timer when playback stops
5. Wire Pre/Post toggle: Pre = feed samples before EQ, Post = feed after full chain

**Verification:** Load WAV → Preview → see teal FFT spectrum moving on EQ panel.

---

### Story F-3: Wire Vectorscope to Audio Playback (P0)

**Problem:** Vectorscope shows empty half-circle — no stereo samples fed.

**What to do in `modules/master/ui_panel.py`:**
1. In chain meter callback, extract L/R channel samples
2. Feed to vectorscope: `self._vectorscope.feed_samples(left_channel, right_channel)`
3. Vectorscope auto-updates via its internal repaint
4. During idle, show last known state (fade out)

**Verification:** Load stereo WAV → Preview → vectorscope shows dot pattern indicating stereo width.

---

### Story F-4: Wire Transfer Curve to Dynamics Parameters (P0)

**Problem:** Transfer curve is static — doesn't update when user changes threshold/ratio/knee.

**What to do in `modules/master/ui_panel.py`:**
1. In `_build_dynamics_view`, after each dynamics parameter change (threshold slider, ratio slider, knee slider), call:
   ```python
   self._transfer_curve.set_params(
       threshold=self.chain.dynamics.single_band.threshold,
       ratio=self.chain.dynamics.single_band.ratio,
       knee=self.chain.dynamics.single_band.knee,
       makeup=self.chain.dynamics.single_band.makeup,
   )
   ```
2. Also update when preset is loaded
3. Also update when AI recommend applies dynamics settings

**Verification:** Change Dynamics threshold slider → transfer curve visually updates the knee point.

---

### Story F-5: Wire Rotary Knobs Bidirectional (P0)

**Problem:** Ozone rotary knobs set values TO chain, but don't update FROM chain (e.g., when AI Recommend sets gain).

**What to do in `modules/master/ui_panel.py`:**
1. After `apply_recommendation()`, update all Ozone knobs:
   ```python
   if hasattr(self, '_oz_gain_knob'):
       self._oz_gain_knob.setValue(self.chain.maximizer.gain_db)
   if hasattr(self, '_oz_char_knob'):
       self._oz_char_knob.setValue(self.chain.maximizer.character)
   ```
2. After `load_settings()`, same update
3. After preset change, same update

**Verification:** Click AI Master → Ozone GAIN knob auto-rotates to recommended value.

---

### Story F-6: Production Pipeline Wizard (P1)

**Problem:** Production Pipeline is just a message box showing text steps.

**What to do in `gui/main.py`:**
1. Create `ProductionPipelineDialog(QDialog)` — a step-by-step wizard
2. Steps: Import → AI DJ → Compile → Master → Hook Extract → Video → Export
3. Each step has: description, action button, status indicator (pending/running/done)
4. "Hook Extract" step has radio: "○ From Original  ○ From Mastered"
5. "Video" step has option: "○ Standard Video  ○ Lip-sync Video"
6. Each action button calls the existing corresponding method
7. Progress bar per step
8. Add "Pipeline" button to main toolbar to open this dialog

**Verification:** Click Pipeline → step through Import → AI DJ → Master → Export → get output file.

---

### Story F-7: Video Export End-to-End with Text + Transitions (P1)

**Problem:** Video export with text overlay and transitions never tested with real FFmpeg.

**What to do:**
1. In `gui/video/multi_track_export.py`, ensure text clips generate FFmpeg drawtext filter
2. Ensure transitions generate FFmpeg xfade filter between video clips
3. Test with real files:
   - Create 2 short test videos
   - Add text overlay clip
   - Add crossfade transition
   - Export → verify output has text visible and transition smooth

**Test script:**
```bash
# Generate 2 test videos
ffmpeg -y -f lavfi -i "color=c=red:s=1280x720:d=3" -c:v libx264 /tmp/v1.mp4
ffmpeg -y -f lavfi -i "color=c=blue:s=1280x720:d=3" -c:v libx264 /tmp/v2.mp4
# Export should produce video with red→crossfade→blue + text overlay
```

**Verification:** Open exported MP4 in VLC → see text + transition.

---

### Story F-8: Hook Extractor Before/After Master Option (P1)

**Problem:** Hook Extractor always extracts from original — no option to extract from mastered version.

**What to do in `gui/dialogs/hook_extractor.py` or the hook extractor dialog in `gui/main.py`:**
1. Add radio buttons: "Extract from: ○ Original  ○ Mastered"
2. If "Mastered" selected, use the mastered output file path instead of original
3. If mastered file doesn't exist yet, show warning "Please master first"

**Verification:** Master a file → open Hook Extractor → select "From Mastered" → extract hooks → hooks have mastered sound.

---

### Story F-9: WLM Plus Meter Real-time Feed + Visual Polish (P1)

**Problem:** WLM Plus meter exists but doesn't receive real LUFS data during processing, and colors need to match Waves more closely.

**What to do in `modules/master/ui_panel.py`:**
1. In chain meter callback, feed LUFS data to WLM meter:
   ```python
   if hasattr(self, '_wlm_plus') and self._wlm_plus:
       self._wlm_plus.update(
           momentary=levels.get('lufs_momentary', -70),
           short_term=levels.get('lufs_short_term', -70),
           integrated=levels.get('lufs_integrated', -70),
           true_peak_l=levels.get('left_peak_db', -60),
           true_peak_r=levels.get('right_peak_db', -60),
           lra=levels.get('lra', 0),
       )
   ```
2. Polish colors to match Waves WLM Plus reference:
   - Background: dark teal gradient (#0a1a2a → #0d2233)
   - Meter bars: green (#00cc66) → yellow (#ffcc00) → red (#ff3333)
   - Text: white LED-segment style (Courier New or 7-segment font)
   - Numeric displays: large, high-contrast

**Verification:** Preview audio → WLM meter shows M/S/I values updating, bars moving, True Peak L/R bouncing.

---

### Story F-10: Match EQ Spectrum Overlay Visualization (P2)

**Problem:** Match EQ has load/apply but no visual showing reference vs current spectrum.

**What to do in `modules/master/ui_panel.py`:**
1. When reference loaded, get `self._match_eq.reference_spectrum`
2. When current analyzed, get `self._match_eq.current_spectrum`
3. Draw both spectrums on the spectrum analyzer widget as overlays:
   - Reference: orange dashed line
   - Current: teal solid line
   - Difference: white area fill
4. Show correction curve when "Apply" is clicked

**Verification:** Load reference → load current → see two spectrum lines overlaid on EQ → click Apply → see correction.

---

### Story F-11: Real-time Audio Monitor via Rust/cpal (P2)

**Problem:** Real-time monitor toggle exists but actual audio goes through offline render → playback.

**What to do:**
1. In `modules/master/realtime_monitor.py`, implement the actual audio routing:
   - If Rust backend available: use longplay Python bindings to stream audio
   - If not: use `sounddevice` or `pyaudio` library for Python-native real-time
   - Process audio through chain in 512-sample blocks
   - Output to system audio device
2. Feed meter data from processing thread to UI at 30fps
3. Bypass mode: switch between chain output and direct passthrough

**Fallback:** If neither Rust nor sounddevice available, keep offline render method.

**Verification:** Toggle Monitor ON → hear mastered audio in real-time → change EQ → hear change immediately.

---

### Story F-12: GPU Preview Enhancement (P2)

**Problem:** GPU preview framework exists but uses OpenCV fallback, not actual GPU compositing.

**What to do:**
1. In `gui/video/gpu_preview.py`, implement Metal rendering for macOS:
   - Use `ctypes` to access Metal framework, OR
   - Use `moderngl` (pip installable OpenGL wrapper) as cross-platform alternative
   - Composite video frame + text overlay + effects in GPU
2. Frame cache: decode ahead by 30 frames using FFmpeg pipe
3. Scrubbing: serve from cache for instant response
4. Fallback: if no GPU context, use OpenCV (current behavior)

**Verification:** Load video → scrub timeline → preview updates smoothly at 30fps.

---

### Story F-13: Final End-to-End Integration Test (P0)

**What to do:**
1. Run the `verify_longplay.sh` script — ALL checks must pass
2. Manual test on Mac:
   - Load 3 audio files → AI DJ → Compile → Master (with Ozone panel visible)
   - During mastering: verify Level Meters move, Spectrum Analyzer shows FFT, Vectorscope shows stereo field, Transfer Curve matches Dynamics settings
   - Export loudness report → verify CSV/PDF content
   - Load video + audio → add text → add transition → export
   - Open in VLC → verify text and transition visible
   - Hook extract from mastered → verify hooks have mastered sound
3. Fix ANY issue found
4. All tests pass (261+)
5. Push to main

**Verification:** User can go from zero to finished YouTube video in one session using only this app.

---

## EXECUTION ORDER

```
CRITICAL (widgets show real data):
F-1  → Level Meters (bars move)
F-2  → Spectrum Analyzer (FFT visible)
F-3  → Vectorscope (stereo dots)
F-4  → Transfer Curve (responds to params)
F-5  → Rotary Knobs (bidirectional)

WORKFLOW (end-to-end features):
F-6  → Production Pipeline wizard
F-7  → Video export with text + transitions
F-8  → Hook Extractor before/after
F-9  → WLM Plus real-time + polish

ADVANCED:
F-10 → Match EQ spectrum overlay
F-11 → Real-time audio monitor
F-12 → GPU preview

FINAL:
F-13 → End-to-end integration test
```

---

## WHAT "100% DONE" MEANS

When F-13 passes, a user can:

1. Open the app → see dark vintage theme
2. Load audio files → AI DJ orders them
3. Compile playlist with crossfade
4. Open Master panel → see Ozone 12 style interface
5. Click AI Master → Ozone knobs auto-rotate to recommended values
6. See spectrum analyzer showing real FFT on EQ
7. See vectorscope showing stereo field on Imager
8. See transfer curve updating on Dynamics
9. See L/R level meters bouncing green→yellow→red
10. See WLM Plus meter with M/S/I LUFS updating
11. Click A/B to compare original vs mastered
12. Export loudness report (CSV/PDF)
13. Extract hooks (from original or mastered)
14. Add video + text overlay + transitions
15. Export final video with all effects baked in
16. Ctrl+Z undo any action
17. Auto-save protects against crash
18. Production Pipeline wizard guides the whole flow

That is 100%.
