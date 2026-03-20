# LongPlay Studio V5.10.0 — Guardian Audit Report

## Date: 2026-03-18
## Auditor: Claude Code (Opus 4.6)
## Scope: Maximizer Gain + RT Audio Stuttering

---

## Executive Summary

Two critical bugs were identified in the real-time mastering pipeline:
1. **Maximizer Gain Knob disconnected from RT Engine** — OzoneRotaryKnob handler does not forward gain to `PyRtEngine`
2. **Buffer Boundary Stuttering** — cpal audio callback processes only 512 samples but may be asked for 1024+, causing sample repetition at ~86 Hz

Both bugs have been fixed. The stuttering fix required restructuring the audio callback to process multiple DSP blocks per cpal callback.

---

## Bug #1: Maximizer Gain Knob Not Affecting Audio

### Symptom
Turning the Maximizer gain rotary knob updates the UI display value but audio volume does not change during real-time playback.

### Root Cause
**File:** `modules/master/ui_panel.py:5027-5035`

Two gain handlers exist:
- `_on_gain_knob_changed()` (line 5027) — connected to `OzoneRotaryKnob`, **does NOT forward to RT engine**
- `_on_gain_changed()` (line 5827) — the old int-based slider handler, **correctly forwards** to `self._rt_engine.set_gain()`

The V5.8 refactor introduced `OzoneRotaryKnob` widgets that connect to `_on_gain_knob_changed()`, but this handler was never updated to include the V5.10 RT engine forwarding that exists in `_on_gain_changed()`.

### Signal Flow (Before Fix)
```
OzoneRotaryKnob.valueChanged
  → _on_gain_knob_changed()
    → chain.maximizer.set_gain()      ✅ (offline render only)
    → QAudioOutput.setVolume()         ⚠️ (fake volume hack, not real DSP)
    → _rt_engine.set_gain()            ❌ MISSING
```

### Signal Flow (After Fix)
```
OzoneRotaryKnob.valueChanged
  → _on_gain_knob_changed()
    → chain.maximizer.set_gain()       ✅ (offline render)
    → _rt_engine.set_gain()            ✅ (real-time DSP via lock-free atomic)
    → color feedback + auto-measure    ✅ (visual parity with _on_gain_changed)
```

### Fix Applied
- `ui_panel.py:5027` — Added `self._rt_engine.set_gain(gain_db)` with null check
- `ui_panel.py:5056` — Added `self._rt_engine.set_ceiling(value)` to ceiling knob handler
- Added gain color feedback (teal → yellow → orange → red) matching `_on_gain_changed()`
- Added `_schedule_auto_measure()` and `_schedule_auto_preview()` calls

### Also Checked
- `gui.py` (standalone monolith) — already correctly forwards at line 8115. No fix needed.
- Rust parameter flow (`params.rs` → `stream.rs:337`) — verified correct.

---

## Bug #2: Audio Stuttering at Buffer Boundaries

### Symptom
Audio plays with choppy/stuttering artifacts. FFT analysis shows a strong peak at ~86 Hz (= 44100 Hz / 512 samples), indicating discontinuities at every buffer boundary.

### Root Cause
**File:** `rust/crates/longplay-rt/src/stream.rs:170-231`

The audio callback processes audio in blocks of `BLOCK_SIZE = 512` samples. However, the cpal audio backend may request **more than 512 frames** per callback (e.g., 1024 or 2048 depending on the platform/driver).

**The old code:**
```rust
// Line 171: Only process 512 samples max
let block_frames = num_output_frames.min(BLOCK_SIZE);

// Line 206-207: For frames beyond 512, CLAMP to last valid index
for i in 0..num_output_frames {
    let src_i = i.min(block_frames.saturating_sub(1)); // BUG!
    data[base] = buffer[0][src_i] * volume;
}

// Line 234: Position advances by FULL num_output_frames
let advance = (num_output_frames as f64 * src_ratio) as u64;
```

**What happened:**
- cpal requests 1024 frames
- DSP processes only 512 frames
- Output frames 512-1023: `src_i` clamped to 511 → **same sample repeated 513 times**
- Position advances by 1024 frames → **audio skips 512 frames ahead**
- Net effect: half the audio is a repeated sample, half is skipped = **86 Hz stutter**

### Fix Applied
Restructured the audio callback to use a **multi-block processing loop**:

```rust
while frames_written < num_output_frames {
    let block_frames = remaining.min(BLOCK_SIZE);
    // Read block_frames samples from source
    // Process through full DSP chain (EQ → Imager → Maximizer → Limiter)
    // Write to output at correct offset
    // Advance position by block_frames (not total)
    frames_written += block_frames;
}
```

Key changes:
1. **Loop processes multiple 512-sample blocks** until the entire cpal buffer is filled
2. **Position advances per-block** (not per-callback), ensuring no audio is skipped
3. **Pre-allocated block buffers** outside the loop to minimize heap allocation in the audio thread
4. **End-of-file handling** fills remaining output with silence instead of repeating

### Verification
- DSP state (EQ IIR filters, IRC limiter envelopes) is maintained continuously across blocks within the same callback — no discontinuities
- Meter data accumulates across all blocks in the callback — accurate peak/RMS measurement
- Position update is atomic — single store after all blocks processed

---

## Files Modified

| File | Change | Lines |
|------|--------|-------|
| `modules/master/ui_panel.py` | Added RT engine forwarding to `_on_gain_knob_changed()` | 5027-5054 |
| `modules/master/ui_panel.py` | Added RT engine forwarding to `_on_ceiling_knob_changed()` | 5056-5061 |
| `rust/crates/longplay-rt/src/stream.rs` | Multi-block processing loop replacing single-block | 154-264 |

## Files Verified (No Changes Needed)

| File | Status |
|------|--------|
| `gui.py` | Already correctly forwards gain to RT engine (line 8115) |
| `rust/crates/longplay-rt/src/params.rs` | Lock-free atomics working correctly |
| `rust/crates/longplay-rt/src/lib.rs` | `set_gain()` clamp + dirty flag correct |
| `rust/crates/longplay-python/src/rt_engine.rs` | PyO3 bindings correct |

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Multi-block loop increases callback duration | Max 2-4 blocks per callback (typical). DSP is O(n) — scales linearly. |
| Vec allocation per block (`to_vec()` for buffer) | Block buffers pre-allocated. Only the AudioBuffer slices allocate. Future: use fixed-size arrays. |
| IRC limiter state across blocks | Limiter maintains internal buffers — state is continuous. No discontinuity. |
| Meter accuracy with multiple blocks | Accumulators span all blocks — peak/RMS correct for full callback. |
