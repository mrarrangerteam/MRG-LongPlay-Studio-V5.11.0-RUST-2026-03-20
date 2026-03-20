# LongPlay Studio V5.10.0 — Technical Specification: RT Pipeline Fix

## 1. Architecture Overview

### Signal Flow (Real-Time)
```
Python GUI Thread                    Rust Audio Thread (cpal)
─────────────────                    ───────────────────────
OzoneRotaryKnob.valueChanged
  → _on_gain_knob_changed()
    → chain.maximizer.set_gain()     (offline chain, for export)
    → _rt_engine.set_gain(gain_db)
      → PyRtEngine.set_gain()        (PyO3 binding)
        → RtEngine.set_gain()
          → params.gain_db.store()   ──AtomicF32──→  params.gain_db.load()
          → params.mark_dirty()      ──AtomicBool─→  params.take_dirty()
                                                       → apply_params_to_dsp()
                                                         → maximizer.set_gain_db()
                                                       → maximizer.process(block)
                                                         → apply_input_gain()
                                                           → sample *= db_to_linear(gain_db)
                                                       → cpal output buffer
```

### DSP Chain Order (Per Block)
```
Source Audio (memory)
  → EQ (8-band parametric, IIR biquad filters)
  → Imager (stereo width, optional multiband)
  → Maximizer
    ├─ Tone Pre-EQ (if enabled)
    ├─ Upward Compression (low-level boost)
    ├─ Transient Emphasis (high/mid/low bands)
    ├─ Soft Clipping
    ├─ Input Gain Push (gain_db applied HERE)
    └─ IRC Limiter (6 algorithms: IRC 1-5, LL)
  → LookAhead Limiter (true peak brickwall, if enabled)
  → Master Volume (* linear)
  → cpal output (interleaved stereo)
```

---

## 2. Files Modified

### 2.1 `modules/master/ui_panel.py`

#### Fix: `_on_gain_knob_changed()` (line 5027)

**Before:**
```python
def _on_gain_knob_changed(self, gain_db: float):
    self.chain.maximizer.set_gain(gain_db)
    self.max_gain_display.setText(f"+{gain_db:.1f}")
    volume = min(3.0, 0.5 + gain_db / 20.0 * 2.5)
    if hasattr(self, '_audio_output_master'):
        self._audio_output_master.setVolume(volume)
```

**After:**
```python
def _on_gain_knob_changed(self, gain_db: float):
    self.chain.maximizer.set_gain(gain_db)
    self.max_gain_display.setText(f"+{gain_db:.1f}")
    if self._rt_engine is not None:
        self._rt_engine.set_gain(gain_db)
    else:
        volume = min(3.0, 0.5 + gain_db / 20.0 * 2.5)
        if hasattr(self, '_audio_output_master'):
            self._audio_output_master.setVolume(volume)
        if hasattr(self, '_audio_output_bypass'):
            self._audio_output_bypass.setVolume(volume)
    # + color feedback + auto-measure + auto-preview
```

**Why:** The OzoneRotaryKnob (V5.8 A-3) was connected to a handler that predated the RT engine (V5.10). The old handler used QAudioOutput volume as a fake approximation. The new code forwards the real gain_db to the RT engine for proper DSP processing.

#### Fix: `_on_ceiling_knob_changed()` (line 5056)

Added `self._rt_engine.set_ceiling(value)` — same pattern as gain.

### 2.2 `rust/crates/longplay-rt/src/stream.rs`

#### Fix: Multi-Block Processing Loop (lines 154-264)

**Before (single block):**
```rust
let block_frames = num_output_frames.min(BLOCK_SIZE);  // Max 512
// ... process block_frames ...
for i in 0..num_output_frames {
    let src_i = i.min(block_frames - 1);  // CLAMP → repeat last sample!
    data[base] = buffer[0][src_i] * volume;
}
let advance = (num_output_frames as f64 * src_ratio) as u64;  // Skip ahead
```

**After (multi-block loop):**
```rust
let mut block_l = vec![0.0f32; BLOCK_SIZE];  // Pre-alloc outside loop
let mut block_r = vec![0.0f32; BLOCK_SIZE];

while frames_written < num_output_frames {
    let block_frames = remaining.min(BLOCK_SIZE);
    // Zero + fill block from source
    // Process: EQ → Imager → Maximizer → Limiter
    // Write to output at correct offset
    // Advance position by block_frames (not total!)
    current_pos += advance;
    frames_written += block_frames;
    // Handle end-of-file: fill remaining with silence
}
position.store(current_pos, Ordering::Relaxed);
```

**Key differences:**
1. Processes N blocks (N = ceil(num_output_frames / 512)) instead of 1
2. Position advances incrementally per block, not by total output frames
3. No sample clamping/repetition — every output frame has unique DSP-processed audio
4. End-of-file fills silence instead of repeating

---

## 3. Anti-Patterns to Watch For

### 3.1 Knob Handler Missing RT Forwarding
**Pattern:** Any `_on_*_knob_changed()` that calls `self.chain.*.set_*(value)` but does NOT check `self._rt_engine`.

**Check all knob handlers:**
- `_on_gain_knob_changed` → ✅ Fixed
- `_on_ceiling_knob_changed` → ✅ Fixed
- `_on_character_knob_changed` → ⚠️ No RT param (character not in RtParams)
- `_on_upward_knob_changed` → ⚠️ No RT param
- `_on_softclip_knob_changed` → ⚠️ No RT param

### 3.2 Single-Block Processing
**Pattern:** `let block_frames = num_output_frames.min(BLOCK_SIZE)` followed by a single DSP pass.

This is only valid if `num_output_frames <= BLOCK_SIZE` is guaranteed, which it is NOT — cpal buffer sizes vary by platform (macOS CoreAudio often uses 512, but WASAPI/PulseAudio may use 1024-4096).

### 3.3 Output Clamping
**Pattern:** `let src_i = i.min(block_frames.saturating_sub(1))` when writing to output.

This repeats the last valid sample for any frame beyond the processed range. Always indicates a buffer underrun bug.

---

## 4. Diagnostic Checklist

Use this step-by-step to verify the fix:

### Step 1: Verify Parameter Flow
```python
# In Python console or test:
from longplay import PyRtEngine
engine = PyRtEngine()
engine.load_file("test.wav")
engine.play()

# Should hear audio get louder:
engine.set_gain(10.0)

# Should hear ceiling change:
engine.set_ceiling(-0.5)
```

### Step 2: Verify No Stuttering
```bash
# Record RT engine output via loopback and analyze:
python3 -c "
import numpy as np
import soundfile as sf

# Load recorded output
data, sr = sf.read('rt_output.wav')

# Check for buffer boundary artifacts
block_freq = sr / 512  # ~86 Hz for 44.1k
fft = np.abs(np.fft.rfft(data[:, 0]))
freqs = np.fft.rfftfreq(len(data), 1/sr)

# Find peak near block_freq
mask = (freqs > block_freq * 0.9) & (freqs < block_freq * 1.1)
artifact_level = 20 * np.log10(fft[mask].max() + 1e-10)
noise_floor = 20 * np.log10(np.median(fft) + 1e-10)
print(f'Artifact at {block_freq:.1f} Hz: {artifact_level:.1f} dB')
print(f'Noise floor: {noise_floor:.1f} dB')
print(f'Artifact above noise: {artifact_level - noise_floor:.1f} dB')
# Should be < 6 dB above noise floor (no artifact)
"
```

### Step 3: Verify Multi-Block Processing
```rust
// Add temporary debug logging in stream.rs:
// (Remove before production)
if frames_written == 0 {
    eprintln!("[longplay-rt] callback: {} output frames, {} blocks",
        num_output_frames, (num_output_frames + BLOCK_SIZE - 1) / BLOCK_SIZE);
}
```

### Step 4: Verify DSP State Continuity
- Play a sine sweep (20 Hz → 20 kHz) through the RT engine
- EQ with a narrow notch at 1 kHz
- The notch should be continuous with no clicks at block boundaries
- If clicks present: EQ IIR filter state is being reset per block

### Step 5: Null Test (Offline vs RT)
```python
# Process same audio through offline chain and RT engine
# Difference should be < -60 dBFS (only rounding differences)
import numpy as np

offline_output = chain.process(audio)  # chain.py offline render
rt_output = record_rt_output(audio)    # RT engine playback captured

diff = offline_output - rt_output
null_level = 20 * np.log10(np.max(np.abs(diff)) + 1e-10)
print(f"Null test: {null_level:.1f} dBFS")
# Target: < -60 dBFS
```

---

## 5. Fix Priority Order

| Priority | Fix | Risk | Effort |
|----------|-----|------|--------|
| **P0** | stream.rs multi-block loop | High — audio artifact | Done |
| **P0** | ui_panel.py gain knob RT forwarding | High — core feature broken | Done |
| **P1** | ui_panel.py ceiling knob RT forwarding | Medium — secondary parameter | Done |
| **P2** | Add gain smoothing (ramp) at block boundaries | Low — micro-clicks on fast knob turns | Future |
| **P2** | Replace Vec allocation with fixed-size array in callback | Low — performance optimization | Future |
| **P3** | Add character/upward/softclip to RtParams | Low — not yet in Rust DSP | Future |

---

## 6. Build & Test

### Rust Build
```bash
cd rust
cargo build --release
# Produces: target/release/liblongplay.dylib (macOS) / .so (Linux)
```

### Python Smoke Test
```bash
python3 -c 'from gui import LongPlayStudioV4; print("OK")'
```

### RT Engine Test
```bash
cd rust
cargo test -p longplay-rt
```

### Full Integration Test
```bash
python3 gui.py
# 1. Load an audio file
# 2. Open Mastering panel
# 3. Press Play (RT mode)
# 4. Turn Gain knob → audio should get louder
# 5. Listen for clean playback (no stuttering)
```
