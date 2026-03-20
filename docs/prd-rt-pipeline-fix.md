# LongPlay Studio V5.10.0 — PRD: RT Mastering Pipeline Fix

## Problem Statement

The real-time mastering preview feature (V5.10) has two defects:
1. The Maximizer gain rotary knob does not affect real-time audio playback
2. Audio stutters/chops at ~86 Hz intervals during real-time playback

Users cannot use the mastering preview for critical listening — the core value proposition of V5.10.

---

## Functional Requirements

### FR-1: Gain Knob → RT Engine Parameter Flow
The OzoneRotaryKnob for Maximizer Gain MUST forward gain_db to the RT engine via `_rt_engine.set_gain(gain_db)` when the RT engine is available.

### FR-2: Ceiling Knob → RT Engine Parameter Flow
The OzoneRotaryKnob for Maximizer Ceiling MUST forward ceiling_db to the RT engine via `_rt_engine.set_ceiling(value)` when the RT engine is available.

### FR-3: Fallback to QAudioOutput Volume
When `_rt_engine is None` (Rust backend unavailable), the gain knob MUST fall back to adjusting QAudioOutput volume as an approximation.

### FR-4: Full Buffer Processing
The cpal audio callback MUST process enough DSP blocks to fill the entire requested output buffer. No sample repetition or skipping is acceptable.

### FR-5: Continuous DSP State
DSP module state (EQ IIR filters, IRC limiter envelopes, imager) MUST be maintained continuously across block boundaries within a single callback and across callbacks.

### FR-6: Accurate Metering
Peak and RMS meters MUST reflect the actual output audio across all processed blocks, not just the first block.

### FR-7: End-of-File Handling
When playback reaches the end of the audio file mid-callback, remaining output frames MUST be filled with silence (0.0), not repeated samples.

---

## User Stories

### US-1: Mastering Engineer Adjusts Gain
**As a** mastering engineer,
**I want** to turn the Maximizer gain knob and immediately hear the audio get louder,
**so that** I can find the optimal gain push for my master in real-time.

**Acceptance:** Turning the gain knob from 0 to +10 dB produces an audible, immediate increase in loudness through the full DSP chain (not just volume scaling).

### US-2: Stutter-Free Playback
**As a** user previewing a master,
**I want** clean, continuous audio playback without clicks, stutters, or artifacts,
**so that** I can make accurate critical listening decisions.

**Acceptance:** Audio plays back cleanly at all sample rates (44.1k, 48k, 96k) with no audible artifacts, regardless of the system's cpal buffer size.

### US-3: Ceiling Knob Real-Time
**As a** mastering engineer,
**I want** to adjust the ceiling and hear the result instantly,
**so that** I can fine-tune the output level without re-rendering.

---

## Acceptance Tests

### AT-1: Gain Knob Null Test
1. Load a test tone (1 kHz sine, -12 dBFS)
2. Set Maximizer gain to 0 dB, play
3. Measure output level → should be ~-12 dBFS
4. Set gain to +6 dB while playing
5. Measure output level → should be ~-6 dBFS (±1 dB for IRC limiting)

### AT-2: No Stutter Test
1. Load any audio file (24 seconds+)
2. Play with default settings
3. Record output via loopback
4. FFT analysis: no spectral peak at buffer-boundary frequencies (86 Hz for 512@44.1k)
5. Zoomed waveform: no discontinuities at block boundaries

### AT-3: Large Buffer Test
1. Configure system audio to use large buffer (2048+ samples)
2. Play audio through RT engine
3. Verify clean playback (multiple DSP blocks per callback)

### AT-4: End-of-File Clean Stop
1. Play audio to completion
2. Verify no click/pop at end of file
3. Verify playback stops and position resets to 0

### AT-5: Meter Accuracy
1. Play a -6 dBFS test tone with gain = 0
2. Verify peak meters read ~-6 dBFS
3. Set gain to +6 dB
4. Verify peak meters read ~0 dBFS (or show gain reduction from limiter)
