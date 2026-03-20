//! LongPlay Real-Time Audio Engine
//!
//! Provides real-time audio playback with lock-free DSP parameter control.
//! Uses cpal for native audio output and processes DSP per-block (~11ms @ 48kHz).
//!
//! Signal flow:
//! ```text
//! File → Memory → [EQ → Imager → Maximizer → Limiter] → cpal output
//!                        ↑ atomic parameters (lock-free)
//!                        ↑ Python GUI knob changes = instant
//! ```

pub mod params;
pub mod stream;

use std::sync::Arc;
use std::sync::atomic::Ordering;

use crossbeam_channel::{bounded, Receiver};

use crate::params::RtParams;
use crate::stream::{AudioStream, MeterData, StreamState};

/// The main real-time audio engine.
///
/// Loads an audio file into memory, creates a cpal output stream,
/// and processes DSP in real-time with lock-free parameter control.
pub struct RtEngine {
    /// Shared atomic parameters (Python writes, audio thread reads).
    params: Arc<RtParams>,
    /// The active audio stream (if a file is loaded).
    stream: Option<AudioStream>,
    /// Meter data receiver (audio thread → Python, ~30Hz).
    meter_rx: Option<Receiver<MeterData>>,
    /// Sample rate of the currently loaded file.
    sample_rate: u32,
    /// Total frames of the currently loaded file.
    total_frames: u64,
    /// Latest meter snapshot (cached for Python polling).
    last_meter: MeterData,
}

impl RtEngine {
    pub fn new() -> Self {
        Self {
            params: Arc::new(RtParams::new()),
            stream: None,
            meter_rx: None,
            sample_rate: 0,
            total_frames: 0,
            last_meter: MeterData::default(),
        }
    }

    /// Load an audio file into memory and prepare for playback.
    pub fn load_file(&mut self, path: &str) -> Result<(), String> {
        // Stop any existing playback
        if let Some(ref s) = self.stream {
            s.stop();
        }
        self.stream = None;
        self.meter_rx = None;

        // Read audio file
        let (audio_data, info) = longplay_io::read_audio_any(path)
            .map_err(|e| format!("Failed to read audio: {}", e))?;

        if audio_data.is_empty() || audio_data[0].is_empty() {
            return Err("Audio file is empty".to_string());
        }

        self.sample_rate = info.sample_rate as u32;
        self.total_frames = audio_data[0].len() as u64;

        // Create meter channel (bounded, drop old if full)
        let (tx, rx) = bounded::<MeterData>(4);
        self.meter_rx = Some(rx);

        // Reset params
        self.params = Arc::new(RtParams::new());

        // Build audio stream
        let state = StreamState {
            audio_data,
            sample_rate: self.sample_rate,
            total_frames: self.total_frames,
        };

        let stream = AudioStream::new(state, self.params.clone(), tx)?;
        self.stream = Some(stream);

        Ok(())
    }

    /// Start playback.
    pub fn play(&self) {
        if let Some(ref s) = self.stream {
            s.play();
        }
    }

    /// Pause playback (maintains position).
    pub fn pause(&self) {
        if let Some(ref s) = self.stream {
            s.pause();
        }
    }

    /// Stop playback (resets position to 0).
    pub fn stop(&self) {
        if let Some(ref s) = self.stream {
            s.stop();
        }
    }

    /// Seek to a position in milliseconds.
    pub fn seek(&self, position_ms: u64) {
        if let Some(ref s) = self.stream {
            let frame = (position_ms as f64 * self.sample_rate as f64 / 1000.0) as u64;
            s.seek(frame.min(self.total_frames.saturating_sub(1)));
        }
    }

    /// Get current playback position in milliseconds.
    pub fn get_position_ms(&self) -> u64 {
        if let Some(ref s) = self.stream {
            let frames = s.current_position();
            if self.sample_rate > 0 {
                (frames as f64 * 1000.0 / self.sample_rate as f64) as u64
            } else {
                0
            }
        } else {
            0
        }
    }

    /// Get total duration in milliseconds.
    pub fn get_duration_ms(&self) -> u64 {
        if self.sample_rate > 0 {
            (self.total_frames as f64 * 1000.0 / self.sample_rate as f64) as u64
        } else {
            0
        }
    }

    /// Whether the engine is currently playing.
    pub fn is_playing(&self) -> bool {
        self.stream.as_ref().map_or(false, |s| s.is_playing())
    }

    // ========== Lock-free parameter setters ==========
    // These are safe to call from any thread (Python GUI).

    /// Set maximizer gain in dB (0-20).
    pub fn set_gain(&self, gain_db: f32) {
        self.params.gain_db.store(gain_db.clamp(0.0, 20.0));
        self.params.mark_dirty();
    }

    /// Set maximizer ceiling in dBFS (-3.0 to -0.1).
    pub fn set_ceiling(&self, ceiling_db: f32) {
        self.params.ceiling_db.store(ceiling_db.clamp(-3.0, -0.1));
        self.params.mark_dirty();
    }

    /// Set IRC mode (1-5, 0=LL).
    pub fn set_irc_mode(&self, mode: &str) {
        let mode_int = match mode {
            "IRC 1" => 1,
            "IRC 2" => 2,
            "IRC 3" => 3,
            "IRC 4" => 4,
            "IRC 5" => 5,
            "IRC LL" => 0,
            _ => 3,
        };
        self.params.irc_mode.store(mode_int, Ordering::Relaxed);
        self.params.mark_dirty();
    }

    /// Set stereo width (0-200%).
    pub fn set_width(&self, width_pct: f32) {
        self.params.width_pct.store(width_pct.clamp(0.0, 200.0));
        self.params.mark_dirty();
    }

    /// Set multiband imager widths.
    pub fn set_multiband_width(&self, low: f32, mid: f32, high: f32) {
        self.params.low_width.store(low.clamp(0.0, 200.0));
        self.params.mid_width.store(mid.clamp(0.0, 200.0));
        self.params.high_width.store(high.clamp(0.0, 200.0));
        self.params.imager_multiband.store(true, Ordering::Relaxed);
        self.params.mark_dirty();
    }

    /// Set EQ band gain (band 0-7, gain in dB).
    pub fn set_eq_gain(&self, band: usize, gain_db: f32) {
        if band < 8 {
            self.params.eq_gains[band].store(gain_db.clamp(-12.0, 12.0));
            self.params.mark_dirty();
        }
    }

    /// Set EQ bypass.
    pub fn set_eq_bypass(&self, bypass: bool) {
        self.params.eq_bypass.store(bypass, Ordering::Relaxed);
        self.params.mark_dirty();
    }

    /// Set output limiter ceiling.
    pub fn set_limiter_ceiling(&self, ceiling_db: f32) {
        self.params.limiter_ceiling_db.store(ceiling_db);
        self.params.mark_dirty();
    }

    /// Set master volume (0.0-1.0).
    pub fn set_volume(&self, volume: f32) {
        self.params.volume.store(volume.clamp(0.0, 2.0));
    }

    // ========== Meter data ==========

    /// Poll latest meter data. Returns the most recent MeterData.
    pub fn get_meter_data(&mut self) -> &MeterData {
        if let Some(ref rx) = self.meter_rx {
            // Drain channel, keep latest
            while let Ok(data) = rx.try_recv() {
                self.last_meter = data;
            }
        }
        &self.last_meter
    }

    /// Get sample rate of loaded file.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Get total frames of loaded file.
    pub fn total_frames(&self) -> u64 {
        self.total_frames
    }
}

impl Default for RtEngine {
    fn default() -> Self {
        Self::new()
    }
}
