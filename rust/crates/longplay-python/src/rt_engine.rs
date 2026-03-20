//! PyO3 bindings for the real-time audio engine.
//!
//! Exposes PyRtEngine to Python with lock-free parameter control.
//! Uses `unsendable` because cpal::Stream is not Send on all platforms.

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;

use longplay_rt::RtEngine;

/// Real-time audio engine with lock-free DSP parameter control.
///
/// Usage:
///     engine = PyRtEngine()
///     engine.load_file("track.wav")
///     engine.play()
///     engine.set_gain(15.0)       # instant, lock-free
///     engine.set_width(120)       # instant
///     engine.set_ceiling(-1.0)    # instant
///     engine.set_irc_mode("IRC 2") # instant
///     pos = engine.get_position() # current position in ms
///     meters = engine.get_meter_data()  # latest meter levels
#[pyclass(unsendable)]
pub struct PyRtEngine {
    inner: RtEngine,
}

#[pymethods]
impl PyRtEngine {
    #[new]
    fn new() -> Self {
        Self {
            inner: RtEngine::new(),
        }
    }

    /// Load an audio file into memory for real-time playback.
    fn load_file(&mut self, path: &str) -> PyResult<()> {
        self.inner
            .load_file(path)
            .map_err(|e| PyRuntimeError::new_err(e))
    }

    /// Start playback.
    fn play(&self) {
        self.inner.play();
    }

    /// Pause playback (maintains position).
    fn pause(&self) {
        self.inner.pause();
    }

    /// Stop playback (resets to beginning).
    fn stop(&self) {
        self.inner.stop();
    }

    /// Seek to a position in milliseconds.
    fn seek(&self, position_ms: u64) {
        self.inner.seek(position_ms);
    }

    /// Get current playback position in milliseconds.
    fn get_position(&self) -> u64 {
        self.inner.get_position_ms()
    }

    /// Get total duration in milliseconds.
    fn get_duration(&self) -> u64 {
        self.inner.get_duration_ms()
    }

    /// Whether the engine is currently playing.
    fn is_playing(&self) -> bool {
        self.inner.is_playing()
    }

    // ========== Lock-free parameter setters ==========

    /// Set maximizer gain in dB (0-20). Instant, lock-free.
    fn set_gain(&self, gain_db: f32) {
        self.inner.set_gain(gain_db);
    }

    /// Set maximizer ceiling in dBFS (-3.0 to -0.1). Instant, lock-free.
    fn set_ceiling(&self, ceiling_db: f32) {
        self.inner.set_ceiling(ceiling_db);
    }

    /// Set IRC mode ("IRC 1" through "IRC 5", or "IRC LL"). Instant, lock-free.
    fn set_irc_mode(&self, mode: &str) {
        self.inner.set_irc_mode(mode);
    }

    /// Set stereo width (0-200%). Instant, lock-free.
    fn set_width(&self, width_pct: f32) {
        self.inner.set_width(width_pct);
    }

    /// Set multiband imager widths. Instant, lock-free.
    fn set_multiband_width(&self, low: f32, mid: f32, high: f32) {
        self.inner.set_multiband_width(low, mid, high);
    }

    /// Set EQ band gain (band 0-7, gain in dB). Instant, lock-free.
    fn set_eq_gain(&self, band: usize, gain_db: f32) {
        self.inner.set_eq_gain(band, gain_db);
    }

    /// Set EQ bypass. Instant, lock-free.
    fn set_eq_bypass(&self, bypass: bool) {
        self.inner.set_eq_bypass(bypass);
    }

    /// Set output limiter ceiling in dB. Instant, lock-free.
    fn set_limiter_ceiling(&self, ceiling_db: f32) {
        self.inner.set_limiter_ceiling(ceiling_db);
    }

    /// Set master volume (0.0-2.0). Instant, lock-free.
    fn set_volume(&self, volume: f32) {
        self.inner.set_volume(volume);
    }

    // ========== Resonance Suppressor (Soothe2-style) ==========

    /// Set resonance suppressor depth (0-20 dB). Instant, lock-free.
    fn set_res_depth(&self, depth: f32) {
        self.inner.set_res_depth(depth);
    }

    /// Set resonance suppressor sharpness (1-10). Instant, lock-free.
    fn set_res_sharpness(&self, val: f32) {
        self.inner.set_res_sharpness(val);
    }

    /// Set resonance suppressor selectivity (1-10). Instant, lock-free.
    fn set_res_selectivity(&self, val: f32) {
        self.inner.set_res_selectivity(val);
    }

    /// Set resonance suppressor attack (0.5-50 ms). Instant, lock-free.
    fn set_res_attack(&self, ms: f32) {
        self.inner.set_res_attack(ms);
    }

    /// Set resonance suppressor release (5-500 ms). Instant, lock-free.
    fn set_res_release(&self, ms: f32) {
        self.inner.set_res_release(ms);
    }

    /// Set resonance suppressor mode ("soft" or "hard"). Instant, lock-free.
    fn set_res_mode(&self, mode: &str) {
        self.inner.set_res_mode(mode);
    }

    /// Set resonance suppressor wet/dry mix (0-100%). Instant, lock-free.
    fn set_res_mix(&self, mix_pct: f32) {
        self.inner.set_res_mix(mix_pct);
    }

    /// Set resonance suppressor trim (-12 to +12 dB). Instant, lock-free.
    fn set_res_trim(&self, db: f32) {
        self.inner.set_res_trim(db);
    }

    /// Set resonance suppressor delta mode. Instant, lock-free.
    fn set_res_delta(&self, enabled: bool) {
        self.inner.set_res_delta(enabled);
    }

    /// Set resonance suppressor bypass. Instant, lock-free.
    fn set_res_bypass(&self, bypass: bool) {
        self.inner.set_res_bypass(bypass);
    }

    // ========== Dynamics (Compressor) ==========

    /// Set dynamics threshold (-60 to 0 dB). Instant, lock-free.
    fn set_dyn_threshold(&self, db: f32) {
        self.inner.set_dyn_threshold(db);
    }

    /// Set dynamics ratio (1-20). Instant, lock-free.
    fn set_dyn_ratio(&self, ratio: f32) {
        self.inner.set_dyn_ratio(ratio);
    }

    /// Set dynamics attack (0.1-100 ms). Instant, lock-free.
    fn set_dyn_attack(&self, ms: f32) {
        self.inner.set_dyn_attack(ms);
    }

    /// Set dynamics release (1-1000 ms). Instant, lock-free.
    fn set_dyn_release(&self, ms: f32) {
        self.inner.set_dyn_release(ms);
    }

    /// Set dynamics makeup gain (-12 to +24 dB). Instant, lock-free.
    fn set_dyn_makeup(&self, db: f32) {
        self.inner.set_dyn_makeup(db);
    }

    /// Set dynamics knee (0-20 dB). Instant, lock-free.
    fn set_dyn_knee(&self, db: f32) {
        self.inner.set_dyn_knee(db);
    }

    /// Set dynamics bypass. Instant, lock-free.
    fn set_dyn_bypass(&self, bypass: bool) {
        self.inner.set_dyn_bypass(bypass);
    }

    /// Get latest meter data as a dict.
    ///
    /// Returns: {
    ///     "peak_l": float,     # dB
    ///     "peak_r": float,     # dB
    ///     "rms_l": float,      # dB
    ///     "rms_r": float,      # dB
    ///     "gain_reduction_db": float,
    ///     "position_ms": int,
    /// }
    fn get_meter_data<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let sr = self.inner.sample_rate();
        let meter = self.inner.get_meter_data().clone();
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("peak_l", meter.peak_l)?;
        dict.set_item("peak_r", meter.peak_r)?;
        dict.set_item("rms_l", meter.rms_l)?;
        dict.set_item("rms_r", meter.rms_r)?;
        dict.set_item("gain_reduction_db", meter.gain_reduction_db)?;
        dict.set_item("position_ms", {
            if sr > 0 {
                (meter.position_frames as f64 * 1000.0 / sr as f64) as u64
            } else {
                0u64
            }
        })?;
        Ok(dict)
    }

    /// Get sample rate of loaded file.
    fn get_sample_rate(&self) -> u32 {
        self.inner.sample_rate()
    }

    /// Get total frames of loaded file.
    fn get_total_frames(&self) -> u64 {
        self.inner.total_frames()
    }
}
