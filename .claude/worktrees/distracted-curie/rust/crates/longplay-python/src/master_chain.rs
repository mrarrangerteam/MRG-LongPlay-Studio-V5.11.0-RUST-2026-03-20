//! PyMasterChain - Full PyO3 binding for the mastering chain
//! Sub-module delegation for EQ, Dynamics, Imager, Maximizer

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;

use longplay_chain::MasterChain;

use crate::types::PyMasterRecommendation;

/// Python-facing MasterChain wrapper with full module delegation
#[pyclass]
pub struct PyMasterChain {
    inner: MasterChain,
}

#[pymethods]
impl PyMasterChain {
    #[new]
    #[pyo3(signature = (ffmpeg_path="ffmpeg"))]
    fn new(ffmpeg_path: &str) -> Self {
        let _ = ffmpeg_path; // MasterChain uses ffmpeg from PATH
        Self {
            inner: MasterChain::new(),
        }
    }

    /// Load an audio file for processing
    fn load_audio(&mut self, path: &str) -> bool {
        self.inner.load_audio(path);
        true
    }

    /// Set target platform
    fn set_platform(&mut self, platform: &str) -> PyResult<()> {
        self.inner.set_platform(platform)
            .map_err(|e| PyRuntimeError::new_err(e))
    }

    /// Set processing intensity (0-100)
    fn set_intensity(&mut self, intensity: f32) -> PyResult<()> {
        self.inner.set_intensity(intensity)
            .map_err(|e| PyRuntimeError::new_err(e))
    }

    fn set_target_lufs(&mut self, lufs: f32) {
        self.inner.set_target_lufs(lufs);
    }

    fn set_target_tp(&mut self, tp: f32) {
        self.inner.set_target_tp(tp);
    }

    fn get_target_lufs(&self) -> f32 {
        self.inner.get_target_lufs()
    }

    fn get_target_tp(&self) -> f32 {
        self.inner.get_target_tp()
    }

    fn get_intensity(&self) -> f32 {
        self.inner.get_intensity()
    }

    /// Get AI recommendation, returns PyMasterRecommendation object
    fn ai_recommend(&mut self, genre: &str, platform: &str, intensity: f32) -> PyResult<PyMasterRecommendation> {
        let rec = self.inner.ai_recommend(genre, platform, intensity)
            .map_err(|e| PyRuntimeError::new_err(e))?;

        let py_rec = PyMasterRecommendation {
            genre: rec.genre.clone(),
            platform: rec.platform.clone(),
            intensity: rec.intensity,
            confidence: rec.confidence,
            explanations: rec.explanations.clone(),
        };

        self.inner.apply_recommendation(&rec);
        Ok(py_rec)
    }

    /// Apply a recommendation object
    /// V5.5 FIX: Build internal rec from PyMasterRecommendation fields
    /// instead of re-running ai_recommend (which wastes computation and may differ)
    fn apply_recommendation(&mut self, rec: &PyMasterRecommendation) -> PyResult<()> {
        // Set parameters directly from the recommendation
        self.inner.set_intensity(rec.intensity)
            .map_err(|e| PyRuntimeError::new_err(e))?;
        let _ = self.inner.set_platform(&rec.platform);
        // Re-run only if we need full module settings (current limitation)
        let inner_rec = self.inner.ai_recommend(&rec.genre, &rec.platform, rec.intensity)
            .map_err(|e| PyRuntimeError::new_err(e))?;
        self.inner.apply_recommendation(&inner_rec);
        Ok(())
    }

    /// Preview a section of audio, returns path to temp WAV or None on error
    #[pyo3(signature = (start_sec, duration_sec, callback=None))]
    fn preview(&mut self, start_sec: f32, duration_sec: f32, callback: Option<Py<PyAny>>) -> Option<String> {
        let _ = callback; // TODO: wire up progress callback

        // V5.5 FIX: Read sample rate from input file instead of hardcoding 44100
        let sample_rate = if !self.inner.get_input_path().is_empty() {
            match longplay_io::read_audio(self.inner.get_input_path()) {
                Ok((_, info)) => info.sample_rate as u32,
                Err(_) => 44100,
            }
        } else {
            44100
        };

        let buffer = self.inner.preview(start_sec, duration_sec);
        if buffer.is_empty() || buffer[0].is_empty() {
            return None;
        }

        // Write to temp file
        let temp_path = std::env::temp_dir().join("longplay_preview.wav");
        let temp_str = temp_path.to_str().unwrap_or("/tmp/longplay_preview.wav");

        match longplay_io::write_audio(temp_str, &buffer, sample_rate as i32, 24) {
            Ok(_) => Some(temp_str.to_string()),
            Err(e) => {
                eprintln!("Preview write error: {}", e);
                None
            }
        }
    }

    /// Render the mastered audio to output_path
    #[pyo3(signature = (output_path=None, callback=None))]
    fn render(&mut self, py: Python<'_>, output_path: Option<&str>, callback: Option<Py<PyAny>>) -> Option<String> {
        let path = output_path.unwrap_or("mastered_output.wav");

        let progress_cb: Option<longplay_chain::ProgressCallback> = callback.map(|py_cb| {
            let cb: longplay_chain::ProgressCallback = Box::new(move |progress: f32, msg: &str| {
                // The render runs on the same thread with the GIL held
                let _ = Python::try_attach(|py| {
                    let _ = py_cb.call(py, (progress, msg), None);
                });
            });
            cb
        });

        let success = self.inner.render(path, progress_cb);
        let _ = py;
        if success {
            Some(path.to_string())
        } else {
            None
        }
    }

    fn get_summary(&self) -> String {
        self.inner.get_chain_summary()
    }

    fn save_settings(&self, filepath: &str) -> bool {
        self.inner.save_settings(filepath)
    }

    fn load_settings(&mut self, filepath: &str) -> bool {
        self.inner.load_settings(filepath)
    }

    fn reset_all(&mut self) {
        self.inner.reset_all();
    }

    fn get_ab_comparison(&self) -> String {
        let ab = self.inner.get_ab_comparison();
        format!(
            "Before LUFS: {:.2}\nAfter LUFS: {:.2}\nBefore TP: {:.2}\nAfter TP: {:.2}\nLUFS Change: {:+.2}\nTP Change: {:+.2}",
            ab.before_lufs, ab.after_lufs,
            ab.before_tp, ab.after_tp,
            ab.lufs_change, ab.tp_change
        )
    }

    // ==================== EQ Delegation ====================

    fn eq_load_tone_preset(&mut self, name: &str) {
        self.inner.get_equalizer_mut().apply_tone_preset(name);
    }

    fn eq_set_bypass(&mut self, bypass: bool) {
        self.inner.get_equalizer_mut().set_bypass(bypass);
    }

    fn eq_is_bypassed(&self) -> bool {
        self.inner.get_equalizer().is_bypassed()
    }

    fn eq_get_tone_preset_names(&self) -> Vec<String> {
        longplay_dsp::equalizer::Equalizer::get_tone_preset_names()
    }

    fn eq_set_band(&mut self, index: usize, frequency: f64, gain_db: f64, q: f64) {
        let band = self.inner.get_equalizer_mut().band_mut(index);
        band.set_frequency(frequency);
        band.set_gain(gain_db);
        band.set_q(q);
    }

    fn eq_set_band_enabled(&mut self, index: usize, enabled: bool) {
        self.inner.get_equalizer_mut().band_mut(index).set_enabled(enabled);
    }

    fn eq_reset(&mut self) {
        self.inner.get_equalizer_mut().reset();
    }

    // ==================== Dynamics Delegation ====================

    fn dynamics_set_bypass(&mut self, bypass: bool) {
        self.inner.get_dynamics_mut().set_bypass(bypass);
    }

    fn dynamics_is_bypassed(&self) -> bool {
        self.inner.get_dynamics().is_bypassed()
    }

    fn dynamics_set_multiband(&mut self, enabled: bool) {
        self.inner.get_dynamics_mut().set_multiband(enabled);
    }

    fn dynamics_apply_preset(&mut self, name: &str) {
        self.inner.get_dynamics_mut().apply_preset(name);
    }

    fn dynamics_get_preset_names(&self) -> Vec<String> {
        longplay_dsp::dynamics::Dynamics::get_preset_names()
    }

    fn dynamics_set_threshold(&mut self, threshold_db: f64) {
        self.inner.get_dynamics_mut().single_band_mut().set_threshold(threshold_db);
    }

    fn dynamics_set_ratio(&mut self, ratio: f64) {
        self.inner.get_dynamics_mut().single_band_mut().set_ratio(ratio);
    }

    fn dynamics_set_attack(&mut self, attack_ms: f64) {
        self.inner.get_dynamics_mut().single_band_mut().set_attack(attack_ms);
    }

    fn dynamics_set_release(&mut self, release_ms: f64) {
        self.inner.get_dynamics_mut().single_band_mut().set_release(release_ms);
    }

    fn dynamics_set_makeup_gain(&mut self, gain_db: f64) {
        self.inner.get_dynamics_mut().single_band_mut().set_makeup_gain(gain_db);
    }

    fn dynamics_reset(&mut self) {
        self.inner.get_dynamics_mut().reset();
    }

    // ==================== Imager Delegation ====================

    fn imager_set_bypass(&mut self, bypass: bool) {
        self.inner.get_imager_mut().set_bypass(bypass);
    }

    fn imager_is_bypassed(&self) -> bool {
        self.inner.get_imager().is_bypassed()
    }

    fn imager_set_width(&mut self, width_pct: f32) {
        self.inner.get_imager_mut().set_width(width_pct);
    }

    fn imager_get_width(&self) -> f32 {
        self.inner.get_imager().width()
    }

    fn imager_set_multiband(&mut self, enabled: bool) {
        self.inner.get_imager_mut().set_multiband(enabled);
    }

    fn imager_set_low_width(&mut self, width_pct: f32) {
        self.inner.get_imager_mut().set_low_width(width_pct);
    }

    fn imager_set_mid_width(&mut self, width_pct: f32) {
        self.inner.get_imager_mut().set_mid_width(width_pct);
    }

    fn imager_set_high_width(&mut self, width_pct: f32) {
        self.inner.get_imager_mut().set_high_width(width_pct);
    }

    fn imager_set_balance(&mut self, balance: f32) {
        self.inner.get_imager_mut().set_balance(balance);
    }

    fn imager_apply_preset(&mut self, name: &str) {
        self.inner.get_imager_mut().apply_preset(name);
    }

    fn imager_get_preset_names(&self) -> Vec<String> {
        longplay_dsp::imager::Imager::get_preset_names()
    }

    fn imager_reset(&mut self) {
        self.inner.get_imager_mut().reset();
    }

    // ==================== Maximizer Delegation ====================

    fn maximizer_set_bypass(&mut self, bypass: bool) {
        self.inner.get_maximizer_mut().set_bypass(bypass);
    }

    fn maximizer_is_bypassed(&self) -> bool {
        self.inner.get_maximizer().is_bypassed()
    }

    fn maximizer_set_gain(&mut self, gain_db: f32) {
        self.inner.get_maximizer_mut().set_gain_db(gain_db);
    }

    fn maximizer_get_gain(&self) -> f32 {
        self.inner.get_maximizer().gain_db()
    }

    fn maximizer_set_ceiling(&mut self, ceiling: f32) {
        self.inner.get_maximizer_mut().set_ceiling(ceiling);
    }

    fn maximizer_get_ceiling(&self) -> f32 {
        self.inner.get_maximizer().ceiling()
    }

    #[pyo3(signature = (mode, sub_mode=None))]
    fn maximizer_set_irc_mode(&mut self, mode: &str, sub_mode: Option<&str>) {
        self.inner.get_maximizer_mut().set_irc_mode(mode);
        if let Some(sm) = sub_mode {
            self.inner.get_maximizer_mut().set_sub_mode(sm);
        }
    }

    fn maximizer_get_irc_mode(&self) -> String {
        self.inner.get_maximizer().get_irc_mode().to_string()
    }

    fn maximizer_set_true_peak(&mut self, enabled: bool) {
        self.inner.get_maximizer_mut().set_true_peak(enabled);
    }

    fn maximizer_set_character(&mut self, character: f32) {
        self.inner.get_maximizer_mut().set_character(character as i32);
    }

    fn maximizer_set_tone(&mut self, tone: &str) {
        self.inner.get_maximizer_mut().set_tone(tone);
    }

    fn maximizer_set_upward_compress(&mut self, amount: f32) {
        self.inner.get_maximizer_mut().set_upward_compress_db(amount);
    }

    fn maximizer_set_soft_clip(&mut self, amount: f32) {
        self.inner.get_maximizer_mut().set_soft_clip(amount);
    }

    fn maximizer_set_transient_emphasis(&mut self, amount: f32) {
        self.inner.get_maximizer_mut().set_transient_emphasis(amount);
    }

    fn maximizer_reset(&mut self) {
        self.inner.get_maximizer_mut().reset();
    }
}
