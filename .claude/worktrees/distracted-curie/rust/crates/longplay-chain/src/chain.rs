//! MasterChain - main orchestrator for audio mastering pipeline
//! Ported from C++ chain.h / chain.cpp

use longplay_core::AudioBuffer;
use longplay_core::conversions::db_to_linear_f;
use longplay_dsp::equalizer::Equalizer;
use longplay_dsp::dynamics::Dynamics;
use longplay_dsp::imager::Imager;
use longplay_dsp::maximizer::Maximizer;
use longplay_analysis::analyzer::{AudioAnalyzer, AudioAnalysis};
use longplay_analysis::loudness::{LoudnessMeter, LoudnessAnalysis};
use longplay_ai::recommend::{AIAssist, MasterRecommendation};
use longplay_profiles::genre_profiles::{GenreProfiles, PlatformTarget};

use crate::settings_io;

/// Before/After comparison structure
#[derive(Debug, Clone)]
pub struct ABComparison {
    pub before_lufs: f32,
    pub after_lufs: f32,
    pub before_tp: f32,
    pub after_tp: f32,
    pub lufs_change: f32,
    pub tp_change: f32,
}

impl Default for ABComparison {
    fn default() -> Self {
        Self {
            before_lufs: 0.0,
            after_lufs: 0.0,
            before_tp: 0.0,
            after_tp: 0.0,
            lufs_change: 0.0,
            tp_change: 0.0,
        }
    }
}

/// Progress callback type
pub type ProgressCallback = Box<dyn FnMut(f32, &str)>;

/// Main orchestrator that ties together all processing modules.
///
/// The MasterChain coordinates the complete mastering signal flow including
/// EQ, dynamics processing, stereo imaging, maximization, and loudness normalization.
pub struct MasterChain {
    // Processing modules
    equalizer: Equalizer,
    dynamics: Dynamics,
    imager: Imager,
    maximizer: Maximizer,
    analyzer: AudioAnalyzer,
    loudness_meter: LoudnessMeter,
    ai_assist: AIAssist,

    // File paths
    input_path: String,
    output_path: String,

    // Processing parameters
    intensity: f32,
    target_lufs: f32,
    target_tp: f32,

    // Analysis results
    input_analysis: Option<AudioAnalysis>,
    output_analysis: Option<AudioAnalysis>,
    input_loudness: Option<LoudnessAnalysis>,
    output_loudness: Option<LoudnessAnalysis>,

    // AI recommendation
    recommendation: Option<MasterRecommendation>,
}

impl MasterChain {
    /// Construct a new MasterChain instance with default settings
    pub fn new() -> Self {
        Self {
            equalizer: Equalizer::new(),
            dynamics: Dynamics::new(),
            imager: Imager::new(),
            maximizer: Maximizer::new(),
            analyzer: AudioAnalyzer::new(),
            loudness_meter: LoudnessMeter::new("ffmpeg"),
            ai_assist: AIAssist::new(),
            input_path: String::new(),
            output_path: String::new(),
            intensity: 75.0,
            target_lufs: -14.0,
            target_tp: -1.0,
            input_analysis: None,
            output_analysis: None,
            input_loudness: None,
            output_loudness: None,
            recommendation: None,
        }
    }

    // ========== Audio I/O Methods ==========

    /// Load audio file for processing
    pub fn load_audio(&mut self, path: &str) {
        self.input_path = path.to_string();
        self.analyze_input_file();
    }

    /// Get the current input file path
    pub fn get_input_path(&self) -> &str {
        &self.input_path
    }

    /// Get the current output file path
    pub fn get_output_path(&self) -> &str {
        &self.output_path
    }

    // ========== Platform & Target Methods ==========

    /// Set processing targets based on platform name.
    /// Supported: spotify, apple_music, youtube, tidal, amazon, soundcloud, radio, cd
    pub fn set_platform(&mut self, platform: &str) -> Result<(), String> {
        let targets = GenreProfiles::get_platform_targets();
        let key = platform.to_lowercase();
        match targets.get(&key) {
            Some(target) => {
                self.target_lufs = target.target_lufs as f32;
                self.target_tp = target.max_true_peak as f32;
                Ok(())
            }
            None => Err(format!("Unknown platform: {}", platform)),
        }
    }

    /// Get available platform targets
    pub fn get_platform_targets() -> std::collections::HashMap<String, PlatformTarget> {
        GenreProfiles::get_platform_targets().clone()
    }

    /// Get current target LUFS
    pub fn get_target_lufs(&self) -> f32 {
        self.target_lufs
    }

    /// Set target LUFS value
    pub fn set_target_lufs(&mut self, lufs: f32) {
        self.target_lufs = lufs;
    }

    /// Get current target true peak
    pub fn get_target_tp(&self) -> f32 {
        self.target_tp
    }

    /// Set target true peak in dBTP
    pub fn set_target_tp(&mut self, tp: f32) {
        self.target_tp = tp;
    }

    // ========== Processing Intensity Methods ==========

    /// Get current processing intensity (0-100)
    pub fn get_intensity(&self) -> f32 {
        self.intensity
    }

    /// Set processing intensity (0-100)
    pub fn set_intensity(&mut self, intensity: f32) -> Result<(), String> {
        if intensity < 0.0 || intensity > 100.0 {
            return Err("Intensity must be between 0 and 100".to_string());
        }
        self.intensity = intensity;
        Ok(())
    }

    // ========== AI Recommendation Methods ==========

    /// Get AI-assisted mastering recommendations
    pub fn ai_recommend(
        &mut self,
        genre: &str,
        platform: &str,
        intensity: f32,
    ) -> Result<MasterRecommendation, String> {
        if intensity < 0.0 || intensity > 100.0 {
            return Err("Intensity must be between 0 and 100".to_string());
        }

        // Validate platform
        let targets = GenreProfiles::get_platform_targets();
        let key = platform.to_lowercase();
        if !targets.contains_key(&key) {
            return Err(format!("Unknown platform: {}", platform));
        }

        if self.input_path.is_empty() {
            return Err("No input audio loaded for AI recommendation".to_string());
        }

        let rec = self.ai_assist.analyze_and_recommend(
            &self.input_path,
            genre,
            &key,
            intensity,
        )?;

        self.recommendation = Some(rec.clone());
        Ok(rec)
    }

    /// Apply recommendation settings to all modules
    pub fn apply_recommendation(&mut self, rec: &MasterRecommendation) {
        self.equalizer.from_settings(&rec.equalizer_settings);
        self.dynamics.from_settings(&rec.dynamics_settings);
        self.imager.from_settings(&rec.imager_settings);
        self.maximizer.from_settings(&rec.maximizer_settings);

        self.intensity = rec.intensity;

        if !rec.platform.is_empty() {
            let _ = self.set_platform(&rec.platform);
        }

        self.recommendation = Some(rec.clone());
    }

    /// Get the last recommendation if available
    pub fn get_last_recommendation(&self) -> Option<&MasterRecommendation> {
        self.recommendation.as_ref()
    }

    // ========== Core DSP Processing Methods ==========

    /// Process audio buffer through the complete mastering chain.
    ///
    /// Signal flow (7 stages):
    /// 1. Pre-gain: Apply -3dB headroom for hot tracks
    /// 2. EQ: Apply equalizer processing
    /// 3. Dynamics: Apply dynamic range processing
    /// 4. Imager: Apply stereo imaging
    /// 5. Maximizer: Apply limiting/maximization
    /// 6. Loudness normalization: Gain adjustment to reach target LUFS
    /// 7. True peak safety: Hard clip at ceiling
    pub fn process_audio(&mut self, input_buffer: &AudioBuffer, sample_rate: u32) -> AudioBuffer {
        if input_buffer.is_empty() || input_buffer[0].is_empty() {
            return input_buffer.clone();
        }

        let mut buffer = input_buffer.clone();

        // Stage 1: Pre-gain (-3dB headroom for hot tracks)
        let pre_gain: f32 = 0.707; // -3dB
        for channel in &mut buffer {
            for sample in channel.iter_mut() {
                *sample *= pre_gain;
            }
        }

        // Stage 2: EQ Processing (returns new buffer)
        buffer = self.equalizer.process(&buffer, sample_rate as i32);

        // Stage 3: Dynamics Processing (returns new buffer)
        buffer = self.dynamics.process(&buffer, sample_rate as i32);

        // Stage 4: Stereo Imaging (modifies in-place)
        self.imager.process(&mut buffer, sample_rate as i32);

        // Stage 5: Maximizer/Limiter (modifies in-place)
        self.maximizer.process(&mut buffer, sample_rate as i32);

        // Stage 6: Loudness Normalization (RMS-based estimation)
        let loudness_gain = self.estimate_rms_gain(&buffer);
        for channel in &mut buffer {
            for sample in channel.iter_mut() {
                *sample *= loudness_gain;
            }
        }

        // Stage 7: True Peak Safety (hard clip at ceiling)
        self.apply_hard_clip(&mut buffer, self.target_tp);

        buffer
    }

    // ========== File Processing Methods ==========

    /// Render input file to output with progress callback.
    ///
    /// Reads the entire input file, processes through the mastering chain,
    /// and writes to output file with optional progress reporting.
    pub fn render(
        &mut self,
        output_path: &str,
        mut progress_callback: Option<ProgressCallback>,
    ) -> bool {
        self.output_path = output_path.to_string();

        if self.input_path.is_empty() {
            eprintln!("Error: No input audio loaded");
            return false;
        }

        // Report progress: Reading
        if let Some(ref mut cb) = progress_callback {
            cb(0.1, "Reading input file");
        }

        // Read input file
        let (buffer, info) = match longplay_io::read_audio(&self.input_path) {
            Ok(result) => result,
            Err(e) => {
                eprintln!("Error reading input: {}", e);
                if let Some(ref mut cb) = progress_callback {
                    cb(1.0, &format!("Error: {}", e));
                }
                return false;
            }
        };

        // Report progress: Processing
        if let Some(ref mut cb) = progress_callback {
            cb(0.2, "Processing audio");
        }

        // Process audio through chain
        let processed = self.process_audio(&buffer, info.sample_rate as u32);

        // Report progress: Analyzing
        if let Some(ref mut cb) = progress_callback {
            cb(0.8, "Analyzing output");
        }

        // Report progress: Writing
        if let Some(ref mut cb) = progress_callback {
            cb(0.9, "Writing output file");
        }

        // Write output file
        let bit_depth = if info.bit_depth > 0 { info.bit_depth } else { 24 };
        let write_result = longplay_io::write_audio(
            output_path,
            &processed,
            info.sample_rate,
            bit_depth,
        );

        let write_success = write_result.is_ok();

        if write_success {
            self.analyze_output_file(output_path);
        }

        if let Some(ref mut cb) = progress_callback {
            if write_success {
                cb(1.0, "Render complete");
            } else {
                cb(1.0, "Render failed");
            }
        }

        if let Err(e) = write_result {
            eprintln!("Error writing output: {}", e);
        }

        write_success
    }

    /// Generate a preview of a section of the audio
    pub fn preview(&mut self, start_sec: f32, duration_sec: f32) -> AudioBuffer {
        if self.input_path.is_empty() {
            eprintln!("Error: No input audio loaded");
            return AudioBuffer::new();
        }

        if start_sec < 0.0 || duration_sec <= 0.0 {
            eprintln!("Error: Invalid preview parameters");
            return AudioBuffer::new();
        }

        // Read entire file
        let (buffer, info) = match longplay_io::read_audio(&self.input_path) {
            Ok(result) => result,
            Err(e) => {
                eprintln!("Error reading input for preview: {}", e);
                return AudioBuffer::new();
            }
        };

        if buffer.is_empty() || buffer[0].is_empty() {
            return AudioBuffer::new();
        }

        let num_channels = buffer.len();
        let total_frames = buffer[0].len();
        let start_frame = (start_sec * info.sample_rate as f32) as usize;
        let duration_frames = (duration_sec * info.sample_rate as f32) as usize;

        // Clamp to valid range
        if start_frame >= total_frames {
            return AudioBuffer::new();
        }
        let end_frame = (start_frame + duration_frames).min(total_frames);

        // Extract section
        let mut section = vec![Vec::new(); num_channels];
        for ch in 0..num_channels {
            section[ch] = buffer[ch][start_frame..end_frame].to_vec();
        }

        // Process through chain
        self.process_audio(&section, info.sample_rate as u32)
    }

    // ========== Analysis Methods ==========

    /// Get before/after comparison statistics
    pub fn get_ab_comparison(&self) -> ABComparison {
        let mut comparison = ABComparison::default();

        if let Some(ref input_loudness) = self.input_loudness {
            comparison.before_lufs = input_loudness.integrated_lufs;
            comparison.before_tp = input_loudness.true_peak_dbtp;
        }

        if let Some(ref output_loudness) = self.output_loudness {
            comparison.after_lufs = output_loudness.integrated_lufs;
            comparison.after_tp = output_loudness.true_peak_dbtp;
        }

        comparison.lufs_change = comparison.after_lufs - comparison.before_lufs;
        comparison.tp_change = comparison.after_tp - comparison.before_tp;

        comparison
    }

    /// Get input file analysis if available
    pub fn get_input_analysis(&self) -> Option<&AudioAnalysis> {
        self.input_analysis.as_ref()
    }

    /// Get output file analysis if available
    pub fn get_output_analysis(&self) -> Option<&AudioAnalysis> {
        self.output_analysis.as_ref()
    }

    /// Get input loudness analysis if available
    pub fn get_input_loudness(&self) -> Option<&LoudnessAnalysis> {
        self.input_loudness.as_ref()
    }

    /// Get output loudness analysis if available
    pub fn get_output_loudness(&self) -> Option<&LoudnessAnalysis> {
        self.output_loudness.as_ref()
    }

    // ========== Settings Management Methods ==========

    /// Save all module settings to JSON file
    pub fn save_settings(&self, filepath: &str) -> bool {
        settings_io::save_chain_settings(
            filepath,
            self.intensity,
            self.target_lufs,
            self.target_tp,
            &self.equalizer.to_settings(),
            &self.dynamics.to_settings(),
            &self.imager.to_settings(),
            &self.maximizer.to_settings(),
        )
    }

    /// Load all module settings from JSON file
    pub fn load_settings(&mut self, filepath: &str) -> bool {
        match settings_io::load_chain_settings(filepath) {
            Some(loaded) => {
                self.intensity = loaded.intensity;
                self.target_lufs = loaded.target_lufs;
                self.target_tp = loaded.target_tp;

                if let Some(ref s) = loaded.equalizer {
                    self.equalizer.from_settings(s);
                }
                if let Some(ref s) = loaded.dynamics {
                    self.dynamics.from_settings(s);
                }
                if let Some(ref s) = loaded.imager {
                    self.imager.from_settings(s);
                }
                if let Some(ref s) = loaded.maximizer {
                    self.maximizer.from_settings(s);
                }

                true
            }
            None => false,
        }
    }

    /// Reset all modules to default settings
    pub fn reset_all(&mut self) {
        self.intensity = 75.0;
        self.target_lufs = -14.0;
        self.target_tp = -1.0;

        self.equalizer.reset();
        self.dynamics.reset();
        self.imager.reset();
        self.maximizer.reset();

        self.input_analysis = None;
        self.output_analysis = None;
        self.input_loudness = None;
        self.output_loudness = None;
        self.recommendation = None;
    }

    // ========== Status Methods ==========

    /// Get human-readable chain status summary
    pub fn get_chain_summary(&self) -> String {
        let mut summary = String::new();

        summary.push_str("=== MasterChain Summary ===\n");
        summary.push_str(&format!(
            "Input: {}\n",
            if self.input_path.is_empty() { "[None]" } else { &self.input_path }
        ));
        summary.push_str(&format!(
            "Output: {}\n",
            if self.output_path.is_empty() { "[None]" } else { &self.output_path }
        ));
        summary.push_str(&format!("Intensity: {}%\n", self.intensity as i32));
        summary.push_str(&format!("Target LUFS: {:.1}\n", self.target_lufs));
        summary.push_str(&format!("Target TP: {:.1} dBTP\n", self.target_tp));

        if let Some(ref loudness) = self.input_loudness {
            summary.push_str("\nInput Loudness:\n");
            summary.push_str(&format!("  LUFS: {:.1}\n", loudness.integrated_lufs));
            summary.push_str(&format!("  True Peak: {:.1} dBTP\n", loudness.true_peak_dbtp));
        }

        if let Some(ref analysis) = self.input_analysis {
            summary.push_str("\nInput Spectral:\n");
            summary.push_str(&format!("  Brightness: {:.2}\n", analysis.spectral.brightness));
            summary.push_str(&format!(
                "  Crest Factor: {:.1} dB\n",
                analysis.dynamics.crest_factor_db
            ));
            summary.push_str(&format!(
                "  Stereo Correlation: {:.2}\n",
                analysis.stereo.correlation
            ));
        }

        if let Some(ref loudness) = self.output_loudness {
            summary.push_str("\nOutput Loudness:\n");
            summary.push_str(&format!("  LUFS: {:.1}\n", loudness.integrated_lufs));
            summary.push_str(&format!("  True Peak: {:.1} dBTP\n", loudness.true_peak_dbtp));
        }

        if let Some(ref rec) = self.recommendation {
            summary.push_str("\nActive Recommendation:\n");
            summary.push_str(&format!("  Genre: {}\n", rec.genre));
            summary.push_str(&format!("  Platform: {}\n", rec.platform));
            summary.push_str(&format!("  Confidence: {:.0}%\n", rec.confidence));
        }

        summary.push_str("\nModules:\n");
        summary.push_str(&format!(
            "  Equalizer: {}\n",
            if self.equalizer.is_bypassed() { "Bypassed" } else { "Active" }
        ));
        summary.push_str(&format!(
            "  Dynamics: {}\n",
            if self.dynamics.is_bypassed() { "Bypassed" } else { "Active" }
        ));
        summary.push_str(&format!(
            "  Imager: {}\n",
            if self.imager.is_bypassed() { "Bypassed" } else { "Active" }
        ));
        summary.push_str(&format!(
            "  Maximizer: {}\n",
            if self.maximizer.is_bypassed() { "Bypassed" } else { "Active" }
        ));
        summary.push_str("  Analyzer: Active\n");
        summary.push_str("  Loudness Meter: Active\n");

        summary
    }

    // ========== Module Access Methods ==========

    pub fn get_equalizer(&self) -> &Equalizer {
        &self.equalizer
    }

    pub fn get_equalizer_mut(&mut self) -> &mut Equalizer {
        &mut self.equalizer
    }

    pub fn get_dynamics(&self) -> &Dynamics {
        &self.dynamics
    }

    pub fn get_dynamics_mut(&mut self) -> &mut Dynamics {
        &mut self.dynamics
    }

    pub fn get_imager(&self) -> &Imager {
        &self.imager
    }

    pub fn get_imager_mut(&mut self) -> &mut Imager {
        &mut self.imager
    }

    pub fn get_maximizer(&self) -> &Maximizer {
        &self.maximizer
    }

    pub fn get_maximizer_mut(&mut self) -> &mut Maximizer {
        &mut self.maximizer
    }

    pub fn get_analyzer(&self) -> &AudioAnalyzer {
        &self.analyzer
    }

    pub fn get_analyzer_mut(&mut self) -> &mut AudioAnalyzer {
        &mut self.analyzer
    }

    pub fn get_loudness_meter(&self) -> &LoudnessMeter {
        &self.loudness_meter
    }

    pub fn get_ai_assist(&self) -> &AIAssist {
        &self.ai_assist
    }

    pub fn get_ai_assist_mut(&mut self) -> &mut AIAssist {
        &mut self.ai_assist
    }

    // ========== File Processing ==========

    /// Read input, process through chain, write output.
    pub fn process_file(&mut self, input_path: &str, output_path: &str) -> Result<(), String> {
        self.input_path = input_path.to_string();
        self.output_path = output_path.to_string();

        // Read input file
        let (buffer, info) = longplay_io::read_audio(input_path)
            .map_err(|e| format!("{}", e))?;

        // Analyze input
        self.analyze_input_file();

        // Process
        let processed = self.process_audio(&buffer, info.sample_rate as u32);

        // Write output
        let bit_depth = if info.bit_depth > 0 { info.bit_depth } else { 24 };
        longplay_io::write_audio(output_path, &processed, info.sample_rate, bit_depth)
            .map_err(|e| format!("{}", e))?;

        // Analyze output
        self.analyze_output_file(output_path);

        Ok(())
    }

    // ========== Genre Profile ==========

    /// Set genre profile, applying recommended presets to modules.
    pub fn set_genre_profile(&mut self, name: &str) {
        if let Some(profile) = GenreProfiles::get_genre_profile(name) {
            self.target_lufs = profile.target_lufs as f32;
            self.target_tp = profile.target_true_peak as f32;
        }
    }

    // ========== Module Access Aliases ==========

    /// Mutable reference to equalizer (alias for get_equalizer_mut)
    pub fn equalizer_mut(&mut self) -> &mut Equalizer {
        &mut self.equalizer
    }

    // ========== Private Helper Methods ==========

    fn analyze_input_file(&mut self) {
        if self.input_path.is_empty() {
            return;
        }

        match self.analyzer.analyze(&self.input_path) {
            Ok(analysis) => {
                self.input_analysis = Some(analysis);
            }
            Err(e) => {
                eprintln!("Warning: Failed to analyze input file: {}", e);
                self.input_analysis = None;
            }
        }

        match self.loudness_meter.analyze(&self.input_path) {
            Some(loudness) => {
                self.input_loudness = Some(loudness);
            }
            None => {
                eprintln!("Warning: Failed to measure input loudness");
                self.input_loudness = None;
            }
        }
    }

    fn analyze_output_file(&mut self, output_path: &str) {
        if output_path.is_empty() {
            return;
        }

        match self.analyzer.analyze(output_path) {
            Ok(analysis) => {
                self.output_analysis = Some(analysis);
            }
            Err(e) => {
                eprintln!("Warning: Failed to analyze output file: {}", e);
                self.output_analysis = None;
            }
        }

        match self.loudness_meter.analyze(output_path) {
            Some(loudness) => {
                self.output_loudness = Some(loudness);
            }
            None => {
                eprintln!("Warning: Failed to measure output loudness");
                self.output_loudness = None;
            }
        }
    }

    /// Estimate RMS-based gain to approximate target LUFS alignment.
    /// This is a fast buffer-based approximation (not ITU-R BS.1770-4 compliant).
    fn estimate_rms_gain(&self, buffer: &AudioBuffer) -> f32 {
        if buffer.is_empty() || buffer[0].is_empty() {
            return 1.0;
        }

        // Calculate RMS across all channels
        let mut sum_sq = 0.0f64;
        let mut total_samples = 0usize;

        for channel in buffer {
            for &sample in channel {
                sum_sq += (sample as f64) * (sample as f64);
            }
            total_samples += channel.len();
        }

        if total_samples == 0 {
            return 1.0;
        }

        let rms = (sum_sq / total_samples as f64).sqrt();

        if rms <= 1e-10 {
            return 1.0; // Silence -- no gain needed
        }

        // Convert RMS to approximate dBFS (rough LUFS proxy)
        let rms_db = 20.0 * rms.log10();

        // Calculate gain needed to reach target LUFS
        // Note: RMS approximates LUFS within ~3 LU for most program material
        let gain_db = (self.target_lufs as f64) - rms_db;

        // Limit gain range to prevent extreme amplification or attenuation
        let gain_db = gain_db.clamp(-20.0, 20.0);

        // Convert dB to linear gain
        10.0_f64.powf(gain_db / 20.0) as f32
    }

    /// Hard clip all samples exceeding the ceiling level
    fn apply_hard_clip(&self, buffer: &mut AudioBuffer, ceiling_db: f32) {
        let ceiling_linear = db_to_linear_f(ceiling_db);

        for channel in buffer.iter_mut() {
            for sample in channel.iter_mut() {
                if *sample > ceiling_linear {
                    *sample = ceiling_linear;
                } else if *sample < -ceiling_linear {
                    *sample = -ceiling_linear;
                }
            }
        }
    }
}

impl Default for MasterChain {
    fn default() -> Self {
        Self::new()
    }
}
