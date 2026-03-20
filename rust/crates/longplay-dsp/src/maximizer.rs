//! Maximizer — Ozone 12-style mastering limiter/dynamics processor.
//!
//! Signal flow: Tone Pre-EQ -> Upward Compression -> Transient Emphasis ->
//!              Soft Clip -> Input Gain Push -> IRC Limiter (6 algorithms)
//!
//! Now uses IRCLimiter with true multi-algorithm support:
//!   IRC 1: Transparent peak limiter
//!   IRC 2: Program-dependent adaptive release
//!   IRC 3: Multi-band frequency-weighted (3 sub-modes: Pumping/Balanced/Crisp)
//!   IRC 4: Aggressive saturation + limiting (3 sub-modes: Pumping/Balanced/Crisp)
//!   IRC 5: Maximum density (multi-band compression + limiting)
//!   IRC LL: Low-latency feedback limiter

use crate::biquad::FilterType;
use crate::equalizer::Equalizer;
use crate::irc_limiter::{IRCLimiter, IRCModeType};
use longplay_core::{AudioBuffer, Settings, SettingsValue};
use longplay_core::{get_setting_f32, get_setting_int, get_setting_bool, get_setting_string};
use longplay_core::conversions::db_to_linear_f;

/// Transient emphasis band levels (high, mid, low).
#[derive(Debug, Clone)]
pub struct TransientEmphasisBands {
    pub high: f32,
    pub mid: f32,
    pub low: f32,
}

impl Default for TransientEmphasisBands {
    fn default() -> Self { Self { high: 0.0, mid: 0.0, low: 0.0 } }
}

/// Stereo independence levels for transient and sustain.
#[derive(Debug, Clone)]
pub struct StereoIndependence {
    pub transient: f32,
    pub sustain: f32,
}

impl Default for StereoIndependence {
    fn default() -> Self { Self { transient: 0.0, sustain: 0.0 } }
}

/// Level information returned by measure_levels.
#[derive(Debug, Clone, Default)]
pub struct LevelInfo {
    pub peak_db: Vec<f32>,
    pub rms_db: Vec<f32>,
    pub integrated_lufs: f32,
    pub short_term_lufs: f32,
    pub loudness_range: f32,
    pub true_peak_dbtp: f32,
}

/// Maximizer module: Ozone 12-style mastering limiter/dynamics processor.
///
/// Signal flow: Tone Pre-EQ -> Upward Compression -> Transient Emphasis ->
///              Soft Clip -> Input Gain Push -> IRC Limiter
pub struct Maximizer {
    irc_mode: i32,
    sub_mode: String,
    gain_db: f32,
    ceiling_dbfs: f32,
    true_peak: bool,
    character: i32,
    tone: String,
    upward_compress_db: f32,
    soft_clip_pct: f32,
    transient_emphasis: f32,
    transient_emphasis_bands: TransientEmphasisBands,
    stereo_independence: StereoIndependence,
    bypass: bool,
    irc_mode_str: String,
    tone_eq: Equalizer,
    irc_limiter: IRCLimiter,
}

impl Maximizer {
    pub fn new() -> Self {
        let mut irc_limiter = IRCLimiter::new();
        irc_limiter.set_ceiling(-1.0);
        irc_limiter.set_true_peak(true);
        irc_limiter.set_mode(IRCModeType::IRC3); // Default: most popular mode

        Self {
            irc_mode: 3,
            sub_mode: "Balanced".into(),
            gain_db: 0.0,
            ceiling_dbfs: -1.0,
            true_peak: true,
            character: 5,
            tone: "Off".into(),
            upward_compress_db: 0.0,
            soft_clip_pct: 0.0,
            transient_emphasis: 0.0,
            transient_emphasis_bands: TransientEmphasisBands::default(),
            stereo_independence: StereoIndependence::default(),
            bypass: false,
            irc_mode_str: "IRC 3".into(),
            tone_eq: Equalizer::new(),
            irc_limiter,
        }
    }

    // ========== Parameter Setters ==========

    pub fn set_irc_mode(&mut self, mode: &str) {
        self.irc_mode_str = mode.to_string();
        let irc_mode = IRCModeType::from_str(mode);
        self.irc_limiter.set_mode(irc_mode);
        // Map to int for backward compat
        self.irc_mode = match irc_mode {
            IRCModeType::IRC1 => 1,
            IRCModeType::IRC2 => 2,
            IRCModeType::IRC3 => 3,
            IRCModeType::IRC4 => 4,
            IRCModeType::IRC5 => 5,
            IRCModeType::IRCLL => 0,
        };
    }

    pub fn set_irc_mode_int(&mut self, mode: i32) {
        self.irc_mode = mode.clamp(0, 5);
        let irc_mode = match self.irc_mode {
            1 => IRCModeType::IRC1,
            2 => IRCModeType::IRC2,
            3 => IRCModeType::IRC3,
            4 => IRCModeType::IRC4,
            5 => IRCModeType::IRC5,
            0 => IRCModeType::IRCLL,
            _ => IRCModeType::IRC3,
        };
        self.irc_limiter.set_mode(irc_mode);
        self.irc_mode_str = irc_mode.name().to_string();
    }

    pub fn set_sub_mode(&mut self, mode: &str) {
        self.sub_mode = mode.to_string();
        self.irc_limiter.set_sub_mode_str(mode);
    }

    pub fn set_gain_db(&mut self, gain: f32) {
        self.gain_db = gain.clamp(0.0, 20.0);
    }

    pub fn set_ceiling(&mut self, ceiling: f32) {
        self.ceiling_dbfs = ceiling.clamp(-3.0, -0.1);
        self.irc_limiter.set_ceiling(self.ceiling_dbfs as f64);
    }

    pub fn set_true_peak(&mut self, enabled: bool) {
        self.true_peak = enabled;
        self.irc_limiter.set_true_peak(enabled);
    }

    pub fn set_character(&mut self, character: i32) {
        self.character = character.clamp(0, 10);
    }

    pub fn set_tone(&mut self, tone: &str) {
        self.tone = tone.to_string();
    }

    pub fn set_upward_compress_db(&mut self, amount: f32) {
        self.upward_compress_db = amount.clamp(0.0, 12.0);
    }

    pub fn set_soft_clip(&mut self, amount: f32) {
        self.soft_clip_pct = amount.clamp(0.0, 100.0);
    }

    pub fn set_transient_emphasis(&mut self, amount: f32) {
        self.transient_emphasis = amount.clamp(0.0, 100.0);
    }

    pub fn set_transient_emphasis_bands(&mut self, bands: &TransientEmphasisBands) {
        self.transient_emphasis_bands.high = bands.high.clamp(0.0, 100.0);
        self.transient_emphasis_bands.mid = bands.mid.clamp(0.0, 100.0);
        self.transient_emphasis_bands.low = bands.low.clamp(0.0, 100.0);
    }

    pub fn set_stereo_independence(&mut self, stereo: &StereoIndependence) {
        self.stereo_independence.transient = stereo.transient.clamp(0.0, 100.0);
        self.stereo_independence.sustain = stereo.sustain.clamp(0.0, 100.0);
    }

    pub fn set_bypass(&mut self, bypass: bool) {
        self.bypass = bypass;
    }

    // ========== Parameter Getters ==========

    pub fn get_gain_db(&self) -> f32 { self.gain_db }
    pub fn get_ceiling(&self) -> f32 { self.ceiling_dbfs }
    pub fn get_true_peak(&self) -> bool { self.true_peak }
    pub fn get_character(&self) -> i32 { self.character }
    pub fn get_soft_clip(&self) -> f32 { self.soft_clip_pct }
    pub fn get_irc_mode(&self) -> &str { &self.irc_mode_str }
    pub fn is_bypassed(&self) -> bool { self.bypass }
    pub fn irc_mode(&self) -> i32 { self.irc_mode }
    pub fn sub_mode(&self) -> &str { &self.sub_mode }
    pub fn gain_db(&self) -> f32 { self.gain_db }
    pub fn ceiling(&self) -> f32 { self.ceiling_dbfs }
    pub fn character(&self) -> i32 { self.character }
    pub fn tone(&self) -> &str { &self.tone }
    pub fn upward_compress_db(&self) -> f32 { self.upward_compress_db }
    pub fn transient_emphasis(&self) -> f32 { self.transient_emphasis }
    pub fn transient_emphasis_bands(&self) -> &TransientEmphasisBands { &self.transient_emphasis_bands }
    pub fn stereo_independence(&self) -> &StereoIndependence { &self.stereo_independence }

    pub fn gain_reduction(&self) -> &[f32] { self.irc_limiter.gain_reduction() }
    pub fn peak_reduction_db(&self) -> f64 { self.irc_limiter.peak_reduction_db() }

    /// Get available IRC modes (for UI dropdown)
    pub fn get_irc_mode_names() -> Vec<&'static str> {
        IRCModeType::all().iter().map(|m| m.name()).collect()
    }

    /// Get available sub-modes for current IRC mode (for UI dropdown)
    pub fn get_sub_mode_names(&self) -> Vec<&'static str> {
        let mode = IRCModeType::from_str(&self.irc_mode_str);
        mode.sub_modes().iter().map(|s| s.name()).collect()
    }

    /// Get IRC mode description (for UI tooltip)
    pub fn get_irc_mode_description(&self) -> &'static str {
        IRCModeType::from_str(&self.irc_mode_str).description()
    }

    // ========== Main Processing ==========

    /// Process audio buffer in-place through the full maximizer chain.
    pub fn process(&mut self, buffer: &mut AudioBuffer, sample_rate: i32) {
        if self.bypass || buffer.is_empty() || buffer[0].is_empty() {
            return;
        }

        // Set sample rate for IRC limiter
        self.irc_limiter.set_sample_rate(sample_rate);

        // 1. Tone pre-EQ
        if self.tone != "Off" {
            self.apply_tone_eq(buffer, sample_rate);
        }

        // 2. Upward compression
        if self.upward_compress_db > 0.0 {
            self.apply_upward_compression(buffer, sample_rate);
        }

        // 3. Transient emphasis
        if self.transient_emphasis > 0.0 {
            self.apply_transient_emphasis(buffer, sample_rate);
        }

        // 4. Soft clipping
        if self.soft_clip_pct > 0.0 {
            self.apply_soft_clip(buffer);
        }

        // 5. Input gain push
        self.apply_input_gain(buffer);

        // 6. IRC Limiter (the real deal — 6 different algorithms)
        self.irc_limiter.process(buffer);
    }

    // ========== Internal DSP ==========

    fn apply_tone_eq(&mut self, buffer: &mut AudioBuffer, sample_rate: i32) {
        self.tone_eq.reset();
        match self.tone.as_str() {
            "Warm" => {
                self.tone_eq.band_mut(0).set_frequency(200.0);
                self.tone_eq.band_mut(0).set_gain(2.0);
                self.tone_eq.band_mut(0).set_type(FilterType::LowShelf);
                self.tone_eq.band_mut(0).set_enabled(true);
                self.tone_eq.band_mut(1).set_frequency(8000.0);
                self.tone_eq.band_mut(1).set_gain(-1.0);
                self.tone_eq.band_mut(1).set_type(FilterType::Peak);
                self.tone_eq.band_mut(1).set_q(1.0);
                self.tone_eq.band_mut(1).set_enabled(true);
            }
            "Bright" => {
                self.tone_eq.band_mut(0).set_frequency(8000.0);
                self.tone_eq.band_mut(0).set_gain(2.0);
                self.tone_eq.band_mut(0).set_type(FilterType::HighShelf);
                self.tone_eq.band_mut(0).set_enabled(true);
            }
            "Dark" => {
                self.tone_eq.band_mut(0).set_frequency(8000.0);
                self.tone_eq.band_mut(0).set_gain(-2.0);
                self.tone_eq.band_mut(0).set_type(FilterType::HighShelf);
                self.tone_eq.band_mut(0).set_enabled(true);
                self.tone_eq.band_mut(1).set_frequency(200.0);
                self.tone_eq.band_mut(1).set_gain(1.0);
                self.tone_eq.band_mut(1).set_type(FilterType::Peak);
                self.tone_eq.band_mut(1).set_q(1.0);
                self.tone_eq.band_mut(1).set_enabled(true);
            }
            "Air" => {
                self.tone_eq.band_mut(0).set_frequency(12000.0);
                self.tone_eq.band_mut(0).set_gain(3.0);
                self.tone_eq.band_mut(0).set_type(FilterType::HighShelf);
                self.tone_eq.band_mut(0).set_enabled(true);
            }
            "Telephone" => {
                self.tone_eq.band_mut(0).set_frequency(300.0);
                self.tone_eq.band_mut(0).set_gain(0.0);
                self.tone_eq.band_mut(0).set_type(FilterType::HighPass);
                self.tone_eq.band_mut(0).set_enabled(true);
                self.tone_eq.band_mut(1).set_frequency(3000.0);
                self.tone_eq.band_mut(1).set_gain(0.0);
                self.tone_eq.band_mut(1).set_type(FilterType::LowPass);
                self.tone_eq.band_mut(1).set_enabled(true);
            }
            _ => {}
        }
        let processed = self.tone_eq.process(buffer, sample_rate);
        *buffer = processed;
    }

    fn apply_upward_compression(&self, buffer: &mut AudioBuffer, sample_rate: i32) {
        if buffer.is_empty() || buffer[0].is_empty() { return; }
        let envelope = Self::detect_envelope(buffer, sample_rate, 10.0, 300.0);
        let threshold_linear = db_to_linear_f(-20.0);
        for ch in 0..buffer.len() {
            for i in 0..buffer[ch].len() {
                let env = envelope[i].max(1e-10);
                if env < threshold_linear {
                    let ratio = threshold_linear / env;
                    let boost_db = self.upward_compress_db * (1.0 - (ratio / 2.0).min(1.0));
                    buffer[ch][i] *= db_to_linear_f(boost_db);
                }
            }
        }
    }

    fn apply_transient_emphasis(&self, buffer: &mut AudioBuffer, sample_rate: i32) {
        if buffer.is_empty() || buffer[0].is_empty() { return; }
        let transient_env = Self::detect_envelope(buffer, sample_rate, 1.0, 50.0);
        let sustain_env = Self::detect_envelope(buffer, sample_rate, 50.0, 5.0);

        let max_t = transient_env.iter().fold(1e-10_f32, |a, &b| a.max(b));
        let max_s = sustain_env.iter().fold(1e-10_f32, |a, &b| a.max(b));

        let emphasis = self.transient_emphasis / 100.0;
        let band_emph = self.transient_emphasis_bands.mid / 100.0;

        for ch in 0..buffer.len() {
            for i in 0..buffer[ch].len() {
                let nt = transient_env[i] / max_t;
                let ns = sustain_env[i] / max_s;
                let boost = 1.0 + 3.0 * emphasis * band_emph * nt;
                let reduce = 1.0 - 0.5 * emphasis * (self.stereo_independence.sustain / 100.0) * ns;
                buffer[ch][i] *= boost * reduce;
            }
        }
    }

    fn apply_soft_clip(&self, buffer: &mut AudioBuffer) {
        if buffer.is_empty() { return; }
        // Match V2 Python formula: drive = 1.0 + amount * 2.0
        // and normalize with tanh(drive) to preserve output level
        let amount = self.soft_clip_pct / 100.0;
        let drive = 1.0 + amount * 2.0;
        let tanh_drive = drive.tanh();
        // Safety: tanh_drive is always > 0 for drive > 0
        let inv_tanh_drive = if tanh_drive > 1e-10 { 1.0 / tanh_drive } else { 1.0 };
        for ch in 0..buffer.len() {
            for i in 0..buffer[ch].len() {
                let driven = buffer[ch][i] * drive;
                buffer[ch][i] = driven.tanh() * inv_tanh_drive;
            }
        }
    }

    fn apply_input_gain(&self, buffer: &mut AudioBuffer) {
        if self.gain_db <= 0.0 { return; }
        let gain_linear = db_to_linear_f(self.gain_db);
        for ch in buffer.iter_mut() {
            for sample in ch.iter_mut() { *sample *= gain_linear; }
        }
    }

    fn detect_envelope(buffer: &AudioBuffer, sample_rate: i32, attack_ms: f32, release_ms: f32) -> Vec<f32> {
        if buffer.is_empty() || buffer[0].is_empty() { return Vec::new(); }
        let num_samples = buffer[0].len();
        let mut mixed = vec![0.0f32; num_samples];
        for channel in buffer {
            let env = Self::compute_channel_envelope(channel, sample_rate, attack_ms, release_ms);
            for i in 0..num_samples { mixed[i] = mixed[i].max(env[i]); }
        }
        mixed
    }

    fn compute_channel_envelope(channel: &[f32], sample_rate: i32, attack_ms: f32, release_ms: f32) -> Vec<f32> {
        let n = channel.len();
        let mut envelope = vec![0.0f32; n];
        let ac = (-1.0_f32 / (attack_ms * sample_rate as f32 / 1000.0)).exp();
        let rc = (-1.0_f32 / (release_ms * sample_rate as f32 / 1000.0)).exp();
        let mut state = 0.0_f32;
        for i in 0..n {
            let inp = channel[i].abs();
            if inp > state {
                state = ac * state + (1.0 - ac) * inp;
            } else {
                state = rc * state + (1.0 - rc) * inp;
            }
            envelope[i] = state;
        }
        envelope
    }

    // ========== Control ==========

    pub fn reset(&mut self) {
        self.tone_eq.reset();
        self.irc_limiter.reset();
    }

    // ========== Serialization ==========

    pub fn to_settings(&self) -> Settings {
        let mut s = Settings::new();
        s.insert("irc_mode".into(), SettingsValue::Int(self.irc_mode));
        s.insert("irc_mode_str".into(), SettingsValue::String(self.irc_mode_str.clone()));
        s.insert("sub_mode".into(), SettingsValue::String(self.sub_mode.clone()));
        s.insert("gain_db".into(), SettingsValue::Float32(self.gain_db));
        s.insert("ceiling_dbfs".into(), SettingsValue::Float32(self.ceiling_dbfs));
        s.insert("ceiling".into(), SettingsValue::Float32(self.ceiling_dbfs));
        s.insert("true_peak".into(), SettingsValue::Bool(self.true_peak));
        s.insert("character".into(), SettingsValue::Int(self.character));
        s.insert("tone".into(), SettingsValue::String(self.tone.clone()));
        s.insert("upward_compress_db".into(), SettingsValue::Float32(self.upward_compress_db));
        s.insert("soft_clip".into(), SettingsValue::Float32(self.soft_clip_pct));
        s.insert("transient_emphasis".into(), SettingsValue::Float32(self.transient_emphasis));
        s.insert("transient_emphasis_high".into(), SettingsValue::Float32(self.transient_emphasis_bands.high));
        s.insert("transient_emphasis_mid".into(), SettingsValue::Float32(self.transient_emphasis_bands.mid));
        s.insert("transient_emphasis_low".into(), SettingsValue::Float32(self.transient_emphasis_bands.low));
        s.insert("stereo_transient".into(), SettingsValue::Float32(self.stereo_independence.transient));
        s.insert("stereo_sustain".into(), SettingsValue::Float32(self.stereo_independence.sustain));
        s.insert("bypass".into(), SettingsValue::Bool(self.bypass));
        s.insert("enabled".into(), SettingsValue::Bool(!self.bypass));
        s
    }

    pub fn from_settings(&mut self, s: &Settings) {
        self.irc_mode = get_setting_int(s, "irc_mode", self.irc_mode);
        self.irc_mode_str = get_setting_string(s, "irc_mode_str", &self.irc_mode_str);
        if self.irc_mode_str == "A" {
            self.irc_mode_str = get_setting_string(s, "irc_mode", &self.irc_mode_str);
        }
        self.sub_mode = get_setting_string(s, "sub_mode", &self.sub_mode);
        self.gain_db = get_setting_f32(s, "gain_db", self.gain_db);
        self.ceiling_dbfs = get_setting_f32(s, "ceiling_dbfs", self.ceiling_dbfs);
        if let Some(c) = s.get("ceiling").and_then(|v| v.as_f32()) {
            self.ceiling_dbfs = c;
        }
        self.true_peak = get_setting_bool(s, "true_peak", self.true_peak);
        self.character = get_setting_int(s, "character", self.character);
        self.tone = get_setting_string(s, "tone", &self.tone);
        self.upward_compress_db = get_setting_f32(s, "upward_compress_db", self.upward_compress_db);
        self.soft_clip_pct = get_setting_f32(s, "soft_clip", self.soft_clip_pct);
        self.transient_emphasis = get_setting_f32(s, "transient_emphasis", self.transient_emphasis);
        self.transient_emphasis_bands.high = get_setting_f32(s, "transient_emphasis_high", 0.0);
        self.transient_emphasis_bands.mid = get_setting_f32(s, "transient_emphasis_mid", 0.0);
        self.transient_emphasis_bands.low = get_setting_f32(s, "transient_emphasis_low", 0.0);
        self.stereo_independence.transient = get_setting_f32(s, "stereo_transient", 0.0);
        self.stereo_independence.sustain = get_setting_f32(s, "stereo_sustain", 0.0);
        self.bypass = get_setting_bool(s, "bypass", self.bypass);
        if let Some(enabled) = s.get("enabled").and_then(|v| v.as_bool()) {
            self.bypass = !enabled;
        }

        // Apply IRC mode to limiter
        self.irc_limiter.set_mode(IRCModeType::from_str(&self.irc_mode_str));
        self.irc_limiter.set_sub_mode_str(&self.sub_mode);
        self.irc_limiter.set_ceiling(self.ceiling_dbfs as f64);
        self.irc_limiter.set_true_peak(self.true_peak);
    }
}

impl Default for Maximizer {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maximizer_bypass() {
        let mut max = Maximizer::new();
        max.set_bypass(true);
        let mut buffer = vec![vec![0.5f32; 100]; 2];
        let original = buffer.clone();
        max.process(&mut buffer, 44100);
        assert_eq!(buffer, original);
    }

    #[test]
    fn test_maximizer_ceiling() {
        let mut max = Maximizer::new();
        max.set_gain_db(20.0);
        max.set_ceiling(-1.0);
        let mut buffer = vec![vec![0.5f32; 2000]; 2];
        max.process(&mut buffer, 44100);
        let ceiling_linear = db_to_linear_f(-1.0);
        for ch in &buffer {
            for &sample in ch.iter().skip(500) {
                assert!(sample.abs() <= ceiling_linear + 0.05,
                    "Sample {} exceeds ceiling {}", sample, ceiling_linear);
            }
        }
    }

    #[test]
    fn test_maximizer_settings_round_trip() {
        let mut max = Maximizer::new();
        max.set_gain_db(5.0);
        max.set_ceiling(-0.5);
        max.set_irc_mode("IRC 3");
        max.set_sub_mode("Crisp");
        max.set_character(7);
        max.set_tone("Warm");
        max.set_soft_clip(50.0);
        max.set_upward_compress_db(4.0);
        max.set_transient_emphasis(30.0);

        let settings = max.to_settings();
        let mut max2 = Maximizer::new();
        max2.from_settings(&settings);

        assert!((max2.get_gain_db() - 5.0).abs() < 1e-4);
        assert!((max2.get_ceiling() - (-0.5)).abs() < 1e-4);
        assert_eq!(max2.get_irc_mode(), "IRC 3");
        assert_eq!(max2.sub_mode(), "Crisp");
        assert_eq!(max2.get_character(), 7);
        assert_eq!(max2.tone(), "Warm");
        assert!((max2.get_soft_clip() - 50.0).abs() < 1e-4);
        assert!((max2.upward_compress_db() - 4.0).abs() < 1e-4);
        assert!((max2.transient_emphasis() - 30.0).abs() < 1e-4);
    }

    #[test]
    fn test_character_range() {
        let mut max = Maximizer::new();
        max.set_character(0);
        assert_eq!(max.character(), 0);
        max.set_character(10);
        assert_eq!(max.character(), 10);
        max.set_character(15);
        assert_eq!(max.character(), 10);
    }

    #[test]
    fn test_all_irc_modes() {
        let modes = ["IRC 1", "IRC 2", "IRC 3", "IRC 4", "IRC 5", "IRC LL"];
        for mode in &modes {
            let mut max = Maximizer::new();
            max.set_irc_mode(mode);
            max.set_gain_db(6.0);
            max.set_ceiling(-1.0);

            let mut buffer = vec![vec![0.5f32; 2000]; 2];
            max.process(&mut buffer, 44100);

            // No NaN/Inf
            for ch in &buffer {
                for &s in ch {
                    assert!(s.is_finite(), "Mode {} produced non-finite sample", mode);
                }
            }
        }
    }

    #[test]
    fn test_irc3_sub_modes() {
        let subs = ["Pumping", "Balanced", "Crisp"];
        for sub in &subs {
            let mut max = Maximizer::new();
            max.set_irc_mode("IRC 3");
            max.set_sub_mode(sub);
            max.set_gain_db(6.0);
            max.set_ceiling(-1.0);

            let mut buffer = vec![vec![0.5f32; 2000]; 2];
            max.process(&mut buffer, 44100);

            for ch in &buffer {
                for &s in ch {
                    assert!(s.is_finite(), "IRC 3 {} produced non-finite", sub);
                }
            }
        }
    }

    #[test]
    fn test_irc_mode_names() {
        let names = Maximizer::get_irc_mode_names();
        assert_eq!(names.len(), 6);
        assert_eq!(names[0], "IRC 1");
        assert_eq!(names[5], "IRC LL");
    }

    #[test]
    fn test_sub_mode_names() {
        let mut max = Maximizer::new();
        max.set_irc_mode("IRC 3");
        let subs = max.get_sub_mode_names();
        assert_eq!(subs.len(), 3);
        assert_eq!(subs, vec!["Pumping", "Balanced", "Crisp"]);

        max.set_irc_mode("IRC 1");
        let subs = max.get_sub_mode_names();
        assert_eq!(subs.len(), 0); // IRC 1 has no sub-modes
    }

    #[test]
    fn test_envelope_detection() {
        let buffer = vec![vec![0.5f32; 1000]; 2];
        let env = Maximizer::detect_envelope(&buffer, 44100, 10.0, 100.0);
        assert_eq!(env.len(), 1000);
        assert!(env[999] > 0.3);
    }
}
