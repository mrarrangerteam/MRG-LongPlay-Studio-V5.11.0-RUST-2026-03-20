//! Look-ahead brickwall limiter with true peak detection.
//! Exact port from C++ limiter.cpp / limiter.h
//!
//! TruePeakDetector: 12-tap half-band FIR, 4x polyphase upsampling.
//! LookAheadLimiter: 6-stage pipeline:
//!   1. Peak envelope
//!   2. Gain reduction
//!   3. Minimum filter lookahead
//!   4. Hanning window attack smooth
//!   5. IIR release smooth with variable release
//!   6. Apply gain + brickwall clip

use std::collections::VecDeque;
use longplay_core::{AudioBuffer, Settings, SettingsValue, TWO_PI};
use longplay_core::{get_setting_f64, get_setting_bool};

// ============================================================================
// TruePeakDetector
// ============================================================================

/// 12-tap half-band FIR filter coefficients for 4x oversampling.
const FIR_TAPS: usize = 12;
const FIR_COEFFS: [f32; FIR_TAPS] = [
    0.001_708_984_4,    // tap 0
    -0.007_934_570_3,   // tap 1
    0.030_517_578,      // tap 2
    -0.091_796_875,     // tap 3
    0.316_650_39,       // tap 4
    0.5,                // tap 5 (center)
    0.316_650_39,       // tap 6
    -0.091_796_875,     // tap 7
    0.030_517_578,      // tap 8
    -0.007_934_570_3,   // tap 9
    0.001_708_984_4,    // tap 10
    0.0,                // tap 11
];

/// True Peak detector using 4x oversampling with polyphase FIR decomposition.
/// Exact port of C++ TruePeakDetector.
#[derive(Debug, Clone)]
pub struct TruePeakDetector {
    fir_history: Vec<VecDeque<f32>>,
}

impl TruePeakDetector {
    pub fn new() -> Self {
        Self { fir_history: Vec::new() }
    }

    /// Initialize detector for given number of channels.
    pub fn init(&mut self, num_channels: usize) {
        self.fir_history.clear();
        self.fir_history.resize(num_channels, VecDeque::new());
        for history in &mut self.fir_history {
            history.resize(FIR_TAPS, 0.0);
        }
    }

    /// Upsample a single sample to 4 interpolated samples using polyphase FIR.
    fn upsample4x(&mut self, sample: f32, channel: usize, out: &mut [f32; 4]) {
        let history = &mut self.fir_history[channel];
        history.pop_back();
        history.push_front(sample);

        let h0 = history[0];
        let h4 = if history.len() > 4 { history[4] } else { 0.0 };
        let h8 = if history.len() > 8 { history[8] } else { 0.0 };

        out[0] = FIR_COEFFS[0] * h0 + FIR_COEFFS[4] * h4 + FIR_COEFFS[8] * h8;
        out[1] = FIR_COEFFS[1] * h0 + FIR_COEFFS[5] * h4 + FIR_COEFFS[9] * h8;
        out[2] = FIR_COEFFS[2] * h0 + FIR_COEFFS[6] * h4 + FIR_COEFFS[10] * h8;
        out[3] = FIR_COEFFS[3] * h0 + FIR_COEFFS[7] * h4 + FIR_COEFFS[11] * h8;
    }

    #[inline]
    fn peak4x(samples: &[f32; 4]) -> f32 {
        samples.iter().fold(0.0_f32, |acc, &s| acc.max(s.abs()))
    }

    /// Measure true peak of buffer in dBTP.
    pub fn measure(&mut self, buffer: &AudioBuffer) -> f64 {
        if buffer.is_empty() || buffer[0].is_empty() {
            return -200.0;
        }
        let num_channels = buffer.len();
        let num_samples = buffer[0].len();
        let mut peak_linear = 0.0_f32;

        for ch in 0..num_channels {
            for i in 0..num_samples {
                let mut upsampled = [0.0f32; 4];
                self.upsample4x(buffer[ch][i], ch, &mut upsampled);
                peak_linear = peak_linear.max(Self::peak4x(&upsampled));
            }
        }

        if peak_linear < 1e-10 { -200.0 } else { 20.0 * (peak_linear as f64).log10() }
    }

    /// Check if buffer exceeds threshold in dBTP.
    pub fn exceeds(&mut self, buffer: &AudioBuffer, threshold_dbtp: f64) -> bool {
        self.measure(buffer) > threshold_dbtp
    }

    pub fn reset(&mut self) {
        for history in &mut self.fir_history {
            for v in history.iter_mut() { *v = 0.0; }
        }
    }
}

impl Default for TruePeakDetector {
    fn default() -> Self { Self::new() }
}

// ============================================================================
// LookAheadLimiter
// ============================================================================

/// Look-ahead brickwall limiter with true peak detection.
/// Exact port of C++ LookAheadLimiter with full 6-stage pipeline.
pub struct LookAheadLimiter {
    ceiling_db: f64,
    lookahead_ms: f64,
    release_ms: f64,
    attack_ms: f64,
    true_peak: bool,
    variable_release: bool,
    bypass: bool,
    gain_reduction_db: Vec<f32>,
    peak_reduction_db: f64,
    true_peak_detector: TruePeakDetector,
    /// Preserved release envelope state across blocks (prevents click/pop at block boundaries)
    release_state: f32,
}

impl LookAheadLimiter {
    pub fn new() -> Self {
        Self {
            ceiling_db: -1.0,
            lookahead_ms: 5.0,
            release_ms: 100.0,
            attack_ms: 5.0,
            true_peak: true,
            variable_release: true,
            bypass: false,
            gain_reduction_db: Vec::new(),
            peak_reduction_db: 0.0,
            true_peak_detector: TruePeakDetector::new(),
            release_state: 1.0,
        }
    }

    pub fn set_ceiling(&mut self, ceiling_db: f64) { self.ceiling_db = ceiling_db; }
    pub fn set_lookahead(&mut self, ms: f64) { self.lookahead_ms = ms.max(0.0); }
    pub fn set_release(&mut self, ms: f64) { self.release_ms = ms.max(1.0); }
    pub fn set_attack(&mut self, ms: f64) { self.attack_ms = ms.max(0.0); }
    pub fn set_true_peak(&mut self, enabled: bool) { self.true_peak = enabled; }
    pub fn set_variable_release(&mut self, enabled: bool) { self.variable_release = enabled; }
    pub fn set_bypass(&mut self, bypass: bool) { self.bypass = bypass; }

    pub fn ceiling(&self) -> f64 { self.ceiling_db }
    pub fn lookahead(&self) -> f64 { self.lookahead_ms }
    pub fn release(&self) -> f64 { self.release_ms }
    pub fn attack(&self) -> f64 { self.attack_ms }
    pub fn true_peak_enabled(&self) -> bool { self.true_peak }
    pub fn variable_release_enabled(&self) -> bool { self.variable_release }
    pub fn is_bypassed(&self) -> bool { self.bypass }
    pub fn gain_reduction(&self) -> &[f32] { &self.gain_reduction_db }
    pub fn peak_reduction_db(&self) -> f64 { self.peak_reduction_db }

    // Aliases for backward compatibility with simplified API
    pub fn get_gain_reduction(&self) -> &[f32] { &self.gain_reduction_db }
    pub fn get_peak_reduction_db(&self) -> f64 { self.peak_reduction_db }

    #[inline]
    fn db_to_linear(db: f64) -> f32 {
        10.0_f32.powf(db as f32 / 20.0)
    }

    #[inline]
    fn linear_to_db(linear: f32) -> f32 {
        20.0 * linear.max(1e-10).log10()
    }

    // Step 1: Peak envelope
    fn compute_peak_envelope(input: &AudioBuffer) -> Vec<f32> {
        let num_channels = input.len();
        let num_samples = input[0].len();
        let mut envelope = vec![0.0f32; num_samples];
        for i in 0..num_samples {
            let mut peak = 0.0_f32;
            for ch in 0..num_channels { peak = peak.max(input[ch][i].abs()); }
            envelope[i] = peak;
        }
        envelope
    }

    // Step 2: Gain reduction
    fn compute_gain_reduction(envelope: &[f32], ceiling_linear: f64) -> Vec<f32> {
        let ceil = ceiling_linear as f32;
        envelope.iter().map(|&e| if e > ceil { ceil / e } else { 1.0 }).collect()
    }

    // Monotonic deque minimum filter O(n)
    pub fn minimum_filter1d(input: &[f32], window_size: usize) -> Vec<f32> {
        if window_size == 0 || input.is_empty() { return input.to_vec(); }
        let mut output = vec![0.0f32; input.len()];
        let mut dq: VecDeque<usize> = VecDeque::new();
        for i in 0..input.len() {
            while !dq.is_empty() && *dq.front().unwrap() + window_size <= i { dq.pop_front(); }
            while !dq.is_empty() && input[*dq.back().unwrap()] >= input[i] { dq.pop_back(); }
            dq.push_back(i);
            output[i] = input[*dq.front().unwrap()];
        }
        output
    }

    // Step 3: Lookahead via minimum filter
    fn apply_lookahead(gr: &[f32], lookahead_samples: usize) -> Vec<f32> {
        if lookahead_samples == 0 { return gr.to_vec(); }
        Self::minimum_filter1d(gr, 2 * lookahead_samples)
    }

    // Hanning window
    fn hanning_window(size: usize) -> Vec<f32> {
        if size == 0 { return Vec::new(); }
        if size == 1 { return vec![1.0]; }
        let mut window = vec![0.0f32; size];
        for i in 0..size {
            let t = TWO_PI * i as f64 / (size - 1) as f64;
            window[i] = (0.5 * (1.0 - t.cos())) as f32;
        }
        window
    }

    // Step 4: Smooth attack with Hanning convolution
    fn smooth_attack(gr: &[f32], attack_samples: usize) -> Vec<f32> {
        if attack_samples == 0 { return gr.to_vec(); }
        let mut window = Self::hanning_window(attack_samples);
        let window_sum: f32 = window.iter().sum();
        if window_sum > 0.0 { for w in &mut window { *w /= window_sum; } }
        let mut smoothed = vec![0.0f32; gr.len()];
        let half_window = attack_samples as isize / 2;
        for i in 0..gr.len() {
            let mut sum = 0.0_f32;
            for j in 0..attack_samples {
                let idx = (i as isize + j as isize - half_window)
                    .max(0).min(gr.len() as isize - 1) as usize;
                sum += gr[idx] * window[j];
            }
            smoothed[i] = sum;
        }
        smoothed
    }

    // Step 5: Smooth release with IIR + variable release
    // Uses self.release_state to preserve envelope across block boundaries
    fn smooth_release(&mut self, gr: &[f32], sample_rate: i32) -> Vec<f32> {
        let mut smoothed = vec![0.0f32; gr.len()];
        let release_coeff = (-1.0 / (self.release_ms * sample_rate as f64 / 1000.0))
            .exp().clamp(0.0, 1.0);
        // Use stored state from previous block instead of resetting to 1.0
        let mut envelope = self.release_state;
        for i in 0..gr.len() {
            if gr[i] < envelope {
                envelope = gr[i];
            } else {
                let mut coeff = release_coeff as f32;
                if self.variable_release && gr[i] < 1.0 {
                    let gr_depth = 1.0 - gr[i];
                    let release_multiplier = 1.0 + 4.0 * gr_depth;
                    coeff = (release_coeff as f32).powf(1.0 / release_multiplier);
                }
                envelope += coeff * (gr[i] - envelope);
            }
            smoothed[i] = envelope;
        }
        // Store final state for next block
        self.release_state = envelope;
        smoothed
    }

    // Step 6: Apply gain and brickwall clip
    fn apply_gain_and_clip(
        input: &AudioBuffer, smooth_gr: &[f32], delay_samples: usize, ceiling_linear: f64,
    ) -> AudioBuffer {
        let num_channels = input.len();
        let num_samples = input[0].len();
        let mut output = vec![vec![0.0f32; num_samples]; num_channels];
        let ceiling_f = ceiling_linear as f32;
        for ch in 0..num_channels {
            for i in 0..num_samples {
                let input_idx = i as isize - delay_samples as isize;
                let input_sample = if input_idx >= 0 && (input_idx as usize) < num_samples {
                    input[ch][input_idx as usize]
                } else {
                    0.0
                };
                output[ch][i] = (input_sample * smooth_gr[i]).clamp(-ceiling_f, ceiling_f);
            }
        }
        output
    }

    /// Process entire buffer. Returns limited audio.
    /// Accepts sample_rate to compute correct lookahead/attack/release timing.
    pub fn process(&mut self, input: &AudioBuffer, sample_rate: i32) -> AudioBuffer {
        let num_channels = input.len();
        let num_samples = if input.is_empty() { 0 } else { input[0].len() };

        if self.bypass {
            self.gain_reduction_db = vec![0.0f32; num_samples];
            self.peak_reduction_db = 0.0;
            return input.clone();
        }
        if num_samples == 0 || num_channels == 0 {
            self.gain_reduction_db.clear();
            self.peak_reduction_db = 0.0;
            return input.clone();
        }

        let ceiling_linear = Self::db_to_linear(self.ceiling_db) as f64;

        if self.true_peak { self.true_peak_detector.init(num_channels); }

        // Pipeline
        let envelope = Self::compute_peak_envelope(input);
        let gain_reduction = Self::compute_gain_reduction(&envelope, ceiling_linear);
        let lookahead_samples = ((self.lookahead_ms * sample_rate as f64 / 1000.0) + 0.5) as usize;
        let lookahead_gr = Self::apply_lookahead(&gain_reduction, lookahead_samples);
        let attack_samples = ((self.attack_ms * sample_rate as f64 / 1000.0) + 0.5).max(1.0) as usize;
        let attack_smoothed = Self::smooth_attack(&lookahead_gr, attack_samples);
        let release_smoothed = self.smooth_release(&attack_smoothed, sample_rate);
        let output = Self::apply_gain_and_clip(input, &release_smoothed, lookahead_samples, ceiling_linear);

        // Metering
        self.gain_reduction_db.resize(release_smoothed.len(), 0.0);
        let mut max_gr_db = 0.0_f32;
        for (i, &gr) in release_smoothed.iter().enumerate() {
            let gr_db = Self::linear_to_db(gr);
            self.gain_reduction_db[i] = gr_db;
            max_gr_db = max_gr_db.max(-gr_db);
        }
        self.peak_reduction_db = max_gr_db as f64;

        if self.true_peak { let _ = self.true_peak_detector.measure(&output); }

        output
    }

    pub fn reset(&mut self) {
        self.gain_reduction_db.clear();
        self.peak_reduction_db = 0.0;
        self.release_state = 1.0;
        self.true_peak_detector.reset();
    }

    pub fn to_settings(&self) -> Settings {
        let mut s = Settings::new();
        s.insert("ceiling_db".into(), SettingsValue::Float(self.ceiling_db));
        s.insert("lookahead_ms".into(), SettingsValue::Float(self.lookahead_ms));
        s.insert("release_ms".into(), SettingsValue::Float(self.release_ms));
        s.insert("attack_ms".into(), SettingsValue::Float(self.attack_ms));
        s.insert("true_peak".into(), SettingsValue::Bool(self.true_peak));
        s.insert("variable_release".into(), SettingsValue::Bool(self.variable_release));
        s
    }

    pub fn from_settings(&mut self, s: &Settings) {
        self.set_ceiling(get_setting_f64(s, "ceiling_db", self.ceiling_db));
        self.set_lookahead(get_setting_f64(s, "lookahead_ms", self.lookahead_ms));
        self.set_release(get_setting_f64(s, "release_ms", self.release_ms));
        self.set_attack(get_setting_f64(s, "attack_ms", self.attack_ms));
        self.set_true_peak(get_setting_bool(s, "true_peak", self.true_peak));
        self.set_variable_release(get_setting_bool(s, "variable_release", self.variable_release));
    }
}

impl Default for LookAheadLimiter {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_true_peak_detector_silence() {
        let mut detector = TruePeakDetector::new();
        detector.init(2);
        let buffer = vec![vec![0.0f32; 100]; 2];
        let peak = detector.measure(&buffer);
        assert!(peak < -100.0);
    }

    #[test]
    fn test_limiter_bypass() {
        let mut limiter = LookAheadLimiter::new();
        limiter.set_bypass(true);
        let input = vec![vec![0.5f32; 100]; 2];
        let output = limiter.process(&input, 44100);
        assert_eq!(output, input);
    }

    #[test]
    fn test_limiter_ceiling() {
        let mut limiter = LookAheadLimiter::new();
        limiter.set_ceiling(-6.0);
        limiter.set_lookahead(0.0);
        limiter.set_attack(1.0);

        let input = vec![vec![1.0f32; 1000]; 2];
        let output = limiter.process(&input, 44100);

        let ceiling_linear = 10.0_f32.powf(-6.0 / 20.0);
        for ch in &output {
            for (i, &sample) in ch.iter().enumerate() {
                assert!(!sample.is_nan(), "NaN at index {}", i);
                if i > 10 {
                    assert!(sample.abs() <= ceiling_linear + 0.05,
                        "Sample {} at {} exceeds ceiling {}", sample, i, ceiling_linear);
                }
            }
        }
    }

    #[test]
    fn test_minimum_filter() {
        let input = vec![1.0, 0.5, 1.0, 0.3, 1.0];
        let result = LookAheadLimiter::minimum_filter1d(&input, 3);
        assert!(result[2] <= 0.5);
        assert!(result[4] <= 0.3);
    }

    #[test]
    fn test_hanning_window_symmetry() {
        let window = LookAheadLimiter::hanning_window(11);
        assert_eq!(window.len(), 11);
        for i in 0..5 {
            assert!((window[i] - window[10 - i]).abs() < 1e-6);
        }
        assert!((window[5] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_settings_round_trip() {
        let mut limiter = LookAheadLimiter::new();
        limiter.set_ceiling(-2.0);
        limiter.set_lookahead(3.0);
        limiter.set_release(200.0);
        limiter.set_attack(2.0);
        limiter.set_true_peak(false);
        limiter.set_variable_release(false);

        let settings = limiter.to_settings();
        let mut restored = LookAheadLimiter::new();
        restored.from_settings(&settings);

        assert!((restored.ceiling() - (-2.0)).abs() < 1e-10);
        assert!((restored.lookahead() - 3.0).abs() < 1e-10);
        assert!((restored.release() - 200.0).abs() < 1e-10);
        assert!((restored.attack() - 2.0).abs() < 1e-10);
        assert!(!restored.true_peak_enabled());
        assert!(!restored.variable_release_enabled());
    }
}
