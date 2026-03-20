//! Dynamics processor - exact port from C++ dynamics.cpp
//!
//! Includes EnvelopeFollower, CrossoverFilter (Linkwitz-Riley 4th order),
//! CompressorBand with soft-knee, and multiband dynamics.

use longplay_core::{AudioBuffer, Settings, SettingsValue, TWO_PI,
    get_setting_f64, get_setting_bool};

// ============================================================================
// EnvelopeFollower - exact port from C++
// ============================================================================

/// Peak-holding envelope follower with separate attack and release coefficients.
#[derive(Debug, Clone)]
pub struct EnvelopeFollower {
    attack_coeff: f64,
    release_coeff: f64,
    envelope: f64,
}

impl EnvelopeFollower {
    pub fn new() -> Self {
        Self {
            attack_coeff: 0.0,
            release_coeff: 0.0,
            envelope: 0.0,
        }
    }

    pub fn set_attack(&mut self, attack_ms: f64, sample_rate: i32) {
        let time_in_samples = attack_ms * sample_rate as f64 / 1000.0;
        self.attack_coeff = (-1.0 / time_in_samples).exp();
    }

    pub fn set_release(&mut self, release_ms: f64, sample_rate: i32) {
        let time_in_samples = release_ms * sample_rate as f64 / 1000.0;
        self.release_coeff = (-1.0 / time_in_samples).exp();
    }

    /// Process a single input sample and return the envelope value.
    /// Exact port of C++ EnvelopeFollower::process.
    pub fn process(&mut self, input: f64) -> f64 {
        let abs_input = input.abs();

        if abs_input > self.envelope {
            // Attack: rising towards input
            self.envelope = self.attack_coeff * self.envelope + (1.0 - self.attack_coeff) * abs_input;
        } else {
            // Release: falling from input
            self.envelope = self.release_coeff * self.envelope + (1.0 - self.release_coeff) * abs_input;
        }

        self.envelope
    }

    pub fn reset(&mut self) {
        self.envelope = 0.0;
    }
}

impl Default for EnvelopeFollower {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// CrossoverFilter (Linkwitz-Riley 4th order = 2x cascaded Butterworth)
// ============================================================================

/// Linkwitz-Riley 4th order crossover filter.
/// Implemented as 2 cascaded 2nd-order Butterworth sections.
/// Exact port of C++ CrossoverFilter.
#[derive(Debug, Clone)]
pub struct CrossoverFilter {
    // Normalized LP coefficients
    lp_b0: f64,
    lp_b1: f64,
    lp_b2: f64,
    lp_a1: f64,
    lp_a2: f64,
    // Normalized HP coefficients
    hp_b0: f64,
    hp_b1: f64,
    hp_b2: f64,
    hp_a1: f64,
    hp_a2: f64,
    // State: 2 cascaded sections, each with z1, z2
    lp_z1: [f64; 2],
    lp_z2: [f64; 2],
    hp_z1: [f64; 2],
    hp_z2: [f64; 2],
}

impl CrossoverFilter {
    pub fn new() -> Self {
        Self {
            lp_b0: 0.0, lp_b1: 0.0, lp_b2: 0.0, lp_a1: 0.0, lp_a2: 0.0,
            hp_b0: 0.0, hp_b1: 0.0, hp_b2: 0.0, hp_a1: 0.0, hp_a2: 0.0,
            lp_z1: [0.0; 2], lp_z2: [0.0; 2],
            hp_z1: [0.0; 2], hp_z2: [0.0; 2],
        }
    }

    pub fn reset(&mut self) {
        self.lp_z1 = [0.0; 2];
        self.lp_z2 = [0.0; 2];
        self.hp_z1 = [0.0; 2];
        self.hp_z2 = [0.0; 2];
    }

    /// Set crossover frequency. Exact port of C++ CrossoverFilter::set_frequency.
    pub fn set_frequency(&mut self, freq: f64, sample_rate: i32) {
        let wc = TWO_PI * freq / sample_rate as f64;
        let c = wc.cos();
        let s = wc.sin();
        let alpha = s / 2.0_f64.sqrt();

        // Butterworth LPF coefficients
        self.lp_b0 = (1.0 - c) / 2.0;
        self.lp_b1 = 1.0 - c;
        self.lp_b2 = (1.0 - c) / 2.0;
        self.lp_a1 = -2.0 * c;
        self.lp_a2 = 1.0 - alpha;

        let a0 = 1.0 + alpha;
        self.lp_b0 /= a0;
        self.lp_b1 /= a0;
        self.lp_b2 /= a0;
        self.lp_a1 /= a0;
        self.lp_a2 /= a0;

        // Butterworth HPF coefficients
        self.hp_b0 = (1.0 + c) / 2.0;
        self.hp_b1 = -(1.0 + c);
        self.hp_b2 = (1.0 + c) / 2.0;
        self.hp_a1 = -2.0 * c;
        self.hp_a2 = 1.0 - alpha;

        let a0 = 1.0 + alpha;
        self.hp_b0 /= a0;
        self.hp_b1 /= a0;
        self.hp_b2 /= a0;
        self.hp_a1 /= a0;
        self.hp_a2 /= a0;
    }

    /// Process a single biquad section (Direct Form II Transposed).
    /// Exact port of C++ CrossoverFilter::process_section.
    ///
    /// NOTE: The C++ version uses negated a1/a2 convention internally
    /// (output = b0*input + z1; z1 = b1*input + a1*output + z2; z2 = b2*input + a2*output).
    /// This matches the C++ code exactly.
    fn process_section(
        input: f32,
        b0: f64, b1: f64, b2: f64,
        a1: f64, a2: f64,
        z1: &mut f64, z2: &mut f64,
    ) -> f32 {
        let input_d = input as f64;
        let output = b0 * input_d + *z1;
        *z1 = b1 * input_d - a1 * output + *z2;
        *z2 = b2 * input_d - a2 * output;
        output as f32
    }

    /// Process a single sample through both LP and HP paths (2 cascaded sections each).
    /// Exact port of C++ CrossoverFilter::process.
    pub fn process(&mut self, input: f32) -> (f32, f32) {
        // Cascaded LP sections (LR4)
        let mut lp_out = input;
        lp_out = Self::process_section(
            lp_out,
            self.lp_b0, self.lp_b1, self.lp_b2, self.lp_a1, self.lp_a2,
            &mut self.lp_z1[0], &mut self.lp_z2[0],
        );
        lp_out = Self::process_section(
            lp_out,
            self.lp_b0, self.lp_b1, self.lp_b2, self.lp_a1, self.lp_a2,
            &mut self.lp_z1[1], &mut self.lp_z2[1],
        );

        // Cascaded HP sections (LR4)
        let mut hp_out = input;
        hp_out = Self::process_section(
            hp_out,
            self.hp_b0, self.hp_b1, self.hp_b2, self.hp_a1, self.hp_a2,
            &mut self.hp_z1[0], &mut self.hp_z2[0],
        );
        hp_out = Self::process_section(
            hp_out,
            self.hp_b0, self.hp_b1, self.hp_b2, self.hp_a1, self.hp_a2,
            &mut self.hp_z1[1], &mut self.hp_z2[1],
        );

        (lp_out, hp_out)
    }
}

impl Default for CrossoverFilter {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// CompressorBand - exact port from C++
// ============================================================================

/// Single compressor band with soft-knee, envelope followers, and parallel mix.
#[derive(Debug, Clone)]
pub struct CompressorBand {
    threshold_db: f64,
    ratio: f64,
    attack_ms: f64,
    release_ms: f64,
    makeup_gain_db: f64,
    knee_db: f64,
    sidechain_hpf: f64,
    parallel_mix: f64,
    envelopes: Vec<EnvelopeFollower>,
    current_gr_db: f64,
}

impl CompressorBand {
    pub fn new() -> Self {
        Self {
            threshold_db: -15.0,
            ratio: 2.0,
            attack_ms: 10.0,
            release_ms: 100.0,
            makeup_gain_db: 0.0,
            knee_db: 6.0,
            sidechain_hpf: 0.0,
            parallel_mix: 1.0,
            envelopes: Vec::new(),
            current_gr_db: 0.0,
        }
    }

    pub fn set_threshold(&mut self, threshold_db: f64) { self.threshold_db = threshold_db; }
    pub fn set_ratio(&mut self, ratio: f64) { self.ratio = ratio.max(1.0); }
    pub fn set_attack(&mut self, attack_ms: f64) { self.attack_ms = attack_ms.max(0.1); }
    pub fn set_release(&mut self, release_ms: f64) { self.release_ms = release_ms.max(1.0); }
    pub fn set_makeup_gain(&mut self, gain_db: f64) { self.makeup_gain_db = gain_db; }
    pub fn set_knee(&mut self, knee_db: f64) { self.knee_db = knee_db.max(0.0); }
    pub fn set_sidechain_hpf(&mut self, freq_hz: f64) { self.sidechain_hpf = freq_hz; }
    pub fn set_parallel_mix(&mut self, mix: f64) { self.parallel_mix = mix.clamp(0.0, 1.0); }

    pub fn current_gr_db(&self) -> f64 { self.current_gr_db }

    /// Compute gain reduction in dB for a given input level.
    /// Exact port of C++ CompressorBand::compute_gain_db.
    pub fn compute_gain_db(&self, input_db: f64) -> f64 {
        let threshold = self.threshold_db;
        let knee_half = self.knee_db / 2.0;

        if input_db < (threshold - knee_half) {
            // Below knee region: no compression
            return 0.0;
        }

        if input_db > (threshold + knee_half) {
            // Above knee region: full compression
            let excess = input_db - threshold;
            return -(excess * (1.0 - 1.0 / self.ratio));
        }

        // Inside knee region: smooth interpolation
        let knee_range = knee_half * 2.0;
        let knee_pos = (input_db - (threshold - knee_half)) / knee_range;
        let knee_pos_squared = knee_pos * knee_pos;
        let compression_above_knee = knee_pos_squared * (1.0 - 1.0 / self.ratio) * knee_half;
        -compression_above_knee
    }

    /// Process audio buffer. Exact port of C++ CompressorBand::process.
    pub fn process(&mut self, input: &AudioBuffer, sample_rate: i32) -> AudioBuffer {
        let num_channels = input.len();
        let num_samples = if num_channels > 0 { input[0].len() } else { 0 };

        // Initialize envelopes if needed
        if self.envelopes.len() != num_channels {
            self.envelopes.resize(num_channels, EnvelopeFollower::new());
        }

        // Update envelopes with timing
        for env in &mut self.envelopes {
            env.set_attack(self.attack_ms, sample_rate);
            env.set_release(self.release_ms, sample_rate);
        }

        let mut output = vec![vec![0.0f32; num_samples]; num_channels];
        let dry = input.clone();

        let mut max_gr = 0.0_f64;

        for s in 0..num_samples {
            let mut sidechain_max = 0.0_f64;

            for c in 0..num_channels {
                let sample = input[c][s] as f64;
                let level = self.envelopes[c].process(sample);
                sidechain_max = sidechain_max.max(level);
            }

            // Convert to dB
            let sidechain_db = if sidechain_max > 1e-8 {
                20.0 * sidechain_max.log10()
            } else {
                -160.0
            };

            // Compute gain reduction
            let gain_reduction_db = self.compute_gain_db(sidechain_db);
            let gain_linear = 10.0_f64.powf((gain_reduction_db + self.makeup_gain_db) / 20.0);
            max_gr = max_gr.max(gain_reduction_db.abs());

            // Apply gain to all channels
            for c in 0..num_channels {
                let wet = input[c][s] as f64 * gain_linear;
                let out = dry[c][s] as f64 * (1.0 - self.parallel_mix) + wet * self.parallel_mix;
                output[c][s] = out as f32;
            }
        }

        self.current_gr_db = max_gr;
        output
    }

    pub fn reset(&mut self) {
        for env in &mut self.envelopes {
            env.reset();
        }
        self.current_gr_db = 0.0;
    }

    pub fn to_settings(&self) -> Settings {
        let mut s = Settings::new();
        s.insert("threshold".into(), SettingsValue::Float(self.threshold_db));
        s.insert("ratio".into(), SettingsValue::Float(self.ratio));
        s.insert("attack".into(), SettingsValue::Float(self.attack_ms));
        s.insert("release".into(), SettingsValue::Float(self.release_ms));
        s.insert("makeup_gain".into(), SettingsValue::Float(self.makeup_gain_db));
        s.insert("knee".into(), SettingsValue::Float(self.knee_db));
        s.insert("sidechain_hpf".into(), SettingsValue::Float(self.sidechain_hpf));
        s.insert("parallel_mix".into(), SettingsValue::Float(self.parallel_mix));
        s
    }

    pub fn from_settings(s: &Settings) -> Self {
        let mut band = CompressorBand::new();
        band.set_threshold(get_setting_f64(s, "threshold", -15.0));
        band.set_ratio(get_setting_f64(s, "ratio", 2.0));
        band.set_attack(get_setting_f64(s, "attack", 10.0));
        band.set_release(get_setting_f64(s, "release", 100.0));
        band.set_makeup_gain(get_setting_f64(s, "makeup_gain", 0.0));
        band.set_knee(get_setting_f64(s, "knee", 6.0));
        band.set_sidechain_hpf(get_setting_f64(s, "sidechain_hpf", 0.0));
        band.set_parallel_mix(get_setting_f64(s, "parallel_mix", 1.0));
        band
    }
}

impl Default for CompressorBand {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// DynamicsPreset
// ============================================================================

/// Dynamics processor preset definition.
#[derive(Debug, Clone)]
pub struct DynamicsPreset {
    pub name: String,
    pub threshold: f64,
    pub ratio: f64,
    pub attack_ms: f64,
    pub release_ms: f64,
    pub makeup_gain: f64,
    pub knee: f64,
    pub parallel_mix: f64,
    pub multiband: bool,
}

// ============================================================================
// Dynamics - main processor. Exact port of C++ Dynamics class.
// ============================================================================

/// Dynamics processor with single-band and multiband (3-band) modes.
pub struct Dynamics {
    single_band: CompressorBand,
    bands: [CompressorBand; 3],
    multiband: bool,
    bypass: bool,
    crossover_low: f64,
    crossover_high: f64,
    xover_low: Vec<CrossoverFilter>,
    xover_high: Vec<CrossoverFilter>,
}

impl Dynamics {
    pub fn new() -> Self {
        Self {
            single_band: CompressorBand::new(),
            bands: [CompressorBand::new(), CompressorBand::new(), CompressorBand::new()],
            multiband: false,
            bypass: false,
            crossover_low: 200.0,
            crossover_high: 4000.0,
            xover_low: Vec::new(),
            xover_high: Vec::new(),
        }
    }

    pub fn single_band(&self) -> &CompressorBand { &self.single_band }
    pub fn single_band_mut(&mut self) -> &mut CompressorBand { &mut self.single_band }

    pub fn band(&self, index: usize) -> &CompressorBand { &self.bands[index] }
    pub fn band_mut(&mut self, index: usize) -> &mut CompressorBand { &mut self.bands[index] }

    pub fn set_multiband(&mut self, enabled: bool) { self.multiband = enabled; }
    pub fn is_multiband(&self) -> bool { self.multiband }

    pub fn set_bypass(&mut self, bypass: bool) { self.bypass = bypass; }
    pub fn is_bypassed(&self) -> bool { self.bypass }

    pub fn set_crossover_low(&mut self, freq: f64) {
        self.crossover_low = freq.max(20.0);
    }

    pub fn set_crossover_high(&mut self, freq: f64) {
        self.crossover_high = freq.max(self.crossover_low + 100.0);
    }

    /// Process audio buffer. Exact port of C++ Dynamics::process.
    pub fn process(&mut self, input: &AudioBuffer, sample_rate: i32) -> AudioBuffer {
        if self.bypass {
            return input.clone();
        }

        if !self.multiband {
            return self.single_band.process(input, sample_rate);
        }

        // Multiband processing
        let num_channels = input.len();
        let num_samples = input[0].len();

        // Initialize crossover filters if needed
        if self.xover_low.len() != num_channels {
            self.xover_low.resize(num_channels, CrossoverFilter::new());
            self.xover_high.resize(num_channels, CrossoverFilter::new());
        }

        // Set crossover frequencies
        for xover in &mut self.xover_low {
            xover.set_frequency(self.crossover_low, sample_rate);
        }
        for xover in &mut self.xover_high {
            xover.set_frequency(self.crossover_high, sample_rate);
        }

        let mut output = vec![vec![0.0f32; num_samples]; num_channels];

        // Process each sample
        for s in 0..num_samples {
            // Create mini-buffers for band splitting
            let mut low_buffer = vec![vec![0.0f32; 1]; num_channels];
            let mut mid_buffer = vec![vec![0.0f32; 1]; num_channels];
            let mut high_buffer = vec![vec![0.0f32; 1]; num_channels];

            // Split into bands
            for c in 0..num_channels {
                let input_sample = input[c][s];

                // First crossover: split into low and mid+high
                let (low, mid_high) = self.xover_low[c].process(input_sample);

                // Second crossover: split mid+high into mid and high
                let (mid, high) = self.xover_high[c].process(mid_high);

                low_buffer[c][0] = low;
                mid_buffer[c][0] = mid;
                high_buffer[c][0] = high;
            }

            // Process each band
            let low_processed = self.bands[0].process(&low_buffer, sample_rate);
            let mid_processed = self.bands[1].process(&mid_buffer, sample_rate);
            let high_processed = self.bands[2].process(&high_buffer, sample_rate);

            // Sum bands
            for c in 0..num_channels {
                output[c][s] = low_processed[c][0] + mid_processed[c][0] + high_processed[c][0];
            }
        }

        output
    }

    /// Apply a preset by name. Exact port of C++ Dynamics::apply_preset.
    pub fn apply_preset(&mut self, name: &str) {
        let preset = match get_dynamics_preset(name) {
            Some(p) => p,
            None => return,
        };

        // Apply to single band
        self.single_band.set_threshold(preset.threshold);
        self.single_band.set_ratio(preset.ratio);
        self.single_band.set_attack(preset.attack_ms);
        self.single_band.set_release(preset.release_ms);
        self.single_band.set_makeup_gain(preset.makeup_gain);
        self.single_band.set_knee(preset.knee);
        self.single_band.set_parallel_mix(preset.parallel_mix);

        // Apply to all multiband bands
        for band in &mut self.bands {
            band.set_threshold(preset.threshold);
            band.set_ratio(preset.ratio);
            band.set_attack(preset.attack_ms);
            band.set_release(preset.release_ms);
            band.set_makeup_gain(preset.makeup_gain);
            band.set_knee(preset.knee);
            band.set_parallel_mix(preset.parallel_mix);
        }

        self.set_multiband(preset.multiband);
    }

    pub fn get_preset_names() -> Vec<String> {
        build_dynamics_presets().into_iter().map(|p| p.name).collect()
    }

    pub fn reset(&mut self) {
        self.single_band.reset();
        for band in &mut self.bands {
            band.reset();
        }
        for xover in &mut self.xover_low {
            xover.reset();
        }
        for xover in &mut self.xover_high {
            xover.reset();
        }
    }

    pub fn to_settings(&self) -> Settings {
        let mut s = Settings::new();
        s.insert("multiband".into(), SettingsValue::Bool(self.multiband));
        s.insert("bypass".into(), SettingsValue::Bool(self.bypass));
        s.insert("crossover_low".into(), SettingsValue::Float(self.crossover_low));
        s.insert("crossover_high".into(), SettingsValue::Float(self.crossover_high));

        // Single band settings
        let single_settings = self.single_band.to_settings();
        for (key, value) in single_settings {
            s.insert(format!("single_{}", key), value);
        }

        // Multiband settings
        let prefixes = ["low_", "mid_", "high_"];
        for (i, prefix) in prefixes.iter().enumerate() {
            let band_settings = self.bands[i].to_settings();
            for (key, value) in band_settings {
                s.insert(format!("{}{}", prefix, key), value);
            }
        }

        s
    }

    pub fn from_settings(&mut self, s: &Settings) {
        self.multiband = get_setting_bool(s, "multiband", self.multiband);
        self.bypass = get_setting_bool(s, "bypass", self.bypass);
        self.crossover_low = get_setting_f64(s, "crossover_low", self.crossover_low);
        self.crossover_high = get_setting_f64(s, "crossover_high", self.crossover_high);

        // Single band settings
        let mut single_settings = Settings::new();
        for (key, value) in s {
            if let Some(stripped) = key.strip_prefix("single_") {
                single_settings.insert(stripped.to_string(), value.clone());
            }
        }
        if !single_settings.is_empty() {
            self.single_band = CompressorBand::from_settings(&single_settings);
        }

        // Multiband settings
        let prefixes = ["low_", "mid_", "high_"];
        for (i, prefix) in prefixes.iter().enumerate() {
            let mut band_settings = Settings::new();
            for (key, value) in s {
                if let Some(stripped) = key.strip_prefix(prefix) {
                    band_settings.insert(stripped.to_string(), value.clone());
                }
            }
            if !band_settings.is_empty() {
                self.bands[i] = CompressorBand::from_settings(&band_settings);
            }
        }
    }
}

impl Default for Dynamics {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Dynamics Presets - exact port of C++ Dynamics::init_presets()
// ============================================================================

fn build_dynamics_presets() -> Vec<DynamicsPreset> {
    vec![
        DynamicsPreset {
            name: "Gentle Glue".into(),
            threshold: -15.0, ratio: 2.0, attack_ms: 20.0, release_ms: 200.0,
            makeup_gain: 2.0, knee: 10.0, parallel_mix: 1.0, multiband: false,
        },
        DynamicsPreset {
            name: "Medium Compression".into(),
            threshold: -18.0, ratio: 3.0, attack_ms: 10.0, release_ms: 150.0,
            makeup_gain: 4.0, knee: 6.0, parallel_mix: 1.0, multiband: false,
        },
        DynamicsPreset {
            name: "Heavy Compression".into(),
            threshold: -24.0, ratio: 6.0, attack_ms: 5.0, release_ms: 100.0,
            makeup_gain: 8.0, knee: 3.0, parallel_mix: 1.0, multiband: false,
        },
        DynamicsPreset {
            name: "Parallel Punch".into(),
            threshold: -30.0, ratio: 8.0, attack_ms: 1.0, release_ms: 50.0,
            makeup_gain: 0.0, knee: 3.0, parallel_mix: 0.5, multiband: false,
        },
        DynamicsPreset {
            name: "Bus Glue".into(),
            threshold: -12.0, ratio: 2.0, attack_ms: 30.0, release_ms: 300.0,
            makeup_gain: 1.0, knee: 10.0, parallel_mix: 1.0, multiband: false,
        },
        DynamicsPreset {
            name: "Vocal Control".into(),
            threshold: -20.0, ratio: 3.5, attack_ms: 5.0, release_ms: 80.0,
            makeup_gain: 3.0, knee: 6.0, parallel_mix: 1.0, multiband: false,
        },
        DynamicsPreset {
            name: "Drum Smash".into(),
            threshold: -25.0, ratio: 10.0, attack_ms: 0.5, release_ms: 40.0,
            makeup_gain: 6.0, knee: 0.0, parallel_mix: 0.7, multiband: false,
        },
        DynamicsPreset {
            name: "Transparent".into(),
            threshold: -16.0, ratio: 1.5, attack_ms: 30.0, release_ms: 250.0,
            makeup_gain: 1.0, knee: 12.0, parallel_mix: 1.0, multiband: false,
        },
    ]
}

/// Get a dynamics preset by name.
pub fn get_dynamics_preset(name: &str) -> Option<DynamicsPreset> {
    build_dynamics_presets().into_iter().find(|p| p.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_envelope_follower_attack() {
        let mut env = EnvelopeFollower::new();
        env.set_attack(1.0, 44100);
        env.set_release(100.0, 44100);

        // Feed a step input — 1ms attack at 44100Hz = ~44 samples time constant.
        // After 100 samples (~2.3 time constants), envelope should be well above 0.
        let mut last = 0.0;
        for _ in 0..500 {
            last = env.process(1.0);
        }
        // After 500 samples (~11 time constants), should be very close to 1.0
        assert!(last > 0.9, "Envelope was {}, expected > 0.9", last);
    }

    #[test]
    fn test_compressor_no_reduction_below_threshold() {
        let comp = CompressorBand::new();
        // Well below threshold of -15 dB
        let gr = comp.compute_gain_db(-40.0);
        assert!((gr - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_dynamics_preset_count() {
        let names = Dynamics::get_preset_names();
        assert_eq!(names.len(), 8);
    }
}
