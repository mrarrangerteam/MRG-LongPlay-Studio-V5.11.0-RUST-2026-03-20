//! Equalizer implementation - exact port from C++ equalizer.cpp
//!
//! 8-band parametric EQ with tone presets.

use crate::biquad::{calc_biquad_coeffs, BiquadFilter, FilterType};
use longplay_core::{AudioBuffer, Settings, SettingsValue, get_setting_f64, get_setting_bool, get_setting_string, PI};

/// Number of EQ bands
pub const NUM_BANDS: usize = 8;

/// Standard frequencies for the 8 bands
const STANDARD_FREQUENCIES: [f64; NUM_BANDS] = [30.0, 64.0, 125.0, 250.0, 1000.0, 4000.0, 8000.0, 16000.0];

/// Band definition for tone presets
#[derive(Debug, Clone)]
pub struct BandDef {
    pub freq: f64,
    pub gain: f64,
    pub q: f64,
    pub filter_type: FilterType,
}

/// Tone EQ preset
#[derive(Debug, Clone)]
pub struct ToneEQPreset {
    pub name: String,
    pub bands: Vec<BandDef>,
}

/// Single EQ band with frequency, gain, Q, filter type, and per-channel biquad filters.
#[derive(Debug, Clone)]
pub struct EQBand {
    frequency: f64,
    gain_db: f64,
    q: f64,
    filter_type: FilterType,
    enabled: bool,
    filters: Vec<BiquadFilter>,
    last_sample_rate: i32,
}

impl EQBand {
    pub fn new(frequency: f64, gain_db: f64, q: f64, filter_type: FilterType, enabled: bool) -> Self {
        Self {
            frequency,
            gain_db,
            q,
            filter_type,
            enabled,
            filters: Vec::new(),
            last_sample_rate: 0,
        }
    }

    pub fn frequency(&self) -> f64 { self.frequency }
    pub fn gain_db(&self) -> f64 { self.gain_db }
    pub fn q(&self) -> f64 { self.q }
    pub fn filter_type(&self) -> FilterType { self.filter_type }
    pub fn is_enabled(&self) -> bool { self.enabled }

    pub fn set_frequency(&mut self, freq: f64) {
        assert!(freq > 0.0, "Frequency must be positive");
        self.frequency = freq;
        self.last_sample_rate = 0; // force recalculation
    }

    pub fn set_gain(&mut self, gain_db: f64) {
        self.gain_db = gain_db;
        self.last_sample_rate = 0;
    }

    pub fn set_q(&mut self, q: f64) {
        assert!(q > 0.0, "Q must be positive");
        self.q = q;
        self.last_sample_rate = 0;
    }

    pub fn set_type(&mut self, filter_type: FilterType) {
        self.filter_type = filter_type;
        self.last_sample_rate = 0;
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Process audio buffer in-place. Exact port of C++ EQBand::process.
    pub fn process(&mut self, buffer: &mut AudioBuffer, sample_rate: i32) {
        if !self.enabled || self.gain_db == 0.0 {
            return;
        }

        let num_channels = buffer.len();

        // Ensure we have filters for all channels
        if self.filters.len() != num_channels {
            self.filters.clear();
            for _ in 0..num_channels {
                self.filters.push(BiquadFilter::new());
            }
            self.last_sample_rate = 0;
        }

        // Update coefficients if sample rate changed
        if sample_rate != self.last_sample_rate {
            let coeffs = calc_biquad_coeffs(
                self.filter_type,
                self.frequency,
                self.gain_db,
                self.q,
                sample_rate,
            );
            for filter in &mut self.filters {
                filter.set_coefficients(&coeffs);
            }
            self.last_sample_rate = sample_rate;
        }

        // Apply filter to each channel
        for (ch, channel) in buffer.iter_mut().enumerate() {
            let filter = &mut self.filters[ch];
            for sample in channel.iter_mut() {
                *sample = filter.process_sample(*sample);
            }
        }
    }

    pub fn reset(&mut self) {
        for filter in &mut self.filters {
            filter.reset();
        }
    }

    /// Compute frequency response magnitude in dB at a given frequency.
    pub fn get_response_db(&self, freq: f64, sample_rate: i32) -> f64 {
        if !self.enabled {
            return 0.0;
        }

        let coeffs = calc_biquad_coeffs(self.filter_type, self.frequency, self.gain_db, self.q, sample_rate);

        let w = 2.0 * PI * freq / sample_rate as f64;
        let cos_w = w.cos();
        let sin_w = w.sin();

        // Transfer function: H(z) = (b0 + b1*z^-1 + b2*z^-2) / (a0 + a1*z^-1 + a2*z^-2)
        let b_real = coeffs.b0 + coeffs.b1 * cos_w + coeffs.b2 * cos_w * 2.0;
        let b_imag = -coeffs.b1 * sin_w - coeffs.b2 * sin_w * 2.0;
        let b_mag_sq = b_real * b_real + b_imag * b_imag;

        let a_real = coeffs.a0 + coeffs.a1 * cos_w + coeffs.a2 * cos_w * 2.0;
        let a_imag = -coeffs.a1 * sin_w - coeffs.a2 * sin_w * 2.0;
        let a_mag_sq = a_real * a_real + a_imag * a_imag;

        let mag = (b_mag_sq / a_mag_sq).sqrt();
        20.0 * (mag + 1e-10).log10()
    }

    pub fn to_settings(&self) -> Settings {
        let mut s = Settings::new();
        s.insert("frequency".into(), SettingsValue::Float(self.frequency));
        s.insert("gain_db".into(), SettingsValue::Float(self.gain_db));
        s.insert("q".into(), SettingsValue::Float(self.q));
        s.insert("type".into(), SettingsValue::String(self.filter_type.to_string_id().to_string()));
        s.insert("enabled".into(), SettingsValue::Bool(self.enabled));
        s
    }

    pub fn from_settings(s: &Settings) -> Self {
        let freq = get_setting_f64(s, "frequency", 1000.0);
        let gain = get_setting_f64(s, "gain_db", 0.0);
        let q = get_setting_f64(s, "q", 1.0);
        let type_str = get_setting_string(s, "type", "peak");
        let filter_type = FilterType::from_string_id(&type_str);
        let enabled = get_setting_bool(s, "enabled", true);

        EQBand::new(freq, gain, q, filter_type, enabled)
    }
}

/// 8-band parametric equalizer with tone presets.
/// Exact port of C++ Equalizer class.
pub struct Equalizer {
    bands: [EQBand; NUM_BANDS],
    bypass: bool,
}

impl Equalizer {
    pub fn new() -> Self {
        let bands = std::array::from_fn(|i| {
            EQBand::new(STANDARD_FREQUENCIES[i], 0.0, 0.7, FilterType::Peak, true)
        });
        Self {
            bands,
            bypass: false,
        }
    }

    pub fn band(&self, index: usize) -> &EQBand {
        assert!(index < NUM_BANDS, "Band index out of range");
        &self.bands[index]
    }

    pub fn band_mut(&mut self, index: usize) -> &mut EQBand {
        assert!(index < NUM_BANDS, "Band index out of range");
        &mut self.bands[index]
    }

    pub fn set_bypass(&mut self, bypass: bool) {
        self.bypass = bypass;
    }

    pub fn is_bypassed(&self) -> bool {
        self.bypass
    }

    /// Process audio buffer. Returns a new buffer (matches C++ signature).
    pub fn process(&mut self, input: &AudioBuffer, sample_rate: i32) -> AudioBuffer {
        let mut output = input.clone();

        if self.bypass {
            return output;
        }

        for band in &mut self.bands {
            band.process(&mut output, sample_rate);
        }

        output
    }

    /// Process audio buffer in-place.
    pub fn process_in_place(&mut self, buffer: &mut AudioBuffer, sample_rate: i32) {
        if self.bypass {
            return;
        }
        for band in &mut self.bands {
            band.process(buffer, sample_rate);
        }
    }

    pub fn reset(&mut self) {
        for band in &mut self.bands {
            band.reset();
        }
    }

    /// Apply a tone preset by name.
    pub fn apply_tone_preset(&mut self, preset_name: &str) {
        let preset = get_tone_preset(preset_name)
            .unwrap_or_else(|| panic!("Unknown tone preset: {}", preset_name));

        for (i, bd) in preset.bands.iter().enumerate() {
            if i >= NUM_BANDS {
                break;
            }
            self.bands[i].set_frequency(bd.freq);
            self.bands[i].set_gain(bd.gain);
            self.bands[i].set_q(bd.q);
            self.bands[i].set_type(bd.filter_type);
            self.bands[i].set_enabled(true);
        }
    }

    /// Get list of available tone preset names.
    pub fn get_tone_preset_names() -> Vec<String> {
        let presets = build_tone_presets();
        let mut names: Vec<String> = presets.into_iter().map(|p| p.name).collect();
        names.sort();
        names
    }

    /// Compute frequency response across all bands.
    pub fn get_frequency_response(&self, sample_rate: i32, num_points: usize) -> Vec<(f64, f64)> {
        let mut response = Vec::with_capacity(num_points);

        let min_freq = 20.0_f64;
        let max_freq = sample_rate as f64 / 2.0;
        let log_min = min_freq.log10();
        let log_max = max_freq.log10();

        for i in 0..num_points {
            let t = i as f64 / (num_points - 1) as f64;
            let log_freq = log_min + t * (log_max - log_min);
            let freq = 10.0_f64.powf(log_freq);

            let mut mag_db = 0.0;
            for band in &self.bands {
                mag_db += band.get_response_db(freq, sample_rate);
            }

            response.push((freq, mag_db));
        }

        response
    }

    pub fn to_settings(&self) -> Settings {
        let mut result = Settings::new();
        result.insert("bypass".into(), SettingsValue::Bool(self.bypass));

        for i in 0..NUM_BANDS {
            let prefix = format!("band_{}_", i);
            let band_settings = self.bands[i].to_settings();
            for (key, value) in band_settings {
                result.insert(format!("{}{}", prefix, key), value);
            }
        }

        result
    }

    pub fn from_settings(&mut self, s: &Settings) {
        self.bypass = get_setting_bool(s, "bypass", false);

        for i in 0..NUM_BANDS {
            let prefix = format!("band_{}_", i);
            let mut band_settings = Settings::new();
            for (key, value) in s {
                if key.starts_with(&prefix) {
                    band_settings.insert(key[prefix.len()..].to_string(), value.clone());
                }
            }
            if !band_settings.is_empty() {
                self.bands[i] = EQBand::from_settings(&band_settings);
            }
        }
    }
}

impl Default for Equalizer {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tone Presets - exact port of C++ Equalizer::init_presets()
// ============================================================================

fn bd(freq: f64, gain: f64, q: f64, ft: FilterType) -> BandDef {
    BandDef { freq, gain, q, filter_type: ft }
}

fn build_tone_presets() -> Vec<ToneEQPreset> {
    use FilterType::*;

    vec![
        ToneEQPreset {
            name: "Flat".into(),
            bands: vec![
                bd(30.0, 0.0, 0.7, Peak), bd(64.0, 0.0, 0.7, Peak),
                bd(125.0, 0.0, 0.7, Peak), bd(250.0, 0.0, 0.7, Peak),
                bd(1000.0, 0.0, 0.7, Peak), bd(4000.0, 0.0, 0.7, Peak),
                bd(8000.0, 0.0, 0.7, Peak), bd(16000.0, 0.0, 0.7, Peak),
            ],
        },
        ToneEQPreset {
            name: "Warm".into(),
            bands: vec![
                bd(30.0, 0.0, 0.7, Peak), bd(64.0, 0.0, 0.7, Peak),
                bd(125.0, 0.0, 0.7, Peak), bd(250.0, 0.0, 0.7, Peak),
                bd(1000.0, 0.0, 0.7, Peak), bd(4000.0, -1.0, 0.7, Peak),
                bd(8000.0, -2.0, 0.7, Peak), bd(16000.0, 0.0, 0.7, Peak),
            ],
        },
        ToneEQPreset {
            name: "Bright".into(),
            bands: vec![
                bd(30.0, 0.0, 0.7, Peak), bd(64.0, 0.0, 0.7, Peak),
                bd(125.0, -1.0, 0.7, Peak), bd(250.0, 0.0, 0.7, Peak),
                bd(1000.0, 0.0, 0.7, Peak), bd(4000.0, 1.0, 0.7, Peak),
                bd(8000.0, 2.0, 0.7, Peak), bd(16000.0, 1.5, 0.7, Peak),
            ],
        },
        ToneEQPreset {
            name: "Bass Boost".into(),
            bands: vec![
                bd(30.0, 2.0, 0.7, LowShelf), bd(64.0, 2.0, 0.7, Peak),
                bd(125.0, 0.0, 0.7, Peak), bd(250.0, 0.0, 0.7, Peak),
                bd(1000.0, 0.0, 0.7, Peak), bd(4000.0, 0.0, 0.7, Peak),
                bd(8000.0, 0.0, 0.7, Peak), bd(16000.0, 0.0, 0.7, Peak),
            ],
        },
        ToneEQPreset {
            name: "Treble Boost".into(),
            bands: vec![
                bd(30.0, 0.0, 0.7, Peak), bd(64.0, 0.0, 0.7, Peak),
                bd(125.0, 0.0, 0.7, Peak), bd(250.0, 0.0, 0.7, Peak),
                bd(1000.0, 0.0, 0.7, Peak), bd(4000.0, 0.0, 0.7, Peak),
                bd(8000.0, 3.0, 0.7, HighShelf), bd(16000.0, 2.0, 0.7, Peak),
            ],
        },
        ToneEQPreset {
            name: "Mid Scoop".into(),
            bands: vec![
                bd(30.0, 0.0, 0.7, Peak), bd(64.0, 1.0, 0.7, Peak),
                bd(125.0, 0.0, 0.7, Peak), bd(250.0, 0.0, 0.7, Peak),
                bd(1000.0, -3.0, 0.7, Peak), bd(4000.0, 0.0, 0.7, Peak),
                bd(8000.0, 1.0, 0.7, Peak), bd(16000.0, 0.0, 0.7, Peak),
            ],
        },
        ToneEQPreset {
            name: "Vocal Boost".into(),
            bands: vec![
                bd(30.0, 0.0, 0.7, Peak), bd(64.0, 0.0, 0.7, Peak),
                bd(125.0, 0.0, 0.7, Peak), bd(250.0, 2.0, 0.7, Peak),
                bd(1000.0, 3.0, 0.7, Peak), bd(4000.0, 2.0, 0.7, Peak),
                bd(8000.0, 0.0, 0.7, Peak), bd(16000.0, 0.0, 0.7, Peak),
            ],
        },
        ToneEQPreset {
            name: "De-Harsh".into(),
            bands: vec![
                bd(30.0, 0.0, 0.7, Peak), bd(64.0, 0.0, 0.7, Peak),
                bd(125.0, 0.0, 0.7, Peak), bd(250.0, 0.0, 0.7, Peak),
                bd(1000.0, 0.0, 0.7, Peak), bd(4000.0, -3.0, 2.0, Peak),
                bd(8000.0, -2.0, 0.7, Peak), bd(16000.0, 0.0, 0.7, Peak),
            ],
        },
        ToneEQPreset {
            name: "Air".into(),
            bands: vec![
                bd(30.0, 0.0, 0.7, Peak), bd(64.0, 0.0, 0.7, Peak),
                bd(125.0, 0.0, 0.7, Peak), bd(250.0, 0.0, 0.7, Peak),
                bd(1000.0, 0.0, 0.7, Peak), bd(4000.0, 0.0, 0.7, Peak),
                bd(8000.0, 2.0, 0.7, HighShelf), bd(16000.0, 3.0, 0.7, Peak),
            ],
        },
        ToneEQPreset {
            name: "Lo-Fi".into(),
            bands: vec![
                bd(30.0, 0.0, 0.7, Peak), bd(64.0, 0.0, 0.7, Peak),
                bd(125.0, 3.0, 0.7, LowShelf), bd(250.0, 0.0, 0.7, Peak),
                bd(1000.0, 0.0, 0.7, Peak), bd(4000.0, -2.0, 0.7, Peak),
                bd(8000.0, 0.0, 0.7, Peak), bd(16000.0, -4.0, 0.7, LowPass),
            ],
        },
    ]
}

/// Get a tone preset by name.
pub fn get_tone_preset(name: &str) -> Option<ToneEQPreset> {
    build_tone_presets().into_iter().find(|p| p.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eq_process_bypass() {
        let mut eq = Equalizer::new();
        eq.set_bypass(true);
        let input = vec![vec![0.5f32; 100]; 2];
        let output = eq.process(&input, 44100);
        assert_eq!(output, input);
    }

    #[test]
    fn test_tone_preset_names() {
        let names = Equalizer::get_tone_preset_names();
        assert_eq!(names.len(), 10);
        assert!(names.contains(&"Flat".to_string()));
        assert!(names.contains(&"Warm".to_string()));
    }

    #[test]
    fn test_eq_band_settings_round_trip() {
        let band = EQBand::new(1000.0, 3.0, 1.5, FilterType::Peak, true);
        let settings = band.to_settings();
        let restored = EQBand::from_settings(&settings);
        assert!((restored.frequency() - 1000.0).abs() < 1e-10);
        assert!((restored.gain_db() - 3.0).abs() < 1e-10);
    }
}
