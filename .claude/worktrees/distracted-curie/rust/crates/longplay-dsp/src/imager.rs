//! Stereo imager — exact port from C++ imager.cpp / imager.h
//!
//! Includes ImagerBand, ImagerPreset, and Imager with M/S encoding/decoding,
//! single-band and multi-band stereo width, mono bass, balance, and 7 presets.

use crate::dynamics::CrossoverFilter;
use longplay_core::{AudioBuffer, Settings, SettingsValue};
use longplay_core::{get_setting_f64, get_setting_bool};

// ============================================================================
// ImagerBand
// ============================================================================

/// Stereo band definition for multi-band imager.
#[derive(Debug, Clone)]
pub struct ImagerBand {
    pub name: String,
    pub low_freq: f32,
    pub high_freq: f32,
    /// Width as percentage: 0 = mono, 100 = original stereo, 200 = super wide.
    pub width: f32,
}

impl ImagerBand {
    pub fn new(name: &str, low_freq: f32, high_freq: f32, width: f32) -> Self {
        Self {
            name: name.to_string(),
            low_freq,
            high_freq,
            width,
        }
    }
}

impl Default for ImagerBand {
    fn default() -> Self {
        Self {
            name: String::new(),
            low_freq: 0.0,
            high_freq: 20000.0,
            width: 100.0,
        }
    }
}

// ============================================================================
// ImagerPreset
// ============================================================================

/// Preset definition for the stereo imager.
#[derive(Debug, Clone)]
pub struct ImagerPreset {
    pub name: String,
    pub low_width: f32,
    pub mid_width: f32,
    pub high_width: f32,
    pub mono_bass_freq: f32,
}

fn build_imager_presets() -> Vec<ImagerPreset> {
    vec![
        ImagerPreset {
            name: "Bypass".into(),
            low_width: 100.0, mid_width: 100.0, high_width: 100.0,
            mono_bass_freq: 0.0,
        },
        ImagerPreset {
            name: "Subtle Wide".into(),
            low_width: 100.0, mid_width: 115.0, high_width: 120.0,
            mono_bass_freq: 0.0,
        },
        ImagerPreset {
            name: "Wide Master".into(),
            low_width: 95.0, mid_width: 125.0, high_width: 140.0,
            mono_bass_freq: 0.0,
        },
        ImagerPreset {
            name: "Super Wide".into(),
            low_width: 90.0, mid_width: 150.0, high_width: 180.0,
            mono_bass_freq: 0.0,
        },
        ImagerPreset {
            name: "Mono Bass Wide Top".into(),
            low_width: 50.0, mid_width: 110.0, high_width: 160.0,
            mono_bass_freq: 150.0,
        },
        ImagerPreset {
            name: "Narrow".into(),
            low_width: 80.0, mid_width: 80.0, high_width: 80.0,
            mono_bass_freq: 0.0,
        },
        ImagerPreset {
            name: "Mono".into(),
            low_width: 0.0, mid_width: 0.0, high_width: 0.0,
            mono_bass_freq: 0.0,
        },
    ]
}

/// Get an imager preset by name.
pub fn get_imager_preset(name: &str) -> Option<ImagerPreset> {
    build_imager_presets().into_iter().find(|p| p.name == name)
}

/// Get all imager preset names.
pub fn get_imager_preset_names() -> Vec<String> {
    build_imager_presets().into_iter().map(|p| p.name).collect()
}

// ============================================================================
// Imager
// ============================================================================

/// Main stereo imager processor with single-band and 3-band multiband modes.
/// Exact port of C++ Imager class.
pub struct Imager {
    // Mode flags
    multiband: bool,
    bypass: bool,
    // Single-band width (0-200%)
    single_band_width: f32,
    // Multiband: 3 bands (low, mid, high)
    bands: [ImagerBand; 3],
    // Crossover frequencies
    crossover_low: f32,
    crossover_high: f32,
    // Mono bass frequency (below this, side = 0; 0 = disabled)
    mono_bass_freq: f32,
    // Balance (-1.0 to +1.0)
    balance: f32,
    // Per-channel crossover filters for multiband
    xover_low: Vec<CrossoverFilter>,
    xover_high: Vec<CrossoverFilter>,
}

impl Imager {
    pub fn new() -> Self {
        let mut imager = Self {
            multiband: false,
            bypass: false,
            single_band_width: 100.0,
            bands: [
                ImagerBand::default(),
                ImagerBand::default(),
                ImagerBand::default(),
            ],
            crossover_low: 200.0,
            crossover_high: 4000.0,
            mono_bass_freq: 0.0,
            balance: 0.0,
            xover_low: Vec::new(),
            xover_high: Vec::new(),
        };
        imager.init_default_bands();
        imager
    }

    fn init_default_bands(&mut self) {
        self.bands[0] = ImagerBand::new("Low", 20.0, 200.0, 100.0);
        self.bands[1] = ImagerBand::new("Mid", 200.0, 4000.0, 100.0);
        self.bands[2] = ImagerBand::new("High", 4000.0, 20000.0, 100.0);
    }

    // -- Mode and configuration --

    pub fn set_multiband(&mut self, enabled: bool) {
        self.multiband = enabled;
    }

    pub fn is_multiband(&self) -> bool {
        self.multiband
    }

    pub fn set_bypass(&mut self, bypass: bool) {
        self.bypass = bypass;
    }

    pub fn is_bypassed(&self) -> bool {
        self.bypass
    }

    // -- Single-band width --

    pub fn set_width(&mut self, width_pct: f32) {
        self.single_band_width = width_pct.clamp(0.0, 200.0);
    }

    pub fn width(&self) -> f32 {
        self.single_band_width
    }

    // -- Multiband width --

    pub fn set_low_width(&mut self, width_pct: f32) {
        self.bands[0].width = width_pct.clamp(0.0, 200.0);
    }

    pub fn set_mid_width(&mut self, width_pct: f32) {
        self.bands[1].width = width_pct.clamp(0.0, 200.0);
    }

    pub fn set_high_width(&mut self, width_pct: f32) {
        self.bands[2].width = width_pct.clamp(0.0, 200.0);
    }

    pub fn low_width(&self) -> f32 { self.bands[0].width }
    pub fn mid_width(&self) -> f32 { self.bands[1].width }
    pub fn high_width(&self) -> f32 { self.bands[2].width }

    // -- Crossover frequencies --

    pub fn set_crossover_low(&mut self, freq: f32) {
        self.crossover_low = freq.clamp(20.0, 10000.0);
    }

    pub fn set_crossover_high(&mut self, freq: f32) {
        self.crossover_high = freq.clamp(1000.0, 20000.0);
    }

    pub fn crossover_low(&self) -> f32 { self.crossover_low }
    pub fn crossover_high(&self) -> f32 { self.crossover_high }

    // -- Mono bass --

    pub fn set_mono_bass_freq(&mut self, freq: f32) {
        self.mono_bass_freq = freq.clamp(0.0, 10000.0);
    }

    pub fn mono_bass_freq(&self) -> f32 { self.mono_bass_freq }

    // -- Balance --

    pub fn set_balance(&mut self, balance: f32) {
        self.balance = balance.clamp(-1.0, 1.0);
    }

    pub fn balance(&self) -> f32 { self.balance }

    // -- Presets --

    pub fn apply_preset(&mut self, preset_name: &str) {
        let preset = match get_imager_preset(preset_name) {
            Some(p) => p,
            None => return,
        };

        self.set_low_width(preset.low_width);
        self.set_mid_width(preset.mid_width);
        self.set_high_width(preset.high_width);
        self.set_mono_bass_freq(preset.mono_bass_freq);
    }

    pub fn get_preset_names() -> Vec<String> {
        get_imager_preset_names()
    }

    // -- M/S helpers --

    /// Encode L/R to Mid/Side.
    #[inline]
    fn ms_encode(left: f32, right: f32) -> (f32, f32) {
        let mid = (left + right) * 0.5;
        let side = (left - right) * 0.5;
        (mid, side)
    }

    /// Decode Mid/Side back to L/R.
    #[inline]
    fn ms_decode(mid: f32, side: f32) -> (f32, f32) {
        let left = mid + side;
        let right = mid - side;
        (left, right)
    }

    /// Apply width scaling to side signal.
    #[inline]
    fn apply_width(side: f32, width_pct: f32) -> f32 {
        let width_factor = width_pct / 100.0;
        side * width_factor
    }

    // -- Main processing --

    /// Process audio buffer in-place. Requires stereo (2+ channels).
    /// Exact port of C++ Imager::process.
    pub fn process(&mut self, buffer: &mut AudioBuffer, sample_rate: i32) {
        if self.bypass || buffer.len() < 2 {
            return;
        }

        let num_samples = buffer[0].len();

        // Initialize crossover filters for multiband if needed
        if self.xover_low.is_empty() && self.multiband {
            let channels = buffer.len();
            self.xover_low.resize(channels, CrossoverFilter::new());
            self.xover_high.resize(channels, CrossoverFilter::new());
            for xo in &mut self.xover_low {
                xo.set_frequency(self.crossover_low as f64, sample_rate);
            }
            for xo in &mut self.xover_high {
                xo.set_frequency(self.crossover_high as f64, sample_rate);
            }
        }

        // Balance parameters
        let pan_left = 1.0_f32.min(1.0 - self.balance);
        let pan_right = 1.0_f32.min(1.0 + self.balance);

        if !self.multiband {
            // Single-band M/S processing
            for i in 0..num_samples {
                let left = buffer[0][i];
                let right = buffer[1][i];

                let (mid, side) = Self::ms_encode(left, right);
                let mut new_side = Self::apply_width(side, self.single_band_width);

                // Handle mono bass if enabled
                if self.mono_bass_freq > 0.0 {
                    new_side = 0.0;
                }

                let (new_left, new_right) = Self::ms_decode(mid, new_side);

                buffer[0][i] = new_left * pan_left;
                buffer[1][i] = new_right * pan_right;
            }
        } else {
            // Multiband M/S processing
            let mut low_left = vec![0.0f32; num_samples];
            let mut low_right = vec![0.0f32; num_samples];
            let mut mid_left = vec![0.0f32; num_samples];
            let mut mid_right = vec![0.0f32; num_samples];
            let mut high_left = vec![0.0f32; num_samples];
            let mut high_right = vec![0.0f32; num_samples];

            // Split into 3 bands using cascaded crossovers
            for i in 0..num_samples {
                let left = buffer[0][i];
                let right = buffer[1][i];

                // First crossover: separate low from mid+high
                let (lo_l, mh_l) = self.xover_low[0].process(left);
                let (lo_r, mh_r) = self.xover_low[1].process(right);

                low_left[i] = lo_l;
                low_right[i] = lo_r;

                // Second crossover: separate mid from high
                let (mid_l, hi_l) = self.xover_high[0].process(mh_l);
                let (mid_r, hi_r) = self.xover_high[1].process(mh_r);

                mid_left[i] = mid_l;
                mid_right[i] = mid_r;
                high_left[i] = hi_l;
                high_right[i] = hi_r;
            }

            // Process each band with its own width
            for i in 0..num_samples {
                // LOW BAND
                let (mid_lo, side_lo) = Self::ms_encode(low_left[i], low_right[i]);
                let mut new_side_lo = Self::apply_width(side_lo, self.bands[0].width);
                if self.mono_bass_freq > 0.0 && (self.crossover_low as f32) < self.mono_bass_freq {
                    new_side_lo = 0.0; // Mono bass
                }
                let (ll, lr) = Self::ms_decode(mid_lo, new_side_lo);
                low_left[i] = ll;
                low_right[i] = lr;

                // MID BAND
                let (mid_md, side_md) = Self::ms_encode(mid_left[i], mid_right[i]);
                let new_side_md = Self::apply_width(side_md, self.bands[1].width);
                let (ml, mr) = Self::ms_decode(mid_md, new_side_md);
                mid_left[i] = ml;
                mid_right[i] = mr;

                // HIGH BAND
                let (mid_hi, side_hi) = Self::ms_encode(high_left[i], high_right[i]);
                let new_side_hi = Self::apply_width(side_hi, self.bands[2].width);
                let (hl, hr) = Self::ms_decode(mid_hi, new_side_hi);
                high_left[i] = hl;
                high_right[i] = hr;
            }

            // Sum bands back together with balance
            for i in 0..num_samples {
                let left = low_left[i] + mid_left[i] + high_left[i];
                let right = low_right[i] + mid_right[i] + high_right[i];

                buffer[0][i] = left * pan_left;
                buffer[1][i] = right * pan_right;
            }
        }
    }

    pub fn reset(&mut self) {
        for xo in &mut self.xover_low {
            xo.reset();
        }
        for xo in &mut self.xover_high {
            xo.reset();
        }
    }

    pub fn to_settings(&self) -> Settings {
        let mut s = Settings::new();
        s.insert("bypass".into(), SettingsValue::Bool(self.bypass));
        s.insert("multiband".into(), SettingsValue::Bool(self.multiband));
        s.insert("single_width".into(), SettingsValue::Float(self.single_band_width as f64));
        s.insert("low_width".into(), SettingsValue::Float(self.bands[0].width as f64));
        s.insert("mid_width".into(), SettingsValue::Float(self.bands[1].width as f64));
        s.insert("high_width".into(), SettingsValue::Float(self.bands[2].width as f64));
        s.insert("crossover_low".into(), SettingsValue::Float(self.crossover_low as f64));
        s.insert("crossover_high".into(), SettingsValue::Float(self.crossover_high as f64));
        s.insert("mono_bass_freq".into(), SettingsValue::Float(self.mono_bass_freq as f64));
        s.insert("balance".into(), SettingsValue::Float(self.balance as f64));
        s
    }

    pub fn from_settings(&mut self, s: &Settings) {
        self.bypass = get_setting_bool(s, "bypass", self.bypass);
        self.multiband = get_setting_bool(s, "multiband", self.multiband);
        self.single_band_width = get_setting_f64(s, "single_width", self.single_band_width as f64) as f32;
        self.bands[0].width = get_setting_f64(s, "low_width", self.bands[0].width as f64) as f32;
        self.bands[1].width = get_setting_f64(s, "mid_width", self.bands[1].width as f64) as f32;
        self.bands[2].width = get_setting_f64(s, "high_width", self.bands[2].width as f64) as f32;
        self.crossover_low = get_setting_f64(s, "crossover_low", self.crossover_low as f64) as f32;
        self.crossover_high = get_setting_f64(s, "crossover_high", self.crossover_high as f64) as f32;
        self.mono_bass_freq = get_setting_f64(s, "mono_bass_freq", self.mono_bass_freq as f64) as f32;
        self.balance = get_setting_f64(s, "balance", self.balance as f64) as f32;
    }
}

impl Default for Imager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ms_encode_decode_identity() {
        let left = 0.8_f32;
        let right = 0.4_f32;
        let (mid, side) = Imager::ms_encode(left, right);
        let (l2, r2) = Imager::ms_decode(mid, side);
        assert!((l2 - left).abs() < 1e-6);
        assert!((r2 - right).abs() < 1e-6);
    }

    #[test]
    fn test_width_100_passthrough() {
        let side = 0.5_f32;
        let result = Imager::apply_width(side, 100.0);
        assert!((result - side).abs() < 1e-6);
    }

    #[test]
    fn test_width_0_mono() {
        let side = 0.5_f32;
        let result = Imager::apply_width(side, 0.0);
        assert!(result.abs() < 1e-6);
    }

    #[test]
    fn test_imager_bypass() {
        let mut imager = Imager::new();
        imager.set_bypass(true);
        let mut buffer = vec![vec![0.5f32; 100]; 2];
        let original = buffer.clone();
        imager.process(&mut buffer, 44100);
        assert_eq!(buffer, original);
    }

    #[test]
    fn test_preset_names() {
        let names = Imager::get_preset_names();
        assert_eq!(names.len(), 7);
        assert!(names.contains(&"Bypass".to_string()));
        assert!(names.contains(&"Wide Master".to_string()));
        assert!(names.contains(&"Mono".to_string()));
    }

    #[test]
    fn test_settings_round_trip() {
        let mut imager = Imager::new();
        imager.set_multiband(true);
        imager.set_low_width(50.0);
        imager.set_mid_width(120.0);
        imager.set_high_width(180.0);
        imager.set_balance(0.3);

        let settings = imager.to_settings();
        let mut restored = Imager::new();
        restored.from_settings(&settings);

        assert!(restored.is_multiband());
        assert!((restored.low_width() - 50.0).abs() < 1e-4);
        assert!((restored.mid_width() - 120.0).abs() < 1e-4);
        assert!((restored.high_width() - 180.0).abs() < 1e-4);
        assert!((restored.balance() - 0.3).abs() < 1e-4);
    }
}
