//! Resonance Suppressor — Soothe2-style dynamic resonance detection and attenuation.
//!
//! Uses 24 log-spaced analysis bands with bandpass envelope detection,
//! compares each band to its spectral neighborhood to find resonance peaks,
//! then applies dynamic peak EQ cuts at resonant frequencies.
//!
//! Signal flow per block:
//!   1. Bandpass analysis → per-band RMS levels
//!   2. Resonance detection → compare to local spectral envelope
//!   3. Attack/release smoothing of reduction amounts
//!   4. Apply peak EQ cuts at resonant frequencies
//!
//! Parameters: depth, sharpness, selectivity, attack, release, mode, mix, trim, delta

use crate::biquad::{calc_biquad_coeffs, BiquadCoeffs, BiquadFilter, FilterType};
use longplay_core::AudioBuffer;

/// Number of analysis bands (log-spaced 60Hz–16kHz).
const NUM_BANDS: usize = 24;
const MIN_FREQ: f64 = 60.0;
const MAX_FREQ: f64 = 16000.0;

/// Soft or hard suppression mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SuppressorMode {
    /// Gentle, proportional suppression
    Soft,
    /// Aggressive, deeper cuts
    Hard,
}

/// Per-band state for analysis and reduction.
#[derive(Debug, Clone)]
struct SuppressorBand {
    center_freq: f64,
    /// Bandpass analysis filters (one per channel, max 2)
    bp_filters: [BiquadFilter; 2],
    /// Peak cut filters applied to the actual signal
    cut_filters: [BiquadFilter; 2],
    /// Smoothed RMS level (linear)
    envelope: f64,
    /// Current gain reduction in dB (0 = no cut, positive = amount of cut)
    reduction_db: f32,
    /// Whether the cut filter coefficients need updating
    coeffs_dirty: bool,
}

impl SuppressorBand {
    fn new(freq: f64, q: f64, sr: i32) -> Self {
        let mut band = Self {
            center_freq: freq,
            bp_filters: [BiquadFilter::new(), BiquadFilter::new()],
            cut_filters: [BiquadFilter::new(), BiquadFilter::new()],
            envelope: 0.0,
            reduction_db: 0.0,
            coeffs_dirty: false,
        };
        // Init bandpass analysis filters
        let bp_coeffs = calc_biquad_coeffs(FilterType::BandPass, freq, 0.0, q, sr);
        for f in &mut band.bp_filters {
            f.set_coefficients(&bp_coeffs);
        }
        band
    }

    fn reset(&mut self) {
        for f in &mut self.bp_filters {
            f.reset();
        }
        for f in &mut self.cut_filters {
            f.reset();
        }
        self.envelope = 0.0;
        self.reduction_db = 0.0;
    }
}

/// Soothe2-style resonance suppressor.
pub struct ResonanceSuppressor {
    bands: Vec<SuppressorBand>,
    /// Maximum reduction in dB (0–20).
    depth: f32,
    /// Narrowness of cuts (1–10, maps to Q 0.5–8.0).
    sharpness: f32,
    /// How selective — higher = only worst resonances (1–10).
    selectivity: f32,
    /// Attack time in ms.
    attack_ms: f32,
    /// Release time in ms.
    release_ms: f32,
    /// Soft or hard mode.
    mode: SuppressorMode,
    /// Wet/dry mix (0.0–1.0).
    mix: f32,
    /// Output trim in dB (-12 to +12).
    trim_db: f32,
    /// Delta mode: output only the removed signal.
    delta: bool,
    /// Bypass.
    bypass: bool,
    /// Last sample rate (for detecting changes).
    last_sr: i32,
    /// Reduction curve for UI display (dB, one per band).
    reduction_curve: [f32; NUM_BANDS],
}

impl ResonanceSuppressor {
    pub fn new() -> Self {
        Self {
            bands: Vec::new(),
            depth: 5.0,
            sharpness: 4.0,
            selectivity: 3.5,
            attack_ms: 5.0,
            release_ms: 50.0,
            mode: SuppressorMode::Soft,
            mix: 1.0,
            trim_db: 0.0,
            delta: false,
            bypass: false,
            last_sr: 0,
            reduction_curve: [0.0; NUM_BANDS],
        }
    }

    // ========== Setters ==========

    pub fn set_depth(&mut self, db: f32) {
        self.depth = db.clamp(0.0, 20.0);
    }

    pub fn set_sharpness(&mut self, val: f32) {
        self.sharpness = val.clamp(1.0, 10.0);
    }

    pub fn set_selectivity(&mut self, val: f32) {
        self.selectivity = val.clamp(1.0, 10.0);
    }

    pub fn set_attack(&mut self, ms: f32) {
        self.attack_ms = ms.clamp(0.5, 50.0);
    }

    pub fn set_release(&mut self, ms: f32) {
        self.release_ms = ms.clamp(5.0, 500.0);
    }

    pub fn set_mode(&mut self, mode: SuppressorMode) {
        self.mode = mode;
    }

    pub fn set_mode_str(&mut self, s: &str) {
        self.mode = match s {
            "hard" | "Hard" => SuppressorMode::Hard,
            _ => SuppressorMode::Soft,
        };
    }

    pub fn set_mix(&mut self, mix: f32) {
        self.mix = mix.clamp(0.0, 1.0);
    }

    pub fn set_trim(&mut self, db: f32) {
        self.trim_db = db.clamp(-12.0, 12.0);
    }

    pub fn set_delta(&mut self, enabled: bool) {
        self.delta = enabled;
    }

    pub fn set_bypass(&mut self, bypass: bool) {
        self.bypass = bypass;
    }

    pub fn is_bypassed(&self) -> bool {
        self.bypass
    }

    /// Get the current reduction curve for UI display (dB per band).
    pub fn reduction_curve(&self) -> &[f32; NUM_BANDS] {
        &self.reduction_curve
    }

    /// Get band center frequencies for UI display.
    pub fn band_frequencies(&self) -> Vec<f64> {
        self.bands.iter().map(|b| b.center_freq).collect()
    }

    // ========== Internal ==========

    /// Initialize or re-initialize bands for the given sample rate.
    fn init_bands(&mut self, sr: i32) {
        if sr == self.last_sr && self.bands.len() == NUM_BANDS {
            return;
        }
        self.last_sr = sr;
        self.bands.clear();

        let log_min = MIN_FREQ.ln();
        let log_max = MAX_FREQ.ln();
        // Analysis Q — moderately narrow for good frequency resolution
        let analysis_q = 2.5;

        for i in 0..NUM_BANDS {
            let t = i as f64 / (NUM_BANDS - 1) as f64;
            let freq = (log_min + t * (log_max - log_min)).exp();
            self.bands.push(SuppressorBand::new(freq, analysis_q, sr));
        }
    }

    /// Map sharpness (1–10) to peak EQ Q factor.
    fn sharpness_to_q(&self) -> f64 {
        // sharpness 1 → Q 0.5 (wide cuts), sharpness 10 → Q 8.0 (narrow cuts)
        0.5 + (self.sharpness as f64 - 1.0) * (8.0 - 0.5) / 9.0
    }

    /// Map selectivity (1–10) to threshold in dB.
    /// Low selectivity = low threshold = catches more resonances.
    fn selectivity_to_threshold(&self) -> f32 {
        // selectivity 1 → 2 dB threshold (catches everything)
        // selectivity 10 → 14 dB threshold (only worst peaks)
        2.0 + (self.selectivity - 1.0) * 12.0 / 9.0
    }

    /// Process audio buffer in-place.
    ///
    /// The processing is done in two phases per block:
    /// 1. Analysis: bandpass → envelope → detect resonances → compute reduction
    /// 2. Application: apply dynamic peak EQ cuts to the signal
    pub fn process(&mut self, buffer: &mut AudioBuffer, sr: i32) {
        if self.bypass || self.depth < 0.1 {
            return;
        }

        self.init_bands(sr);

        let num_channels = buffer.len().min(2);
        let num_samples = if num_channels > 0 { buffer[0].len() } else { return };
        if num_samples == 0 {
            return;
        }

        // Save dry signal for mix/delta
        let dry: Vec<Vec<f32>> = if self.mix < 1.0 || self.delta {
            buffer.iter().take(num_channels).map(|ch| ch.clone()).collect()
        } else {
            Vec::new()
        };

        // ── Phase 1: Analysis ──
        // Run bandpass filters and accumulate per-band energy
        let mut band_energy = vec![0.0f64; NUM_BANDS];

        for s in 0..num_samples {
            for (i, band) in self.bands.iter_mut().enumerate() {
                let mut sum_sq = 0.0f64;
                for ch in 0..num_channels {
                    let bp_out = band.bp_filters[ch].process_sample(buffer[ch][s]);
                    sum_sq += (bp_out as f64) * (bp_out as f64);
                }
                band_energy[i] += sum_sq;
            }
        }

        // Convert to dB (RMS)
        let mut band_levels_db = vec![0.0f32; NUM_BANDS];
        for i in 0..NUM_BANDS {
            let rms = (band_energy[i] / (num_samples * num_channels) as f64).sqrt();
            band_levels_db[i] = if rms > 1e-10 {
                20.0 * rms.log10() as f32
            } else {
                -120.0
            };
        }

        // ── Phase 2: Resonance Detection ──
        let threshold_db = self.selectivity_to_threshold();
        let max_depth = if self.mode == SuppressorMode::Hard {
            self.depth * 1.5
        } else {
            self.depth
        };
        let depth_ratio = if self.mode == SuppressorMode::Hard { 1.2 } else { 0.8 };

        // Attack/release coefficients (per-block smoothing)
        let block_duration_ms = num_samples as f32 * 1000.0 / sr as f32;
        let att_coeff = (-block_duration_ms / self.attack_ms.max(0.5)).exp();
        let rel_coeff = (-block_duration_ms / self.release_ms.max(5.0)).exp();

        let cut_q = self.sharpness_to_q();

        for i in 0..NUM_BANDS {
            // Compute local spectral envelope (average of neighbors)
            let radius = (3.0 + (10.0 - self.sharpness) * 0.5) as usize;
            let start = if i >= radius { i - radius } else { 0 };
            let end = (i + radius + 1).min(NUM_BANDS);
            let mut sum = 0.0f32;
            let mut count = 0;
            for j in start..end {
                if j != i {
                    sum += band_levels_db[j];
                    count += 1;
                }
            }
            let local_avg = if count > 0 { sum / count as f32 } else { band_levels_db[i] };

            // Excess above local envelope + threshold
            let excess = (band_levels_db[i] - local_avg - threshold_db).max(0.0);

            // Target reduction
            let target_reduction = (excess * depth_ratio).min(max_depth);

            // Smooth with attack/release
            let band = &mut self.bands[i];
            if target_reduction > band.reduction_db {
                // Attack (increasing reduction)
                band.reduction_db = att_coeff * band.reduction_db
                    + (1.0 - att_coeff) * target_reduction;
            } else {
                // Release (decreasing reduction)
                band.reduction_db = rel_coeff * band.reduction_db
                    + (1.0 - rel_coeff) * target_reduction;
            }

            // Clamp small values
            if band.reduction_db < 0.1 {
                band.reduction_db = 0.0;
            }

            // Store for UI
            self.reduction_curve[i] = band.reduction_db;

            // Update cut filter coefficients (once per block — avoid per-sample trig)
            if band.reduction_db > 0.1 {
                let coeffs = calc_biquad_coeffs(
                    FilterType::Peak,
                    band.center_freq,
                    -(band.reduction_db as f64),
                    cut_q,
                    sr,
                );
                for ch in 0..num_channels {
                    band.cut_filters[ch].set_coefficients(&coeffs);
                }
            } else {
                // Reset to passthrough
                let unity = BiquadCoeffs::default();
                for ch in 0..num_channels {
                    band.cut_filters[ch].set_coefficients(&unity);
                }
            }
        }

        // ── Phase 3: Apply cuts ──
        for s in 0..num_samples {
            for ch in 0..num_channels {
                let mut sample = buffer[ch][s];
                for band in &mut self.bands {
                    if band.reduction_db > 0.1 {
                        sample = band.cut_filters[ch].process_sample(sample);
                    }
                }
                buffer[ch][s] = sample;
            }
        }

        // ── Phase 4: Trim + Mix + Delta ──
        let trim_linear = 10.0_f32.powf(self.trim_db / 20.0);

        if self.delta {
            // Delta mode: output = dry - wet (the removed signal)
            for ch in 0..num_channels {
                for s in 0..num_samples {
                    buffer[ch][s] = (dry[ch][s] - buffer[ch][s]) * trim_linear;
                }
            }
        } else if self.mix < 1.0 {
            // Wet/dry mix
            let wet_mix = self.mix;
            let dry_mix = 1.0 - wet_mix;
            for ch in 0..num_channels {
                for s in 0..num_samples {
                    buffer[ch][s] = (dry[ch][s] * dry_mix + buffer[ch][s] * wet_mix) * trim_linear;
                }
            }
        } else if (trim_linear - 1.0).abs() > 0.001 {
            // Full wet, just apply trim
            for ch in 0..num_channels {
                for s in 0..num_samples {
                    buffer[ch][s] *= trim_linear;
                }
            }
        }
    }

    pub fn reset(&mut self) {
        for band in &mut self.bands {
            band.reset();
        }
        self.reduction_curve = [0.0; NUM_BANDS];
        self.last_sr = 0;
    }
}

impl Default for ResonanceSuppressor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_passthrough_when_bypassed() {
        let mut rs = ResonanceSuppressor::new();
        rs.set_bypass(true);
        let mut buffer = vec![vec![0.5f32; 512], vec![-0.3f32; 512]];
        let original = buffer.clone();
        rs.process(&mut buffer, 44100);
        assert_eq!(buffer, original);
    }

    #[test]
    fn test_passthrough_zero_depth() {
        let mut rs = ResonanceSuppressor::new();
        rs.set_depth(0.0);
        let mut buffer = vec![vec![0.5f32; 512], vec![-0.3f32; 512]];
        let original = buffer.clone();
        rs.process(&mut buffer, 44100);
        assert_eq!(buffer, original);
    }

    #[test]
    fn test_init_bands() {
        let mut rs = ResonanceSuppressor::new();
        rs.init_bands(48000);
        assert_eq!(rs.bands.len(), NUM_BANDS);
        // First band should be near 60 Hz
        assert!((rs.bands[0].center_freq - 60.0).abs() < 1.0);
        // Last band should be near 16000 Hz
        assert!((rs.bands[NUM_BANDS - 1].center_freq - 16000.0).abs() < 100.0);
    }

    #[test]
    fn test_selectivity_range() {
        let mut rs = ResonanceSuppressor::new();
        rs.set_selectivity(1.0);
        assert!((rs.selectivity_to_threshold() - 2.0).abs() < 0.1);
        rs.set_selectivity(10.0);
        assert!((rs.selectivity_to_threshold() - 14.0).abs() < 0.1);
    }

    #[test]
    fn test_processes_without_panic() {
        let mut rs = ResonanceSuppressor::new();
        rs.set_depth(10.0);
        rs.set_sharpness(5.0);
        rs.set_selectivity(5.0);
        // Generate a test signal with a resonance at 2kHz
        let sr = 48000;
        let n = 1024;
        let mut buffer = vec![vec![0.0f32; n]; 2];
        for i in 0..n {
            let t = i as f32 / sr as f32;
            let signal = 0.5 * (2.0 * std::f32::consts::PI * 2000.0 * t).sin()
                + 0.1 * (2.0 * std::f32::consts::PI * 440.0 * t).sin();
            buffer[0][i] = signal;
            buffer[1][i] = signal;
        }
        rs.process(&mut buffer, sr);
        // Should not panic and output should be finite
        for s in &buffer[0] {
            assert!(s.is_finite());
        }
    }

    #[test]
    fn test_delta_mode() {
        let mut rs = ResonanceSuppressor::new();
        rs.set_depth(10.0);
        rs.set_delta(true);
        let sr = 48000;
        let n = 512;
        let mut buffer = vec![vec![0.3f32; n]; 2];
        rs.process(&mut buffer, sr);
        // Delta output should be finite
        for s in &buffer[0] {
            assert!(s.is_finite());
        }
    }
}
