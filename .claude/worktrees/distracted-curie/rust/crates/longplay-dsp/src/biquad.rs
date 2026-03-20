//! Biquad filter implementation - exact port from C++ equalizer.cpp
//!
//! Implements the Audio EQ Cookbook formulas for biquad filter coefficient calculation
//! and Direct Form II Transposed processing.

use longplay_core::PI;
use serde::{Deserialize, Serialize};

/// Filter type enumeration matching C++ FilterType
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FilterType {
    Peak,
    LowShelf,
    HighShelf,
    HighPass,
    LowPass,
    BandPass,
    Notch,
}

impl FilterType {
    pub fn to_string_id(&self) -> &'static str {
        match self {
            FilterType::Peak => "peak",
            FilterType::LowShelf => "lowshelf",
            FilterType::HighShelf => "highshelf",
            FilterType::HighPass => "highpass",
            FilterType::LowPass => "lowpass",
            FilterType::BandPass => "bandpass",
            FilterType::Notch => "notch",
        }
    }

    pub fn from_string_id(s: &str) -> FilterType {
        match s {
            "peak" => FilterType::Peak,
            "lowshelf" => FilterType::LowShelf,
            "highshelf" => FilterType::HighShelf,
            "highpass" => FilterType::HighPass,
            "lowpass" => FilterType::LowPass,
            "bandpass" => FilterType::BandPass,
            "notch" => FilterType::Notch,
            _ => FilterType::Peak,
        }
    }
}

/// Biquad filter coefficients (unnormalized)
#[derive(Debug, Clone, Copy)]
pub struct BiquadCoeffs {
    pub b0: f64,
    pub b1: f64,
    pub b2: f64,
    pub a0: f64,
    pub a1: f64,
    pub a2: f64,
}

impl Default for BiquadCoeffs {
    fn default() -> Self {
        Self {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a0: 1.0,
            a1: 0.0,
            a2: 0.0,
        }
    }
}

/// Calculate biquad coefficients using Audio EQ Cookbook formulas.
/// Exact port of C++ calc_biquad_coeffs.
pub fn calc_biquad_coeffs(
    filter_type: FilterType,
    frequency: f64,
    gain_db: f64,
    q: f64,
    sample_rate: i32,
) -> BiquadCoeffs {
    let mut result = BiquadCoeffs::default();

    // Precalculate common values
    let w0 = 2.0 * PI * frequency / sample_rate as f64;
    let cos_w0 = w0.cos();
    let sin_w0 = w0.sin();
    let alpha = sin_w0 / (2.0 * q);

    // Calculate amplitude from gain in dB
    let a = 10.0_f64.powf(gain_db / 40.0);
    let sqrt_a = a.sqrt();

    match filter_type {
        FilterType::Peak => {
            // Peaking EQ
            let alpha_a = alpha * a;
            result.b0 = 1.0 + alpha_a;
            result.b1 = -2.0 * cos_w0;
            result.b2 = 1.0 - alpha_a;
            result.a0 = 1.0 + alpha / a;
            result.a1 = -2.0 * cos_w0;
            result.a2 = 1.0 - alpha / a;
        }
        FilterType::LowShelf => {
            let a_plus_1 = a + 1.0;
            let a_minus_1 = a - 1.0;
            let two_sqrt_a_alpha = 2.0 * sqrt_a * alpha;
            let cos_term = a_minus_1 * cos_w0;

            result.b0 = a * (a_plus_1 - cos_term + two_sqrt_a_alpha);
            result.b1 = 2.0 * a * (a_minus_1 - a_plus_1 * cos_w0);
            result.b2 = a * (a_plus_1 - cos_term - two_sqrt_a_alpha);
            result.a0 = a_plus_1 + cos_term + two_sqrt_a_alpha;
            result.a1 = -2.0 * (a_minus_1 + a_plus_1 * cos_w0);
            result.a2 = a_plus_1 + cos_term - two_sqrt_a_alpha;
        }
        FilterType::HighShelf => {
            let a_plus_1 = a + 1.0;
            let a_minus_1 = a - 1.0;
            let two_sqrt_a_alpha = 2.0 * sqrt_a * alpha;
            let cos_term = a_minus_1 * cos_w0;

            result.b0 = a * (a_plus_1 + cos_term + two_sqrt_a_alpha);
            result.b1 = -2.0 * a * (a_minus_1 + a_plus_1 * cos_w0);
            result.b2 = a * (a_plus_1 + cos_term - two_sqrt_a_alpha);
            result.a0 = a_plus_1 - cos_term + two_sqrt_a_alpha;
            result.a1 = 2.0 * (a_minus_1 - a_plus_1 * cos_w0);
            result.a2 = a_plus_1 - cos_term - two_sqrt_a_alpha;
        }
        FilterType::HighPass => {
            let one_plus_cos = 1.0 + cos_w0;
            result.b0 = one_plus_cos / 2.0;
            result.b1 = -one_plus_cos;
            result.b2 = one_plus_cos / 2.0;
            result.a0 = 1.0 + alpha;
            result.a1 = -2.0 * cos_w0;
            result.a2 = 1.0 - alpha;
        }
        FilterType::LowPass => {
            let one_minus_cos = 1.0 - cos_w0;
            result.b0 = one_minus_cos / 2.0;
            result.b1 = one_minus_cos;
            result.b2 = one_minus_cos / 2.0;
            result.a0 = 1.0 + alpha;
            result.a1 = -2.0 * cos_w0;
            result.a2 = 1.0 - alpha;
        }
        FilterType::BandPass => {
            result.b0 = alpha;
            result.b1 = 0.0;
            result.b2 = -alpha;
            result.a0 = 1.0 + alpha;
            result.a1 = -2.0 * cos_w0;
            result.a2 = 1.0 - alpha;
        }
        FilterType::Notch => {
            result.b0 = 1.0;
            result.b1 = -2.0 * cos_w0;
            result.b2 = 1.0;
            result.a0 = 1.0 + alpha;
            result.a1 = -2.0 * cos_w0;
            result.a2 = 1.0 - alpha;
        }
    }

    result
}

/// Direct Form II Transposed biquad filter.
/// Uses f64 for coefficients and state, f32 for audio samples (matching C++).
#[derive(Debug, Clone)]
pub struct BiquadFilter {
    // Normalized coefficients (divided by a0)
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    // State variables
    z1: f64,
    z2: f64,
}

impl BiquadFilter {
    pub fn new() -> Self {
        Self {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    /// Set coefficients from BiquadCoeffs, normalizing by a0.
    /// Exact port of C++ BiquadFilter::set_coefficients.
    pub fn set_coefficients(&mut self, coeffs: &BiquadCoeffs) {
        let a0_inv = 1.0 / coeffs.a0;
        self.b0 = coeffs.b0 * a0_inv;
        self.b1 = coeffs.b1 * a0_inv;
        self.b2 = coeffs.b2 * a0_inv;
        self.a1 = coeffs.a1 * a0_inv;
        self.a2 = coeffs.a2 * a0_inv;
    }

    /// Process a single sample using Direct Form II Transposed.
    /// Exact port of C++ BiquadFilter::process_sample.
    ///
    /// y[n] = b0*x[n] + z1
    /// z1 = b1*x[n] - a1*y[n] + z2
    /// z2 = b2*x[n] - a2*y[n]
    pub fn process_sample(&mut self, input: f32) -> f32 {
        let input_d = input as f64;
        let out = self.b0 * input_d + self.z1;
        self.z1 = self.b1 * input_d - self.a1 * out + self.z2;
        self.z2 = self.b2 * input_d - self.a2 * out;
        out as f32
    }

    /// Reset filter state to zero.
    pub fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }
}

impl Default for BiquadFilter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_type_round_trip() {
        let types = [
            FilterType::Peak,
            FilterType::LowShelf,
            FilterType::HighShelf,
            FilterType::HighPass,
            FilterType::LowPass,
            FilterType::BandPass,
            FilterType::Notch,
        ];
        for ft in &types {
            assert_eq!(FilterType::from_string_id(ft.to_string_id()), *ft);
        }
    }

    #[test]
    fn test_peak_filter_unity_at_zero_gain() {
        let coeffs = calc_biquad_coeffs(FilterType::Peak, 1000.0, 0.0, 1.0, 44100);
        // At 0 dB gain, peak filter should be unity (b0/a0 ≈ 1, b1/a1 equal, etc.)
        let a0_inv = 1.0 / coeffs.a0;
        assert!((coeffs.b0 * a0_inv - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_biquad_passthrough() {
        let mut filter = BiquadFilter::new();
        // Default coefficients are passthrough (b0=1, rest=0)
        let input = 0.5f32;
        let output = filter.process_sample(input);
        assert!((output - input).abs() < 1e-6);
    }
}
