//! Dynamic range analysis - amplitude and dynamics
//! Ported from C++ analyzer.cpp analyze_dynamics()

use longplay_core::AudioBuffer;
use longplay_core::conversions::linear_to_db_f;

const SAMPLE_RATE: i32 = 22050;
const DYNAMICS_WINDOW_MS: i32 = 100;
const PERCENTILE_LOWER: f32 = 10.0;
const PERCENTILE_UPPER: f32 = 95.0;

/// Dynamic range analysis results
#[derive(Debug, Clone)]
pub struct DynamicAnalysis {
    /// Peak level in dB
    pub peak_db: f32,
    /// RMS level in dB
    pub rms_db: f32,
    /// Crest factor: peak_db - rms_db
    pub crest_factor_db: f32,
    /// Dynamic range: P95 - P10 of 100ms RMS windows (dB)
    pub dynamic_range_db: f32,
    /// Text descriptions of dynamic characteristics
    pub descriptions: Vec<String>,
}

impl Default for DynamicAnalysis {
    fn default() -> Self {
        Self {
            peak_db: -200.0,
            rms_db: -200.0,
            crest_factor_db: 0.0,
            dynamic_range_db: 0.0,
            descriptions: Vec::new(),
        }
    }
}

/// Compute percentile of a dataset using linear interpolation
///
/// # Arguments
/// * `data` - Input data (will be sorted internally)
/// * `percentile` - Percentile value (0-100)
pub fn compute_percentile(data: &[f32], percentile: f32) -> f32 {
    if data.is_empty() {
        return 0.0;
    }

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Linear interpolation percentile
    let index = (percentile / 100.0) * (sorted.len() as f32 - 1.0);
    let lower_idx = index.floor() as usize;
    let mut upper_idx = index.ceil() as usize;

    if upper_idx >= sorted.len() {
        upper_idx = sorted.len() - 1;
    }

    if lower_idx == upper_idx {
        return sorted[lower_idx];
    }

    let frac = index - lower_idx as f32;
    sorted[lower_idx] * (1.0 - frac) + sorted[upper_idx] * frac
}

/// Analyze the dynamic characteristics of an audio buffer
///
/// Mixes to mono, computes peak, RMS, crest factor, and dynamic range
/// (P95 - P10 of 100ms RMS windows).
///
/// # Arguments
/// * `samples` - Audio buffer (multi-channel, normalized to [-1, 1])
pub fn analyze_dynamics(samples: &AudioBuffer) -> DynamicAnalysis {
    let mut result = DynamicAnalysis::default();

    if samples.is_empty() || samples[0].is_empty() {
        return result;
    }

    let num_samples = samples[0].len();
    let num_channels = samples.len() as f32;

    // Mix all channels to mono
    let mut mono = vec![0.0f32; num_samples];
    for channel in samples.iter() {
        for (i, &sample) in channel.iter().enumerate() {
            mono[i] += sample;
        }
    }
    for sample in mono.iter_mut() {
        *sample /= num_channels;
    }

    // Calculate peak
    let mut peak: f32 = 0.0;
    for &sample in mono.iter() {
        let abs_val = sample.abs();
        if abs_val > peak {
            peak = abs_val;
        }
    }
    result.peak_db = linear_to_db_f(peak);

    // Calculate RMS
    let rms_sum: f32 = mono.iter().map(|&s| s * s).sum();
    let rms = (rms_sum / mono.len() as f32).sqrt();
    result.rms_db = linear_to_db_f(rms);

    // Crest factor
    result.crest_factor_db = result.peak_db - result.rms_db;

    // Dynamic range: compute RMS of 100ms windows and take P95 - P10
    let window_samples = ((DYNAMICS_WINDOW_MS * SAMPLE_RATE) / 1000).max(1) as usize;

    let mut window_rms_values: Vec<f32> = Vec::new();

    let mut offset = 0;
    while offset + window_samples <= mono.len() {
        let mut window_sum = 0.0f32;
        for i in 0..window_samples {
            let s = mono[offset + i];
            window_sum += s * s;
        }
        let window_rms = (window_sum / window_samples as f32).sqrt();
        window_rms_values.push(linear_to_db_f(window_rms));

        offset += window_samples;
    }

    if !window_rms_values.is_empty() {
        let p10 = compute_percentile(&window_rms_values, PERCENTILE_LOWER);
        let p95 = compute_percentile(&window_rms_values, PERCENTILE_UPPER);
        result.dynamic_range_db = p95 - p10;
    }

    // Generate descriptions
    if result.peak_db > -3.0 {
        result.descriptions.push("Loud master level, near clipping".to_string());
    } else if result.peak_db > -6.0 {
        result.descriptions.push("Good master level with headroom".to_string());
    } else if result.peak_db > -12.0 {
        result.descriptions.push("Conservative master level".to_string());
    } else {
        result.descriptions.push("Very quiet master level".to_string());
    }

    if result.crest_factor_db > 12.0 {
        result.descriptions.push("High crest factor, dynamic content".to_string());
    } else if result.crest_factor_db > 6.0 {
        result.descriptions.push("Moderate dynamics".to_string());
    } else {
        result.descriptions.push("Compressed, steady levels".to_string());
    }

    if result.dynamic_range_db > 15.0 {
        result.descriptions.push("Wide dynamic range across track".to_string());
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_percentile() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let p50 = compute_percentile(&data, 50.0);
        assert!((p50 - 5.5).abs() < 0.01);

        let p0 = compute_percentile(&data, 0.0);
        assert!((p0 - 1.0).abs() < 0.01);

        let p100 = compute_percentile(&data, 100.0);
        assert!((p100 - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_empty_buffer() {
        let samples: AudioBuffer = Vec::new();
        let result = analyze_dynamics(&samples);
        assert_eq!(result.peak_db, -200.0);
    }

    #[test]
    fn test_silence() {
        let samples = vec![vec![0.0f32; 44100]; 2];
        let result = analyze_dynamics(&samples);
        assert!(result.peak_db < -100.0);
        assert!(result.descriptions.iter().any(|d| d.contains("quiet")));
    }

    #[test]
    fn test_full_scale_sine() {
        // Full-scale signal
        let n = 44100;
        let mut signal = vec![0.0f32; n];
        for i in 0..n {
            signal[i] = (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 22050.0).sin();
        }
        let samples = vec![signal.clone(), signal];
        let result = analyze_dynamics(&samples);
        // Peak should be near 0 dB
        assert!(result.peak_db > -1.0);
        // RMS of sine is about -3 dB
        assert!(result.rms_db > -5.0);
    }
}
