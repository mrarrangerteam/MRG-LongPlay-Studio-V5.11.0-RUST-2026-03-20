//! Stereo analysis - spatial characteristics
//! Ported from C++ analyzer.cpp analyze_stereo()

use longplay_core::AudioBuffer;

const EPSILON: f32 = 1e-10;
const MONO_THRESHOLD: f32 = 0.01;

/// Stereo analysis results: spatial characteristics
#[derive(Debug, Clone)]
pub struct StereoAnalysis {
    /// Correlation between L and R channels [-1, +1]
    pub correlation: f32,
    /// Stereo width percentage [0, 100]
    pub width_pct: f32,
    /// Balance between L and R channels [-1, +1], negative = left heavy
    pub balance_lr: f32,
    /// True if detected as mono audio
    pub is_mono: bool,
    /// Text descriptions of stereo characteristics
    pub descriptions: Vec<String>,
}

impl Default for StereoAnalysis {
    fn default() -> Self {
        Self {
            correlation: 0.0,
            width_pct: 0.0,
            balance_lr: 0.0,
            is_mono: false,
            descriptions: Vec::new(),
        }
    }
}

/// Analyze stereo characteristics of an audio buffer
///
/// Computes Pearson correlation between L/R, stereo width from correlation,
/// L/R balance from mean absolute values, and mono detection using mid/side
/// energy ratio.
///
/// # Arguments
/// * `samples` - Audio buffer (should be 2 channels for stereo analysis)
pub fn analyze_stereo(samples: &AudioBuffer) -> StereoAnalysis {
    let mut result = StereoAnalysis::default();

    if samples.len() < 2 {
        result.is_mono = true;
        result.descriptions.push("Mono audio".to_string());
        return result;
    }

    let left = &samples[0];
    let right = &samples[1];

    if left.is_empty() {
        result.is_mono = true;
        result.descriptions.push("Mono audio".to_string());
        return result;
    }

    let num_samples = left.len().min(right.len());

    // Calculate correlation coefficient using Pearson correlation
    let mean_l: f32 = left[..num_samples].iter().sum::<f32>() / num_samples as f32;
    let mean_r: f32 = right[..num_samples].iter().sum::<f32>() / num_samples as f32;

    let mut cov_lr = 0.0f32;
    let mut var_l = 0.0f32;
    let mut var_r = 0.0f32;

    for i in 0..num_samples {
        let dl = left[i] - mean_l;
        let dr = right[i] - mean_r;
        cov_lr += dl * dr;
        var_l += dl * dl;
        var_r += dr * dr;
    }

    cov_lr /= num_samples as f32;
    var_l /= num_samples as f32;
    var_r /= num_samples as f32;

    // Correlation coefficient
    if var_l > EPSILON && var_r > EPSILON {
        result.correlation = (cov_lr / (var_l * var_r).sqrt()).clamp(-1.0, 1.0);
    }

    // Width: estimated from correlation, scaled to percentage
    result.width_pct = ((1.0 - result.correlation.abs()) * 100.0).clamp(0.0, 100.0);

    // Balance: mean absolute value of L vs R
    let mut mean_abs_l = 0.0f32;
    let mut mean_abs_r = 0.0f32;

    for i in 0..num_samples {
        mean_abs_l += left[i].abs();
        mean_abs_r += right[i].abs();
    }
    mean_abs_l /= num_samples as f32;
    mean_abs_r /= num_samples as f32;

    let sum_lr = mean_abs_l + mean_abs_r;
    if sum_lr > EPSILON {
        // Normalize to [-1, 1]: -1 = left heavy, +1 = right heavy
        result.balance_lr = (mean_abs_r - mean_abs_l) / sum_lr;
    }

    // Mono detection: if mid-side energy ratio is very small
    // Mid = (L+R)/2, Side = (L-R)/2
    let mut mid_energy = 0.0f32;
    let mut side_energy = 0.0f32;

    for i in 0..num_samples {
        let mid = (left[i] + right[i]) / 2.0;
        let side = (left[i] - right[i]) / 2.0;
        mid_energy += mid * mid;
        side_energy += side * side;
    }

    let sum_energy = mid_energy + side_energy;
    if sum_energy > EPSILON {
        let side_ratio = side_energy / sum_energy;
        result.is_mono = side_ratio < MONO_THRESHOLD;
    } else {
        result.is_mono = true;
    }

    // Generate descriptions
    if result.is_mono {
        result.descriptions.push("Mono audio".to_string());
    } else {
        result.descriptions.push("Stereo audio".to_string());

        if result.width_pct > 80.0 {
            result.descriptions.push("Very wide stereo field".to_string());
        } else if result.width_pct > 50.0 {
            result.descriptions.push("Wide stereo separation".to_string());
        } else if result.width_pct > 20.0 {
            result.descriptions.push("Moderate stereo width".to_string());
        } else {
            result.descriptions.push("Narrow stereo field".to_string());
        }

        if result.balance_lr.abs() > 0.3 {
            if result.balance_lr < 0.0 {
                result.descriptions.push("Left channel louder".to_string());
            } else {
                result.descriptions.push("Right channel louder".to_string());
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mono_detection_single_channel() {
        let samples = vec![vec![0.5f32; 1000]];
        let result = analyze_stereo(&samples);
        assert!(result.is_mono);
    }

    #[test]
    fn test_identical_channels_is_mono() {
        let signal: Vec<f32> = (0..10000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 22050.0).sin())
            .collect();
        let samples = vec![signal.clone(), signal];
        let result = analyze_stereo(&samples);
        assert!(result.is_mono);
        assert!(result.correlation > 0.99);
    }

    #[test]
    fn test_stereo_content() {
        // Different content in L and R
        let left: Vec<f32> = (0..10000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 22050.0).sin())
            .collect();
        let right: Vec<f32> = (0..10000)
            .map(|i| (2.0 * std::f32::consts::PI * 880.0 * i as f32 / 22050.0).sin())
            .collect();
        let samples = vec![left, right];
        let result = analyze_stereo(&samples);
        assert!(!result.is_mono);
        assert!(result.descriptions.iter().any(|d| d.contains("Stereo")));
    }

    #[test]
    fn test_balance_centered() {
        let signal: Vec<f32> = (0..10000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 22050.0).sin())
            .collect();
        let samples = vec![signal.clone(), signal];
        let result = analyze_stereo(&samples);
        assert!(result.balance_lr.abs() < 0.01);
    }
}
