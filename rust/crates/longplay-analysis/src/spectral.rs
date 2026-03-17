//! Spectral analysis - frequency domain characteristics
//! Ported from C++ analyzer.cpp analyze_spectrum()

use longplay_core::AudioBuffer;

use crate::fft::{apply_hann_window, rfft};

// Frequency band boundaries in Hz
pub const SUB_LOW_HZ: f32 = 20.0;
pub const SUB_HIGH_HZ: f32 = 80.0;
pub const LOW_LOW_HZ: f32 = 80.0;
pub const LOW_HIGH_HZ: f32 = 300.0;
pub const MID_LOW_HZ: f32 = 300.0;
pub const MID_HIGH_HZ: f32 = 6000.0;
pub const HIGH_LOW_HZ: f32 = 6000.0;
pub const HIGH_HIGH_HZ: f32 = 20000.0;

const FFT_SIZE: usize = 4096;
const EPSILON: f32 = 1e-10;

/// Spectral analysis results: frequency domain characteristics
#[derive(Debug, Clone)]
pub struct SpectralAnalysis {
    /// Energy in sub-bass (20-80 Hz), normalized [0,1]
    pub sub_energy: f32,
    /// Energy in low frequencies (80-300 Hz), normalized [0,1]
    pub low_energy: f32,
    /// Energy in midrange (300-6000 Hz), normalized [0,1]
    pub mid_energy: f32,
    /// Energy in high frequencies (6000-20000 Hz), normalized [0,1]
    pub high_energy: f32,
    /// Brightness metric: high_energy / total, [0,1]
    pub brightness: f32,
    /// Weighted average frequency in Hz
    pub spectral_centroid: f32,
    /// Text descriptions of spectral characteristics
    pub descriptions: Vec<String>,
}

impl Default for SpectralAnalysis {
    fn default() -> Self {
        Self {
            sub_energy: 0.0,
            low_energy: 0.0,
            mid_energy: 0.0,
            high_energy: 0.0,
            brightness: 0.0,
            spectral_centroid: 0.0,
            descriptions: Vec::new(),
        }
    }
}

/// Get frequency bins for an FFT result
///
/// # Arguments
/// * `fft_size` - Number of FFT output bins
/// * `sample_rate` - Sample rate in Hz
fn get_frequency_bins(fft_size: usize, sample_rate: i32) -> Vec<f32> {
    let freq_step = sample_rate as f32 / (fft_size as f32 - 1.0);
    (0..fft_size).map(|i| i as f32 * freq_step).collect()
}

/// Analyze the spectral content of an audio buffer
///
/// Mixes to mono, processes in FFT_SIZE=4096 chunks with Hann window and 50% overlap,
/// averages the spectrum, then calculates band energies, brightness, and spectral centroid.
///
/// # Arguments
/// * `samples` - Audio buffer (multi-channel, normalized to [-1, 1])
/// * `sample_rate` - Sample rate in Hz
pub fn analyze_spectrum(samples: &AudioBuffer, sample_rate: i32) -> SpectralAnalysis {
    let mut result = SpectralAnalysis::default();

    if samples.is_empty() || samples[0].is_empty() {
        return result;
    }

    let num_samples = samples[0].len();
    let num_channels = samples.len() as f32;

    // Mix all channels to mono for spectral analysis
    let mut mono = vec![0.0f32; num_samples];
    for channel in samples.iter() {
        for (i, &sample) in channel.iter().enumerate() {
            mono[i] += sample;
        }
    }
    for sample in mono.iter_mut() {
        *sample /= num_channels;
    }

    // Process audio in chunks with Hann window (50% overlap)
    let mut avg_spectrum: Vec<f32> = Vec::new();
    let mut num_chunks = 0;

    let mut offset = 0;
    while offset + FFT_SIZE <= mono.len() {
        // Extract chunk and apply Hann window
        let mut chunk = mono[offset..offset + FFT_SIZE].to_vec();
        apply_hann_window(&mut chunk);

        // Compute FFT
        let spectrum = rfft(&chunk);

        // Accumulate spectrum
        if avg_spectrum.is_empty() {
            avg_spectrum = spectrum;
        } else {
            for (i, &val) in spectrum.iter().enumerate() {
                avg_spectrum[i] += val;
            }
        }
        num_chunks += 1;

        offset += FFT_SIZE / 2;
    }

    // Average the spectrum
    if num_chunks > 0 {
        for val in avg_spectrum.iter_mut() {
            *val /= num_chunks as f32;
        }
    }

    if avg_spectrum.is_empty() {
        return result;
    }

    // Get frequency bins
    let freq_bins = get_frequency_bins(avg_spectrum.len(), sample_rate);

    // Calculate total energy
    let mut total_energy: f32 = avg_spectrum.iter().sum();
    if total_energy < EPSILON {
        total_energy = EPSILON;
    }

    // Extract energy in frequency bands
    let mut sub_sum = 0.0f32;
    let mut low_sum = 0.0f32;
    let mut mid_sum = 0.0f32;
    let mut high_sum = 0.0f32;
    let mut brightness_numerator = 0.0f32;
    let mut centroid_numerator = 0.0f32;
    let mut centroid_denominator = 0.0f32;

    for (i, &magnitude) in avg_spectrum.iter().enumerate() {
        let freq = freq_bins[i];

        // Accumulate for spectral centroid
        centroid_numerator += freq * magnitude;
        centroid_denominator += magnitude;

        // Frequency band energy
        if freq >= SUB_LOW_HZ && freq < SUB_HIGH_HZ {
            sub_sum += magnitude;
        } else if freq >= LOW_LOW_HZ && freq < LOW_HIGH_HZ {
            low_sum += magnitude;
        } else if freq >= MID_LOW_HZ && freq < MID_HIGH_HZ {
            mid_sum += magnitude;
        } else if freq >= HIGH_LOW_HZ && freq <= HIGH_HIGH_HZ {
            high_sum += magnitude;
            brightness_numerator += magnitude;
        }
    }

    // Normalize energies
    result.sub_energy = (sub_sum / total_energy).clamp(0.0, 1.0);
    result.low_energy = (low_sum / total_energy).clamp(0.0, 1.0);
    result.mid_energy = (mid_sum / total_energy).clamp(0.0, 1.0);
    result.high_energy = (high_sum / total_energy).clamp(0.0, 1.0);

    // Brightness: high energy / total energy
    result.brightness = (brightness_numerator / total_energy).clamp(0.0, 1.0);

    // Spectral centroid
    if centroid_denominator > EPSILON {
        result.spectral_centroid = centroid_numerator / centroid_denominator;
    }

    // Generate descriptions
    if result.brightness > 0.7 {
        result.descriptions.push("Bright, presence-heavy sound".to_string());
    } else if result.brightness > 0.4 {
        result.descriptions.push("Well-balanced frequency spectrum".to_string());
    } else {
        result.descriptions.push("Warm, bass-heavy character".to_string());
    }

    if result.sub_energy > 0.15 {
        result.descriptions.push("Strong sub-bass presence".to_string());
    }

    if result.high_energy > 0.25 {
        result.descriptions.push("Elevated treble frequencies".to_string());
    } else if result.high_energy < 0.10 {
        result.descriptions.push("Reduced treble/presence dip".to_string());
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_buffer() {
        let samples: AudioBuffer = Vec::new();
        let result = analyze_spectrum(&samples, 22050);
        assert_eq!(result.sub_energy, 0.0);
        assert_eq!(result.brightness, 0.0);
    }

    #[test]
    fn test_silence() {
        let samples = vec![vec![0.0f32; 8192]; 2];
        let result = analyze_spectrum(&samples, 22050);
        assert!(result.descriptions.len() > 0);
    }

    #[test]
    fn test_frequency_bins() {
        let bins = get_frequency_bins(5, 22050);
        assert_eq!(bins.len(), 5);
        assert!((bins[0] - 0.0).abs() < 1e-6);
        // freq_step = 22050 / 4 = 5512.5
        assert!((bins[1] - 5512.5).abs() < 1e-1);
    }
}
