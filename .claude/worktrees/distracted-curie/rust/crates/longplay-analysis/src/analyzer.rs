//! Audio analyzer - combines spectral, dynamics, and stereo analysis
//! Ported from C++ analyzer.cpp

use longplay_core::AudioBuffer;
use crate::spectral::{analyze_spectrum, SpectralAnalysis};
use crate::dynamics_analysis::{analyze_dynamics, DynamicAnalysis};
use crate::stereo::{analyze_stereo, StereoAnalysis};

/// Complete audio analysis results combining spectral, dynamic, and stereo analysis
#[derive(Debug, Clone)]
pub struct AudioAnalysis {
    /// Frequency domain analysis
    pub spectral: SpectralAnalysis,
    /// Amplitude and dynamics analysis
    pub dynamics: DynamicAnalysis,
    /// Spatial/stereo characteristics
    pub stereo: StereoAnalysis,
    /// Total audio duration in seconds
    pub duration_seconds: f32,
    /// Sample rate of the analyzed audio
    pub sample_rate: i32,
    /// Number of channels
    pub channels: i32,
}

impl Default for AudioAnalysis {
    fn default() -> Self {
        Self {
            spectral: SpectralAnalysis::default(),
            dynamics: DynamicAnalysis::default(),
            stereo: StereoAnalysis::default(),
            duration_seconds: 0.0,
            sample_rate: 44100,
            channels: 2,
        }
    }
}

/// AudioAnalyzer: comprehensive audio file analyzer using FFT and signal processing.
///
/// Combines spectral, dynamics, and stereo analysis modules from this crate
/// to produce a unified `AudioAnalysis` result. Supports both file-based
/// analysis (via `longplay_io`) and direct buffer analysis.
pub struct AudioAnalyzer {
    /// Cache of the last analysis result
    last_analysis: Option<AudioAnalysis>,
}

impl AudioAnalyzer {
    pub fn new() -> Self {
        Self {
            last_analysis: None,
        }
    }

    /// Analyze an audio file by path.
    ///
    /// Reads the file using `longplay_io`, then runs spectral, dynamics,
    /// and stereo analysis on the decoded audio buffer.
    ///
    /// # Arguments
    /// * `file_path` - Path to audio file (WAV format)
    ///
    /// # Returns
    /// `AudioAnalysis` struct with all analysis results.
    ///
    /// # Errors
    /// Returns an error string if the file cannot be read or analyzed.
    pub fn analyze(&mut self, file_path: &str) -> Result<AudioAnalysis, String> {
        let (buffer, info) = longplay_io::read_audio(file_path)
            .map_err(|e| format!("Failed to read audio: {}", e))?;

        let mut result = self.analyze_buffer(&buffer, info.sample_rate);
        result.duration_seconds = info.duration_seconds as f32;
        result.sample_rate = info.sample_rate;
        result.channels = info.channels;

        self.last_analysis = Some(result.clone());
        Ok(result)
    }

    /// Analyze an in-memory audio buffer directly (no file I/O).
    ///
    /// Runs spectral, dynamics, and stereo analysis on the provided buffer.
    /// Useful when PCM samples are already available.
    ///
    /// # Arguments
    /// * `buffer` - Multi-channel audio buffer, normalized to [-1, 1]
    /// * `sample_rate` - Sample rate of the audio in Hz
    pub fn analyze_buffer(&self, buffer: &AudioBuffer, sample_rate: i32) -> AudioAnalysis {
        let spectral = analyze_spectrum(buffer, sample_rate);
        let dynamics = analyze_dynamics(buffer);
        let stereo = analyze_stereo(buffer);

        let duration = if !buffer.is_empty() && !buffer[0].is_empty() {
            buffer[0].len() as f32 / sample_rate as f32
        } else {
            0.0
        };

        AudioAnalysis {
            spectral,
            dynamics,
            stereo,
            duration_seconds: duration,
            sample_rate,
            channels: buffer.len() as i32,
        }
    }

    /// Get the last analysis result, if any.
    pub fn last_analysis(&self) -> Option<&AudioAnalysis> {
        self.last_analysis.as_ref()
    }

    /// Clear the cached last analysis result.
    pub fn reset(&mut self) {
        self.last_analysis = None;
    }
}

impl Default for AudioAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_analysis_default() {
        let analysis = AudioAnalysis::default();
        assert_eq!(analysis.duration_seconds, 0.0);
        assert_eq!(analysis.sample_rate, 44100);
        assert_eq!(analysis.channels, 2);
        assert_eq!(analysis.spectral.brightness, 0.0);
        assert_eq!(analysis.dynamics.peak_db, -200.0);
    }

    #[test]
    fn test_analyze_buffer_empty() {
        let analyzer = AudioAnalyzer::new();
        let samples: AudioBuffer = Vec::new();
        let result = analyzer.analyze_buffer(&samples, 22050);
        assert_eq!(result.duration_seconds, 0.0);
    }

    #[test]
    fn test_analyze_buffer_sine() {
        let analyzer = AudioAnalyzer::new();
        let n = 44100;
        let mut signal = vec![0.0f32; n];
        for i in 0..n {
            signal[i] = (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 22050.0).sin();
        }
        let samples = vec![signal.clone(), signal];
        let result = analyzer.analyze_buffer(&samples, 22050);

        // Duration should be 2.0 seconds (44100 samples / 22050 Hz)
        assert!((result.duration_seconds - 2.0).abs() < 0.01);
        // Should detect as mono (identical channels)
        assert!(result.stereo.is_mono);
        // Peak should be near 0 dB
        assert!(result.dynamics.peak_db > -1.0);
    }

    #[test]
    fn test_analyzer_last_analysis() {
        let analyzer = AudioAnalyzer::new();
        assert!(analyzer.last_analysis().is_none());
    }

    #[test]
    fn test_analyzer_reset() {
        let mut analyzer = AudioAnalyzer::new();
        analyzer.reset();
        assert!(analyzer.last_analysis().is_none());
    }
}
