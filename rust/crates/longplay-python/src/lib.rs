//! PyO3 Python bindings for LongPlay Studio V5
//!
//! Exposes high-level mastering, AI DJ, and Hook Extractor APIs to Python.
//!
//! Classes:
//! - PyMasterChain: Full mastering chain with EQ/Dynamics/Imager/Maximizer delegation
//! - PyAudioAnalyzer: Audio spectral/dynamics/stereo analysis
//! - PyLoudnessMeter: LUFS and True Peak measurement
//! - PyAIDJ: AI-powered playlist ordering and analysis
//! - PyHookExtractor: Audio hook/chorus detection and extraction
//!
//! Return types:
//! - PyMasterRecommendation: AI mastering recommendation
//! - PyAudioAnalysis: Track analysis (BPM, Key, Energy)
//! - PyHookResult: Hook detection result

pub mod types;
pub mod master_chain;
pub mod batch_master;
pub mod ai_dj;
pub mod hook_extractor;
pub mod rt_engine;

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;

use longplay_analysis::analyzer::AudioAnalyzer;
use longplay_analysis::loudness::LoudnessMeter;

// Re-export PyO3 classes from submodules
use master_chain::PyMasterChain;
use batch_master::{PyBatchMasterConfig, PyBatchMasterResult};
use ai_dj::PyAIDJ;
use hook_extractor::PyHookExtractor;
use types::{PyMasterRecommendation, PyAudioAnalysis, PyHookResult};
use rt_engine::PyRtEngine;

/// Python-facing AudioAnalyzer wrapper
#[pyclass]
struct PyAudioAnalyzer {
    inner: AudioAnalyzer,
}

#[pymethods]
impl PyAudioAnalyzer {
    #[new]
    fn new() -> Self {
        Self {
            inner: AudioAnalyzer::new(),
        }
    }

    /// Analyze an audio file and return a summary string
    fn analyze(&mut self, file_path: &str) -> PyResult<String> {
        let analysis = self.inner.analyze(file_path)
            .map_err(|e| PyRuntimeError::new_err(e))?;

        let mut summary = format!("Duration: {:.2}s\n", analysis.duration_seconds);
        summary.push_str("\nSpectral:\n");
        summary.push_str(&format!("  Brightness: {:.1}%\n", analysis.spectral.brightness * 100.0));
        summary.push_str(&format!("  Centroid: {:.1} Hz\n", analysis.spectral.spectral_centroid));
        summary.push_str(&format!("  Sub Energy: {:.1}%\n", analysis.spectral.sub_energy * 100.0));
        summary.push_str(&format!("  Low Energy: {:.1}%\n", analysis.spectral.low_energy * 100.0));
        summary.push_str(&format!("  Mid Energy: {:.1}%\n", analysis.spectral.mid_energy * 100.0));
        summary.push_str(&format!("  High Energy: {:.1}%\n", analysis.spectral.high_energy * 100.0));
        summary.push_str("\nDynamics:\n");
        summary.push_str(&format!("  Peak: {:.2} dB\n", analysis.dynamics.peak_db));
        summary.push_str(&format!("  RMS: {:.2} dB\n", analysis.dynamics.rms_db));
        summary.push_str(&format!("  Crest Factor: {:.2} dB\n", analysis.dynamics.crest_factor_db));
        summary.push_str(&format!("  Dynamic Range: {:.2} dB\n", analysis.dynamics.dynamic_range_db));
        summary.push_str("\nStereo:\n");
        summary.push_str(&format!("  Mono: {}\n", analysis.stereo.is_mono));
        summary.push_str(&format!("  Correlation: {:.2}\n", analysis.stereo.correlation));
        summary.push_str(&format!("  Width: {:.1}%\n", analysis.stereo.width_pct));
        summary.push_str(&format!("  Balance: {:.2}\n", analysis.stereo.balance_lr));

        Ok(summary)
    }
}

/// Python-facing LoudnessMeter wrapper
#[pyclass]
struct PyLoudnessMeter {
    inner: LoudnessMeter,
}

#[pymethods]
impl PyLoudnessMeter {
    #[new]
    fn new() -> Self {
        Self {
            inner: LoudnessMeter::new("ffmpeg"),
        }
    }

    /// Measure loudness of an audio file, returns (LUFS, true_peak_dBTP)
    fn measure(&self, file_path: &str) -> PyResult<(f32, f32)> {
        let result = self.inner.analyze(file_path)
            .ok_or_else(|| PyRuntimeError::new_err("Loudness measurement failed"))?;
        Ok((result.integrated_lufs, result.true_peak_dbtp))
    }

    /// Get full loudness analysis as summary string
    fn analyze(&self, file_path: &str) -> PyResult<String> {
        let result = self.inner.analyze(file_path)
            .ok_or_else(|| PyRuntimeError::new_err("Loudness measurement failed"))?;

        let summary = format!(
            "Integrated LUFS: {:.2}\nTrue Peak: {:.2} dBTP\nLRA: {:.2} LU\nDuration: {:.2}s\nSample Rate: {} Hz\nChannels: {}",
            result.integrated_lufs,
            result.true_peak_dbtp,
            result.lra,
            result.duration_sec,
            result.sample_rate,
            result.channels
        );

        Ok(summary)
    }
}

/// LongPlay Studio V5 Python module
#[pymodule]
fn longplay(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Mastering
    m.add_class::<PyMasterChain>()?;
    m.add_class::<PyMasterRecommendation>()?;
    m.add_class::<PyAudioAnalyzer>()?;
    m.add_class::<PyLoudnessMeter>()?;

    // AI DJ
    m.add_class::<PyAIDJ>()?;
    m.add_class::<PyAudioAnalysis>()?;

    // Hook Extractor
    m.add_class::<PyHookExtractor>()?;
    m.add_class::<PyHookResult>()?;

    // Real-Time Audio Engine
    m.add_class::<PyRtEngine>()?;

    // Batch Mastering (Content Factory)
    m.add_class::<PyBatchMasterConfig>()?;
    m.add_class::<PyBatchMasterResult>()?;
    m.add_function(wrap_pyfunction!(batch_master::batch_master, m)?)?;
    m.add_function(wrap_pyfunction!(batch_master::get_parallelism, m)?)?;

    Ok(())
}
