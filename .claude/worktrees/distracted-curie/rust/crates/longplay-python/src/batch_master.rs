//! PyO3 bindings for BatchMasterLite — parallel batch mastering
//!
//! Exposes batch_master() to Python for Content Factory pipeline.
//! Uses rayon for automatic parallelism across all CPU cores.

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;

use longplay_chain::batch_master as rust_batch;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Python-facing batch master configuration
#[pyclass]
#[derive(Clone)]
pub struct PyBatchMasterConfig {
    #[pyo3(get, set)]
    pub dynamics_enabled: bool,
    #[pyo3(get, set)]
    pub imager_enabled: bool,
    #[pyo3(get, set)]
    pub maximizer_enabled: bool,
    #[pyo3(get, set)]
    pub target_lufs: f32,
    #[pyo3(get, set)]
    pub true_peak_limit: f32,
    #[pyo3(get, set)]
    pub dynamics_threshold: f64,
    #[pyo3(get, set)]
    pub dynamics_ratio: f64,
    #[pyo3(get, set)]
    pub imager_width: f32,
    #[pyo3(get, set)]
    pub maximizer_ceiling: f32,
}

#[pymethods]
impl PyBatchMasterConfig {
    #[new]
    #[pyo3(signature = (
        dynamics_enabled=true,
        imager_enabled=true,
        maximizer_enabled=true,
        target_lufs=-14.0,
        true_peak_limit=-1.0,
        dynamics_threshold=-18.0,
        dynamics_ratio=2.5,
        imager_width=100.0,
        maximizer_ceiling=-0.3,
    ))]
    fn new(
        dynamics_enabled: bool,
        imager_enabled: bool,
        maximizer_enabled: bool,
        target_lufs: f32,
        true_peak_limit: f32,
        dynamics_threshold: f64,
        dynamics_ratio: f64,
        imager_width: f32,
        maximizer_ceiling: f32,
    ) -> Self {
        Self {
            dynamics_enabled,
            imager_enabled,
            maximizer_enabled,
            target_lufs,
            true_peak_limit,
            dynamics_threshold,
            dynamics_ratio,
            imager_width,
            maximizer_ceiling,
        }
    }
}

impl From<&PyBatchMasterConfig> for rust_batch::BatchMasterConfig {
    fn from(py: &PyBatchMasterConfig) -> Self {
        Self {
            dynamics_enabled: py.dynamics_enabled,
            imager_enabled: py.imager_enabled,
            maximizer_enabled: py.maximizer_enabled,
            target_lufs: py.target_lufs,
            true_peak_limit: py.true_peak_limit,
            dynamics_threshold: py.dynamics_threshold,
            dynamics_ratio: py.dynamics_ratio,
            imager_width: py.imager_width,
            maximizer_ceiling: py.maximizer_ceiling,
        }
    }
}

/// Result of mastering a single song
#[pyclass]
#[derive(Clone)]
pub struct PyBatchMasterResult {
    #[pyo3(get)]
    pub input_path: String,
    #[pyo3(get)]
    pub output_path: String,
    #[pyo3(get)]
    pub success: bool,
    #[pyo3(get)]
    pub error: Option<String>,
    #[pyo3(get)]
    pub duration_sec: f64,
}

#[pymethods]
impl PyBatchMasterResult {
    fn __repr__(&self) -> String {
        if self.success {
            format!("BatchMasterResult(ok, {:.1}s, {})", self.duration_sec, self.output_path)
        } else {
            format!("BatchMasterResult(FAILED, {})", self.error.as_deref().unwrap_or("unknown"))
        }
    }
}

/// Batch master multiple songs in parallel using Rust + rayon.
///
/// Args:
///     input_paths: List of input audio file paths
///     output_paths: Corresponding output file paths (same length)
///     config: PyBatchMasterConfig (optional, uses defaults if None)
///     callback: Optional progress callback fn(completed: int, total: int, current_file: str)
///
/// Returns:
///     List of PyBatchMasterResult objects
#[pyfunction]
#[pyo3(signature = (input_paths, output_paths, config=None, callback=None))]
pub fn batch_master(
    py: Python<'_>,
    input_paths: Vec<String>,
    output_paths: Vec<String>,
    config: Option<&PyBatchMasterConfig>,
    callback: Option<Py<PyAny>>,
) -> PyResult<Vec<PyBatchMasterResult>> {
    if input_paths.len() != output_paths.len() {
        return Err(PyRuntimeError::new_err(
            "input_paths and output_paths must have the same length"
        ));
    }

    let rust_config = match config {
        Some(c) => rust_batch::BatchMasterConfig::from(c),
        None => rust_batch::BatchMasterConfig::default(),
    };

    // Build thread-safe progress callback that calls back into Python
    let progress_cb: Option<rust_batch::BatchProgressCallback> = callback.map(|py_cb| {
        let cb: rust_batch::BatchProgressCallback = Box::new(move |done, total, file: &str| {
            if let Some(py) = Python::try_attach(|py| {
                let _ = py_cb.call(py, (done, total, file), None);
            }) {
                // callback executed
            }
        });
        cb
    });

    // Run batch processing (releases GIL automatically via rayon)
    let results = rust_batch::batch_master(
        &input_paths,
        &output_paths,
        &rust_config,
        None,
        progress_cb.as_ref(),
    );

    // Convert to Python results
    let py_results: Vec<PyBatchMasterResult> = results
        .into_iter()
        .map(|r| PyBatchMasterResult {
            input_path: r.input_path,
            output_path: r.output_path,
            success: r.success,
            error: r.error,
            duration_sec: r.duration_sec,
        })
        .collect();

    Ok(py_results)
}

/// Get available CPU parallelism (number of threads rayon will use)
#[pyfunction]
pub fn get_parallelism() -> usize {
    rust_batch::available_parallelism()
}
