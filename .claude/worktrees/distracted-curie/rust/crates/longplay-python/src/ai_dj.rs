//! PyAIDJ - PyO3 binding for AI DJ

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;

use longplay_aidj::AiDj;
use crate::types::PyAudioAnalysis;

/// Python-facing AI DJ wrapper
#[pyclass]
pub struct PyAIDJ {
    inner: AiDj,
}

#[pymethods]
impl PyAIDJ {
    #[new]
    fn new() -> Self {
        Self {
            inner: AiDj::new(),
        }
    }

    /// Analyze a single track, returns PyAudioAnalysis
    fn analyze_track(&self, file_path: &str) -> PyResult<PyAudioAnalysis> {
        self.inner
            .analyze_track(file_path)
            .map(PyAudioAnalysis::from)
            .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))
    }

    /// Suggest playlist order with given strategy
    /// strategy: "smooth", "energy_up", "energy_down", "random_smart"
    fn suggest_order(&self, file_paths: Vec<String>, strategy: &str) -> Vec<String> {
        self.inner.suggest_order(&file_paths, strategy)
    }

    /// Get best opener tracks, returns list of (path, score) tuples
    #[pyo3(signature = (file_paths, top_n=3))]
    fn get_best_opener(&self, file_paths: Vec<String>, top_n: usize) -> Vec<(String, f64)> {
        self.inner.get_best_opener(&file_paths, top_n)
    }

    /// Get playlist statistics as a dict
    fn get_playlist_stats<'py>(&self, py: Python<'py>, file_paths: Vec<String>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let stats = self.inner.get_playlist_stats(&file_paths);

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("smoothness", stats.smoothness)?;
        dict.set_item("energy_balance", stats.energy_balance)?;
        dict.set_item("avg_bpm", stats.avg_bpm)?;
        dict.set_item("avg_energy", stats.avg_energy)?;
        dict.set_item("total_duration_sec", stats.total_duration_sec)?;
        dict.set_item("track_count", stats.track_count)?;

        Ok(dict)
    }

    /// Generate a new unique shuffle
    fn shuffle_again(&self, file_paths: Vec<String>) -> Vec<String> {
        self.inner.shuffle_again(&file_paths)
    }
}
