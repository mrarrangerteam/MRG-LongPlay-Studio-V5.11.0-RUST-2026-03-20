//! PyHookExtractor - PyO3 binding for Hook Extractor

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;

use longplay_hooks::HookExtractor;
use crate::types::PyHookResult;

/// Python-facing Hook Extractor wrapper
#[pyclass]
pub struct PyHookExtractor {
    inner: HookExtractor,
}

#[pymethods]
impl PyHookExtractor {
    #[new]
    #[pyo3(signature = (hook_duration=30.0, min_hook_duration=15.0, max_hook_duration=60.0))]
    fn new(hook_duration: f64, min_hook_duration: f64, max_hook_duration: f64) -> Self {
        Self {
            inner: HookExtractor::new(hook_duration, min_hook_duration, max_hook_duration),
        }
    }

    /// Analyze audio and detect hook section
    fn analyze_audio(&self, file_path: &str) -> PyResult<PyHookResult> {
        self.inner
            .analyze_audio(file_path)
            .map(PyHookResult::from)
            .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))
    }

    /// Extract hook section to output file, returns output path
    fn extract_hook(&self, file_path: &str, output_dir: &str) -> PyResult<String> {
        let filename = std::path::Path::new(file_path)
            .file_stem()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");

        let output_path = format!("{}/{}_hook.wav", output_dir, filename);

        std::fs::create_dir_all(output_dir)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create output dir: {}", e)))?;

        let result = self.inner
            .extract_hook(file_path, &output_path)
            .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))?;

        Ok(result.hook_file_path)
    }

    /// Batch analyze multiple files
    #[pyo3(signature = (file_paths, callback=None))]
    fn batch_analyze(&self, file_paths: Vec<String>, callback: Option<Py<PyAny>>) -> Vec<PyHookResult> {
        let _ = callback; // TODO: wire up Python callback
        self.inner
            .batch_analyze(&file_paths, None::<fn(usize, usize, &str)>)
            .into_iter()
            .map(PyHookResult::from)
            .collect()
    }

    /// Batch extract hooks from multiple files
    fn batch_extract(&self, file_paths: Vec<String>, output_dir: &str) -> Vec<PyHookResult> {
        self.inner
            .batch_extract(&file_paths, output_dir)
            .into_iter()
            .map(PyHookResult::from)
            .collect()
    }
}
