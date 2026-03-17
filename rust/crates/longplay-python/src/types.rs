//! Shared PyO3 return types

use pyo3::prelude::*;

/// AI Mastering Recommendation returned to Python
#[pyclass]
#[derive(Clone)]
pub struct PyMasterRecommendation {
    #[pyo3(get)]
    pub genre: String,
    #[pyo3(get)]
    pub platform: String,
    #[pyo3(get)]
    pub intensity: f32,
    #[pyo3(get)]
    pub confidence: f32,
    #[pyo3(get)]
    pub explanations: Vec<String>,
}

#[pymethods]
impl PyMasterRecommendation {
    fn __repr__(&self) -> String {
        format!(
            "MasterRecommendation(genre='{}', platform='{}', intensity={:.0}, confidence={:.0}%)",
            self.genre, self.platform, self.intensity, self.confidence
        )
    }

    fn __str__(&self) -> String {
        let mut s = format!("Genre: {}\nPlatform: {}\nIntensity: {:.0}\nConfidence: {:.0}%\n",
            self.genre, self.platform, self.intensity, self.confidence);
        if !self.explanations.is_empty() {
            s.push_str("Explanations:\n");
            for exp in &self.explanations {
                s.push_str(&format!("  - {}\n", exp));
            }
        }
        s
    }
}

/// Audio analysis result returned from AI DJ
#[pyclass]
#[derive(Clone)]
pub struct PyAudioAnalysis {
    #[pyo3(get)]
    pub file_path: String,
    #[pyo3(get)]
    pub filename: String,
    #[pyo3(get)]
    pub duration_sec: f64,
    #[pyo3(get)]
    pub bpm: f64,
    #[pyo3(get)]
    pub key: String,
    #[pyo3(get)]
    pub energy: f64,
    #[pyo3(get)]
    pub loudness_db: f64,
    #[pyo3(get)]
    pub intro_score: f64,
}

#[pymethods]
impl PyAudioAnalysis {
    fn __repr__(&self) -> String {
        format!(
            "AudioAnalysis(file='{}', bpm={:.1}, key='{}', energy={:.2})",
            self.filename, self.bpm, self.key, self.energy
        )
    }

    #[getter]
    fn energy_bars(&self) -> String {
        let filled = (self.energy * 6.0) as usize;
        let filled = filled.min(6);
        "\u{2588}".repeat(filled) + &"\u{2591}".repeat(6 - filled)
    }

    #[getter]
    fn bpm_category(&self) -> &'static str {
        if self.bpm < 80.0 {
            "slow"
        } else if self.bpm < 110.0 {
            "medium"
        } else if self.bpm < 140.0 {
            "upbeat"
        } else {
            "fast"
        }
    }
}

impl From<longplay_aidj::AudioAnalysis> for PyAudioAnalysis {
    fn from(a: longplay_aidj::AudioAnalysis) -> Self {
        Self {
            file_path: a.file_path,
            filename: a.filename,
            duration_sec: a.duration_sec,
            bpm: a.bpm,
            key: a.key,
            energy: a.energy,
            loudness_db: a.loudness_db,
            intro_score: a.intro_score,
        }
    }
}

/// Hook detection result returned from HookExtractor
#[pyclass]
#[derive(Clone)]
pub struct PyHookResult {
    #[pyo3(get)]
    pub file_path: String,
    #[pyo3(get)]
    pub filename: String,
    #[pyo3(get)]
    pub duration_sec: f64,
    #[pyo3(get)]
    pub sample_rate: u32,
    #[pyo3(get)]
    pub hook_start_sec: f64,
    #[pyo3(get)]
    pub hook_end_sec: f64,
    #[pyo3(get)]
    pub hook_duration_sec: f64,
    #[pyo3(get)]
    pub hook_confidence: f64,
    #[pyo3(get)]
    pub energy_profile: Vec<f64>,
    #[pyo3(get)]
    pub peak_positions: Vec<f64>,
    #[pyo3(get)]
    pub hook_file_path: String,
}

#[pymethods]
impl PyHookResult {
    fn __repr__(&self) -> String {
        format!(
            "HookResult(file='{}', hook={}, confidence={:.0}%)",
            self.filename, self.hook_time_str(), self.hook_confidence * 100.0
        )
    }

    #[getter]
    fn hook_time_str(&self) -> String {
        let start_min = (self.hook_start_sec / 60.0) as u32;
        let start_sec = (self.hook_start_sec % 60.0) as u32;
        let end_min = (self.hook_end_sec / 60.0) as u32;
        let end_sec = (self.hook_end_sec % 60.0) as u32;
        format!("{:02}:{:02} - {:02}:{:02}", start_min, start_sec, end_min, end_sec)
    }
}

impl From<longplay_hooks::HookResult> for PyHookResult {
    fn from(r: longplay_hooks::HookResult) -> Self {
        Self {
            file_path: r.file_path,
            filename: r.filename,
            duration_sec: r.duration_sec,
            sample_rate: r.sample_rate,
            hook_start_sec: r.hook_start_sec,
            hook_end_sec: r.hook_end_sec,
            hook_duration_sec: r.hook_duration_sec,
            hook_confidence: r.hook_confidence,
            energy_profile: r.energy_profile,
            peak_positions: r.peak_positions,
            hook_file_path: r.hook_file_path,
        }
    }
}
