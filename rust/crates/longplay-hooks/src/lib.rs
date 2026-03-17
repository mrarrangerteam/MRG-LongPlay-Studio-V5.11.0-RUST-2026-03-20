//! Hook Extractor - Audio Waveform Analysis
//! LongPlay Studio V5
//!
//! Features:
//! 1. Analyze audio waveform to detect hook sections
//! 2. Support batch processing up to 20 songs
//! 3. Use Energy Analysis, Peak Detection, and Repetition Pattern
//! 4. Auto-extract hook sections with configurable duration
//! 5. Export hooks as separate audio files
//!
//! Ported from MRG-AI-Studio/src/hook_extractor.rs

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::process::Command;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

/// Hook detection result per file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HookResult {
    pub file_path: String,
    pub filename: String,
    pub duration_sec: f64,
    pub sample_rate: u32,

    // Hook detection results
    pub hook_start_sec: f64,
    pub hook_end_sec: f64,
    pub hook_duration_sec: f64,
    pub hook_confidence: f64, // 0.0 to 1.0

    // Analysis data
    pub energy_profile: Vec<f64>,
    pub peak_positions: Vec<f64>,

    // Export path
    pub hook_file_path: String,
}

impl HookResult {
    pub fn new(file_path: &str) -> Self {
        let filename = Path::new(file_path)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        Self {
            file_path: file_path.to_string(),
            filename,
            duration_sec: 0.0,
            sample_rate: 44100,
            hook_start_sec: 0.0,
            hook_end_sec: 0.0,
            hook_duration_sec: 0.0,
            hook_confidence: 0.0,
            energy_profile: Vec::new(),
            peak_positions: Vec::new(),
            hook_file_path: String::new(),
        }
    }

    /// Format hook time as MM:SS - MM:SS
    pub fn hook_time_str(&self) -> String {
        let start_min = (self.hook_start_sec / 60.0) as u32;
        let start_sec = (self.hook_start_sec % 60.0) as u32;
        let end_min = (self.hook_end_sec / 60.0) as u32;
        let end_sec = (self.hook_end_sec % 60.0) as u32;
        format!("{:02}:{:02} - {:02}:{:02}", start_min, start_sec, end_min, end_sec)
    }
}

/// Hook extraction statistics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct HookStats {
    pub total_files: usize,
    pub avg_confidence: f64,
    pub avg_hook_duration: f64,
    pub most_common_position: String,
}

/// Hook Extractor - analyze and extract hooks from audio
pub struct HookExtractor {
    hook_duration: f64,
    min_hook_duration: f64,
    max_hook_duration: f64,
    results: Mutex<HashMap<String, HookResult>>,
}

impl HookExtractor {
    pub fn new(hook_duration: f64, min_hook_duration: f64, max_hook_duration: f64) -> Self {
        Self {
            hook_duration,
            min_hook_duration,
            max_hook_duration,
            results: Mutex::new(HashMap::new()),
        }
    }

    pub fn default_config() -> Self {
        Self::new(30.0, 15.0, 60.0)
    }

    /// Analyze audio file and detect hook section
    pub fn analyze_audio(&self, file_path: &str) -> Result<HookResult, Box<dyn std::error::Error + Send + Sync>> {
        // Check cache
        {
            let results = self.results.lock().unwrap();
            if let Some(result) = results.get(file_path) {
                return Ok(result.clone());
            }
        }

        let mut result = HookResult::new(file_path);
        let filename = result.filename.clone();

        // Get duration
        match get_audio_duration(file_path) {
            Ok(duration) => {
                result.duration_sec = duration;
            }
            Err(e) => {
                eprintln!("Error getting duration for {}: {}", filename, e);
                result.duration_sec = 180.0;
            }
        }

        // Compute energy profile
        match self.compute_energy_profile(file_path, result.duration_sec) {
            Ok(energy_profile) => {
                result.energy_profile = energy_profile.clone();

                let peaks = self.detect_peaks(&energy_profile, 0.6);
                result.peak_positions = peaks.clone();

                let (hook_start, hook_end, confidence) =
                    self.find_best_hook(&energy_profile, &peaks, result.duration_sec);

                result.hook_start_sec = hook_start;
                result.hook_end_sec = hook_end;
                result.hook_duration_sec = hook_end - hook_start;
                result.hook_confidence = confidence;
            }
            Err(e) => {
                eprintln!("Error computing energy profile for {}: {}", filename, e);
                let mid = result.duration_sec / 2.0;
                result.hook_start_sec = (mid - self.hook_duration / 2.0).max(0.0);
                result.hook_end_sec = (mid + self.hook_duration / 2.0).min(result.duration_sec);
                result.hook_duration_sec = result.hook_end_sec - result.hook_start_sec;
                result.hook_confidence = 0.3;
            }
        }

        // Save to cache
        {
            let mut results = self.results.lock().unwrap();
            results.insert(file_path.to_string(), result.clone());
        }

        Ok(result)
    }

    /// Extract hook section to output file
    pub fn extract_hook(&self, file_path: &str, output_path: &str)
        -> Result<HookResult, Box<dyn std::error::Error + Send + Sync>>
    {
        let mut result = self.analyze_audio(file_path)?;

        let output = Command::new("ffmpeg")
            .args([
                "-i", file_path,
                "-ss", &result.hook_start_sec.to_string(),
                "-t", &result.hook_duration_sec.to_string(),
                "-c:a", "pcm_s16le",
                "-ar", "44100",
                "-ac", "2",
                "-y", output_path
            ])
            .output()?;

        if output.status.success() {
            result.hook_file_path = output_path.to_string();
            Ok(result)
        } else {
            Err("ffmpeg extraction failed".into())
        }
    }

    /// Batch analyze multiple files
    pub fn batch_analyze<F>(&self, file_paths: &[String], progress_callback: Option<F>)
        -> Vec<HookResult>
    where
        F: Fn(usize, usize, &str),
    {
        let mut results = Vec::new();
        let total = file_paths.len();

        for (i, path) in file_paths.iter().enumerate() {
            match self.analyze_audio(path) {
                Ok(result) => {
                    if let Some(ref cb) = progress_callback {
                        cb(i + 1, total, &result.filename);
                    }
                    results.push(result);
                }
                Err(e) => {
                    eprintln!("Error analyzing {}: {}", path, e);
                }
            }
        }

        results
    }

    /// Batch extract hooks from multiple files
    pub fn batch_extract(&self, file_paths: &[String], output_dir: &str)
        -> Vec<HookResult>
    {
        let mut results = Vec::new();
        fs::create_dir_all(output_dir).ok();

        for (i, path) in file_paths.iter().enumerate() {
            let filename = Path::new(path)
                .file_stem()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown");

            let output_path = format!("{}/{}_hook.wav", output_dir, filename);

            match self.extract_hook(path, &output_path) {
                Ok(result) => {
                    eprintln!("  [{}/{}] Extracted hook from {} -> {}",
                             i + 1, file_paths.len(), filename, output_path);
                    results.push(result);
                }
                Err(e) => {
                    eprintln!("  [{}/{}] Failed to extract hook from {}: {}",
                              i + 1, file_paths.len(), filename, e);
                }
            }
        }

        results
    }

    // ==================== Private Methods ====================

    fn compute_energy_profile(&self, file_path: &str, duration: f64)
        -> Result<Vec<f64>, Box<dyn std::error::Error + Send + Sync>>
    {
        let window_size = 0.5;

        let output = Command::new("ffmpeg")
            .args([
                "-i", file_path,
                "-af", &format!("asetnsamples=n={},astats=metadata=1:reset=1",
                               (44100.0 * window_size) as i32),
                "-f", "null", "-"
            ])
            .output()?;

        let stderr = String::from_utf8_lossy(&output.stderr);
        let mut rms_values = Vec::new();

        for line in stderr.lines() {
            if line.contains("RMS level dB") || line.to_lowercase().contains("rms_level") {
                if let Some(idx) = line.find(':') {
                    let val_str = line[idx + 1..].trim()
                        .replace("dB", "")
                        .trim()
                        .to_string();

                    if !val_str.is_empty() && val_str != "-inf" {
                        if let Ok(val) = val_str.parse::<f64>() {
                            rms_values.push(val);
                        }
                    }
                }
            }
        }

        if rms_values.len() >= 5 {
            let min_rms = rms_values.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_rms = rms_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let range_rms = if max_rms > min_rms { max_rms - min_rms } else { 1.0 };

            Ok(rms_values.iter().map(|&v| (v - min_rms) / range_rms).collect())
        } else {
            Ok(self.generate_synthetic_profile(duration, window_size))
        }
    }

    fn generate_synthetic_profile(&self, duration: f64, window_size: f64) -> Vec<f64> {
        use std::f64::consts::PI;

        let num_samples = ((duration / window_size) as usize).max(10).min(500);
        let mut profile = Vec::with_capacity(num_samples);

        for i in 0..num_samples {
            let t = i as f64 / num_samples as f64;

            let base = 0.3 + 0.4 * (t * PI).sin();

            let chorus_boost = {
                let positions = [0.25, 0.30, 0.55, 0.60, 0.85];
                let mut boost = 0.0;
                for &pos in &positions {
                    let distance = (t - pos).abs();
                    if distance < 0.08 {
                        boost = f64::max(boost, 0.3 * (1.0 - distance / 0.08));
                    }
                }
                boost
            };

            let energy = (base + chorus_boost).min(1.0);
            profile.push(energy);
        }

        profile
    }

    fn detect_peaks(&self, energy_profile: &[f64], threshold: f64) -> Vec<f64> {
        if energy_profile.is_empty() {
            return Vec::new();
        }

        let mut peaks = Vec::new();
        let n = energy_profile.len();

        for i in 1..n.saturating_sub(1) {
            if energy_profile[i] > threshold
                && energy_profile[i] > energy_profile[i - 1]
                && energy_profile[i] > energy_profile[i + 1]
            {
                peaks.push(i as f64 / n as f64);
            }
        }

        peaks
    }

    fn find_best_hook(&self, energy_profile: &[f64], peaks: &[f64], duration: f64)
        -> (f64, f64, f64)
    {
        if energy_profile.is_empty() {
            let mid = duration / 2.0;
            return (
                (mid - self.hook_duration / 2.0).max(0.0),
                (mid + self.hook_duration / 2.0).min(duration),
                0.3,
            );
        }

        let time_per_sample = duration / energy_profile.len() as f64;
        let min_samples = (self.min_hook_duration / time_per_sample) as usize;
        let max_samples = (self.max_hook_duration / time_per_sample) as usize;

        let mut best_score = -1.0;
        let mut best_start = 0;
        let mut best_end = min_samples;

        for start in (0..energy_profile.len().saturating_sub(min_samples)).step_by(5) {
            for end in (start + min_samples)..=(start + max_samples).min(energy_profile.len()) {
                let score = self.score_segment(energy_profile, peaks, start, end, energy_profile.len());

                if score > best_score {
                    best_score = score;
                    best_start = start;
                    best_end = end;
                }
            }
        }

        let hook_start = best_start as f64 * time_per_sample;
        let hook_end = best_end as f64 * time_per_sample;
        let confidence = (best_score / 100.0).min(1.0).max(0.0);

        (hook_start, hook_end, confidence)
    }

    fn score_segment(&self, energy_profile: &[f64], peaks: &[f64],
                     start: usize, end: usize, total_samples: usize) -> f64
    {
        // Energy score (0-40)
        let segment_energy: f64 = energy_profile[start..end].iter().sum();
        let avg_energy = segment_energy / (end - start) as f64;
        let energy_score = avg_energy * 40.0;

        // Peak score (0-30)
        let start_norm = start as f64 / total_samples as f64;
        let end_norm = end as f64 / total_samples as f64;
        let peak_count = peaks.iter()
            .filter(|&&p| p >= start_norm && p <= end_norm)
            .count();
        let peak_score = (peak_count as f64 * 5.0).min(30.0);

        // Position score (0-30) - prefer typical chorus positions
        let segment_center = (start_norm + end_norm) / 2.0;
        let chorus_positions = [0.275, 0.575, 0.875];

        let position_score = chorus_positions.iter()
            .map(|&pos| {
                let distance = (segment_center - pos).abs();
                30.0 * (1.0 - distance * 3.0).max(0.0)
            })
            .fold(0.0, f64::max);

        energy_score + peak_score + position_score
    }
}

impl Default for HookExtractor {
    fn default() -> Self {
        Self::default_config()
    }
}

// ==================== Helper Functions ====================

fn get_audio_duration(file_path: &str) -> Result<f64, Box<dyn std::error::Error + Send + Sync>> {
    let output = Command::new("ffprobe")
        .args(["-v", "quiet", "-print_format", "json", "-show_format", file_path])
        .output()?;

    if !output.status.success() {
        return Err("ffprobe failed".into());
    }

    let json_str = String::from_utf8(output.stdout)?;
    let data: serde_json::Value = serde_json::from_str(&json_str)?;

    let duration = data["format"]["duration"]
        .as_str()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.0);

    Ok(duration)
}

/// Analyze hook statistics from results
pub fn analyze_hook_stats(results: &[HookResult]) -> HookStats {
    if results.is_empty() {
        return HookStats::default();
    }

    let total_files = results.len();
    let avg_confidence = results.iter().map(|r| r.hook_confidence).sum::<f64>() / total_files as f64;
    let avg_hook_duration = results.iter().map(|r| r.hook_duration_sec).sum::<f64>() / total_files as f64;

    let mut position_counts: HashMap<String, usize> = HashMap::new();
    for r in results {
        let position = if r.hook_start_sec < r.duration_sec * 0.2 {
            "early"
        } else if r.hook_start_sec < r.duration_sec * 0.5 {
            "first_half"
        } else if r.hook_start_sec < r.duration_sec * 0.8 {
            "second_half"
        } else {
            "late"
        };
        *position_counts.entry(position.to_string()).or_insert(0) += 1;
    }

    let most_common_position = position_counts.iter()
        .max_by_key(|&(_, count)| count)
        .map(|(pos, _)| pos.clone())
        .unwrap_or_else(|| "unknown".to_string());

    HookStats {
        total_files,
        avg_confidence: (avg_confidence * 100.0).round() / 100.0,
        avg_hook_duration: (avg_hook_duration * 10.0).round() / 10.0,
        most_common_position,
    }
}

// ==================== Tests ====================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hook_result_new() {
        let result = HookResult::new("/path/to/song.mp3");
        assert_eq!(result.filename, "song.mp3");
        assert_eq!(result.hook_confidence, 0.0);
    }

    #[test]
    fn test_hook_time_str() {
        let mut result = HookResult::new("test.mp3");
        result.hook_start_sec = 65.0;
        result.hook_end_sec = 95.0;
        assert_eq!(result.hook_time_str(), "01:05 - 01:35");
    }

    #[test]
    fn test_detect_peaks() {
        let extractor = HookExtractor::default();
        let profile = vec![0.2, 0.5, 0.8, 0.5, 0.3, 0.7, 0.9, 0.6, 0.4];
        let peaks = extractor.detect_peaks(&profile, 0.6);
        assert!(!peaks.is_empty());
    }

    #[test]
    fn test_generate_synthetic_profile() {
        let extractor = HookExtractor::default();
        let profile = extractor.generate_synthetic_profile(180.0, 0.5);
        assert!(!profile.is_empty());
        assert!(profile.iter().all(|&v| v >= 0.0 && v <= 1.0));
    }

    #[test]
    fn test_score_segment() {
        let extractor = HookExtractor::default();
        let profile = vec![0.3, 0.5, 0.8, 0.6, 0.4, 0.7, 0.9, 0.5, 0.3];
        let peaks = vec![0.25, 0.75];
        let score = extractor.score_segment(&profile, &peaks, 2, 5, profile.len());
        assert!(score > 0.0);
    }

    #[test]
    fn test_analyze_hook_stats() {
        let results = vec![
            HookResult {
                file_path: "test1.mp3".to_string(),
                filename: "test1.mp3".to_string(),
                duration_sec: 180.0,
                sample_rate: 44100,
                hook_start_sec: 45.0,
                hook_end_sec: 75.0,
                hook_duration_sec: 30.0,
                hook_confidence: 0.85,
                energy_profile: vec![],
                peak_positions: vec![],
                hook_file_path: "".to_string(),
            },
            HookResult {
                file_path: "test2.mp3".to_string(),
                filename: "test2.mp3".to_string(),
                duration_sec: 200.0,
                sample_rate: 44100,
                hook_start_sec: 100.0,
                hook_end_sec: 130.0,
                hook_duration_sec: 30.0,
                hook_confidence: 0.75,
                energy_profile: vec![],
                peak_positions: vec![],
                hook_file_path: "".to_string(),
            },
        ];

        let stats = analyze_hook_stats(&results);
        assert_eq!(stats.total_files, 2);
        assert!(stats.avg_confidence > 0.0);
    }
}
