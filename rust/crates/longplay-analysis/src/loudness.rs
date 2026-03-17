//! Loudness measurement using FFmpeg loudnorm filter (ITU-R BS.1770-4)
//! Ported from C++ loudness.cpp

use std::path::Path;
use std::process::Command;

use regex::Regex;

use longplay_core::AudioBuffer;
use longplay_core::conversions::linear_to_db_f;

/// Results from loudness analysis of an audio file
#[derive(Debug, Clone)]
pub struct LoudnessAnalysis {
    // Primary output metrics
    /// Integrated LUFS (long-term average)
    pub integrated_lufs: f32,
    /// Loudness Range in LU (Loudness Units)
    pub loudness_range: f32,
    /// True Peak in dBTP (4x oversampled)
    pub true_peak_dbtp: f32,
    /// Maximum short-term loudness (3s window) in LUFS
    pub short_term_max: f32,
    /// Maximum momentary loudness (400ms window) in LUFS
    pub momentary_max: f32,

    /// Loudness Range (alias, same as loudness_range)
    pub lra: f32,
    /// Analysis threshold
    pub threshold: f32,

    // Audio file metadata
    pub sample_rate: i32,
    pub channels: i32,
    pub duration_sec: f64,

    // FFmpeg loudnorm input/output measurements
    /// Input integrated LUFS
    pub input_i: f32,
    /// Input true peak dBTP
    pub input_tp: f32,
    /// Input loudness range LU
    pub input_lra: f32,
    /// Input threshold
    pub input_thresh: f32,

    /// Output integrated LUFS (after normalization)
    pub output_i: f32,
    /// Output true peak dBTP
    pub output_tp: f32,
    /// Output loudness range LU
    pub output_lra: f32,
    /// Output threshold
    pub output_thresh: f32,

    // Normalization info
    /// Type of normalization applied
    pub normalization_type: String,
    /// Offset in LU for target alignment
    pub target_offset_lu: f32,
}

impl Default for LoudnessAnalysis {
    fn default() -> Self {
        Self {
            integrated_lufs: -24.0,
            loudness_range: 0.0,
            true_peak_dbtp: -1.0,
            short_term_max: -24.0,
            momentary_max: -24.0,
            lra: 0.0,
            threshold: -70.0,
            sample_rate: 48000,
            channels: 2,
            duration_sec: 0.0,
            input_i: -24.0,
            input_tp: -6.0,
            input_lra: 0.0,
            input_thresh: -70.0,
            output_i: -24.0,
            output_tp: -6.0,
            output_lra: 0.0,
            output_thresh: -70.0,
            normalization_type: String::new(),
            target_offset_lu: 0.0,
        }
    }
}

/// Loudness measurement using FFmpeg loudnorm filter.
///
/// Performs loudness analysis via FFmpeg's loudnorm filter implementing
/// ITU-R BS.1770-4 standard. Provides integrated LUFS, true peak dBTP,
/// loudness range (LU), short-term max, and momentary max.
///
/// Supports two-pass normalization workflow:
/// 1. First pass: `measure()` / `analyze()` - measure existing loudness
/// 2. Second pass: `normalize()` - apply loudness normalization to a file
pub struct LoudnessMeter {
    ffmpeg_path: String,
}

impl LoudnessMeter {
    /// Create a new LoudnessMeter.
    ///
    /// # Arguments
    /// * `ffmpeg_path` - Path to FFmpeg executable. If not found at the given
    ///   path, searches standard system locations and falls back to "ffmpeg".
    pub fn new(ffmpeg_path: &str) -> Self {
        let path = if ffmpeg_path.is_empty() {
            find_ffmpeg_executable("ffmpeg")
        } else {
            find_ffmpeg_executable(ffmpeg_path)
        };
        Self { ffmpeg_path: path }
    }

    /// Measure loudness of a file using FFmpeg loudnorm filter (JSON mode).
    ///
    /// This is the primary measurement method. Runs FFmpeg with the loudnorm
    /// filter in print_format=json mode and parses the full JSON output for
    /// detailed metrics.
    ///
    /// # Arguments
    /// * `file_path` - Path to audio file
    ///
    /// # Returns
    /// `Some(LoudnessAnalysis)` with results, or `None` on error.
    pub fn measure(&self, file_path: &str) -> Option<LoudnessAnalysis> {
        if !Path::new(file_path).exists() {
            return None;
        }

        // Run FFmpeg with loudnorm in JSON measurement mode
        let output = Command::new(&self.ffmpeg_path)
            .args([
                "-i", file_path,
                "-af", "loudnorm=I=-14:dual_mono=true:TP=-1.0:LRA=11:print_format=json",
                "-f", "null",
                "-",
            ])
            .output()
            .ok()?;

        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        // Parse the loudnorm JSON output
        let mut analysis = parse_loudnorm_output(&stderr)?;

        // Extract duration from FFmpeg output: "Duration: HH:MM:SS.ms"
        if let Ok(re) = Regex::new(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)") {
            if let Some(caps) = re.captures(&stderr) {
                if let (Some(h), Some(m), Some(s)) = (caps.get(1), caps.get(2), caps.get(3)) {
                    let hours: f64 = h.as_str().parse().unwrap_or(0.0);
                    let minutes: f64 = m.as_str().parse().unwrap_or(0.0);
                    let seconds: f64 = s.as_str().parse().unwrap_or(0.0);
                    analysis.duration_sec = hours * 3600.0 + minutes * 60.0 + seconds;
                }
            }
        }

        // Extract sample rate (e.g. "48000 Hz")
        if let Ok(re) = Regex::new(r"(\d+)\s*Hz") {
            if let Some(caps) = re.captures(&stderr) {
                if let Some(m) = caps.get(1) {
                    if let Ok(sr) = m.as_str().parse::<i32>() {
                        analysis.sample_rate = sr;
                    }
                }
            }
        }

        // Extract channels (mono/stereo/5.1/7.1)
        if let Ok(re) = Regex::new(r"([Mm]ono|[Ss]tereo|5\.1|7\.1)") {
            if let Some(caps) = re.captures(&stderr) {
                if let Some(m) = caps.get(1) {
                    analysis.channels = match m.as_str().to_lowercase().as_str() {
                        "mono" => 1,
                        "stereo" => 2,
                        "5.1" => 6,
                        "7.1" => 8,
                        _ => 2,
                    };
                }
            }
        }

        // Try to extract short-term and momentary max from ebur128 stats
        // These appear in FFmpeg output when using the ebur128 filter
        extract_ebur128_stats(&stderr, &mut analysis);

        Some(analysis)
    }

    /// Analyze loudness of a file.
    ///
    /// Convenience wrapper that tries FFmpeg-based measurement first,
    /// then falls back to RMS-based estimation from the decoded buffer.
    ///
    /// # Arguments
    /// * `file_path` - Path to audio file
    pub fn analyze(&self, file_path: &str) -> Option<LoudnessAnalysis> {
        // Try full measurement first
        if let Some(analysis) = self.measure(file_path) {
            return Some(analysis);
        }

        // Fallback: read audio via longplay_io and estimate from buffer
        let (buffer, info) = longplay_io::read_audio(file_path).ok()?;
        let (lufs_est, tp) = self.estimate_from_buffer(&buffer);
        Some(LoudnessAnalysis {
            integrated_lufs: lufs_est,
            true_peak_dbtp: tp,
            loudness_range: 0.0,
            lra: 0.0,
            short_term_max: lufs_est,
            momentary_max: lufs_est,
            duration_sec: info.duration_seconds,
            sample_rate: info.sample_rate,
            channels: info.channels,
            ..LoudnessAnalysis::default()
        })
    }

    /// Quick measurement returning (integrated_lufs, true_peak_dbtp).
    ///
    /// # Arguments
    /// * `file_path` - Path to audio file
    pub fn quick_measure(&self, file_path: &str) -> Option<(f32, f32)> {
        let analysis = self.analyze(file_path)?;
        Some((analysis.integrated_lufs, analysis.true_peak_dbtp))
    }

    /// Normalize an audio file to a target loudness using FFmpeg two-pass loudnorm.
    ///
    /// First measures the file (if no analysis is provided), then runs a second
    /// FFmpeg pass with the measured values for accurate linear normalization.
    ///
    /// # Arguments
    /// * `input_path` - Path to input audio file
    /// * `output_path` - Path for the normalized output file
    /// * `target_lufs` - Target integrated LUFS (e.g. -14.0 for streaming)
    /// * `target_tp` - Target true peak dBTP (e.g. -1.0)
    /// * `target_lra` - Target loudness range LU (e.g. 11.0)
    ///
    /// # Returns
    /// `Some(LoudnessAnalysis)` from the first-pass measurement, or `None` on failure.
    pub fn normalize(
        &self,
        input_path: &str,
        output_path: &str,
        target_lufs: f32,
        target_tp: f32,
        target_lra: f32,
    ) -> Option<LoudnessAnalysis> {
        // First pass: measure
        let analysis = self.measure(input_path)?;

        // Build the second-pass loudnorm filter with measured values
        let filter = self.get_loudnorm_filter(&analysis, target_lufs, target_tp, target_lra);

        // Second pass: apply normalization
        let status = Command::new(&self.ffmpeg_path)
            .args([
                "-y",
                "-i", input_path,
                "-af", &filter,
                output_path,
            ])
            .stderr(std::process::Stdio::null())
            .status()
            .ok()?;

        if status.success() {
            Some(analysis)
        } else {
            None
        }
    }

    /// Generate FFmpeg loudnorm filter string for 2nd pass normalization.
    ///
    /// Creates a loudnorm filter string using measured values from `measure()`
    /// for accurate linear normalization.
    ///
    /// # Arguments
    /// * `analysis` - LoudnessAnalysis from first pass
    /// * `target_lufs` - Target integrated LUFS (typically -14.0)
    /// * `target_tp` - Target true peak dBTP (typically -1.0)
    /// * `target_lra` - Target loudness range LU (typically 11.0)
    pub fn get_loudnorm_filter(
        &self,
        analysis: &LoudnessAnalysis,
        target_lufs: f32,
        target_tp: f32,
        target_lra: f32,
    ) -> String {
        format!(
            "loudnorm=I={}:TP={}:LRA={}:measured_I={}:measured_TP={}:measured_LRA={}:measured_thresh={}:offset={}:linear=true:print_format=summary",
            target_lufs,
            target_tp,
            target_lra,
            analysis.input_i,
            analysis.input_tp,
            analysis.input_lra,
            analysis.input_thresh,
            analysis.target_offset_lu,
        )
    }

    /// Check if analysis meets platform loudness target.
    ///
    /// # Arguments
    /// * `analysis` - LoudnessAnalysis from `measure()`
    /// * `target_lufs` - Target loudness in LUFS
    /// * `target_tp` - Target true peak in dBTP
    /// * `lufs_tolerance` - Maximum allowed deviation in LUFS (default 1.0 LU)
    /// * `tp_tolerance` - Maximum allowed deviation in dBTP (default 0.1 dB)
    pub fn meets_target(
        analysis: &LoudnessAnalysis,
        target_lufs: f32,
        target_tp: f32,
        lufs_tolerance: f32,
        tp_tolerance: f32,
    ) -> bool {
        let lufs_delta = (analysis.integrated_lufs - target_lufs).abs();
        let lufs_ok = lufs_delta <= lufs_tolerance;

        let tp_delta = analysis.true_peak_dbtp - target_tp;
        let tp_ok = tp_delta <= tp_tolerance;

        lufs_ok && tp_ok
    }

    /// Estimate loudness from an in-memory buffer using RMS (approximate LUFS proxy).
    ///
    /// Returns (estimated_lufs, true_peak_dbtp). The LUFS estimate is based on
    /// RMS which approximates integrated LUFS within ~3 LU for most material.
    pub fn estimate_from_buffer(&self, buffer: &AudioBuffer) -> (f32, f32) {
        if buffer.is_empty() || buffer[0].is_empty() {
            return (-70.0, -70.0);
        }

        let mut sum_sq = 0.0f64;
        let mut peak = 0.0f32;
        let mut total_samples = 0usize;

        for channel in buffer {
            for &sample in channel {
                sum_sq += (sample as f64) * (sample as f64);
                peak = peak.max(sample.abs());
                total_samples += 1;
            }
        }

        if total_samples == 0 {
            return (-70.0, -70.0);
        }

        let rms = (sum_sq / total_samples as f64).sqrt();
        let rms_db = if rms > 1e-10 {
            20.0 * rms.log10()
        } else {
            -70.0
        };
        let peak_db = linear_to_db_f(peak);

        (rms_db as f32, peak_db)
    }
}

impl Default for LoudnessMeter {
    fn default() -> Self {
        Self::new("ffmpeg")
    }
}

// ============================================================================
// FFmpeg output parsing helpers
// ============================================================================

/// Parse FFmpeg loudnorm JSON output from stderr.
///
/// Implements two parsing strategies:
/// 1. Full JSON block extraction using regex
/// 2. Individual key-value pair extraction as fallback
fn parse_loudnorm_output(stderr_output: &str) -> Option<LoudnessAnalysis> {
    let mut analysis = LoudnessAnalysis::default();

    // Strategy 1: Try to extract full JSON block containing "input_i"
    let json_block_re = Regex::new(r#"\{\s*"[^"]*input_i[^}]*\}"#).ok()?;
    let json_str = json_block_re
        .find(stderr_output)
        .map(|m| m.as_str().to_string());

    if let Some(ref json) = json_str {
        if parse_kv_pairs(json, &mut analysis) {
            return Some(analysis);
        }
    }

    // Strategy 2: Extract individual values using regex patterns
    let patterns: Vec<(&str, &str)> = vec![
        (r#""input_i"\s*:\s*"?([-\d.]+)"?"#, "input_i"),
        (r#""input_tp"\s*:\s*"?([-\d.]+)"?"#, "input_tp"),
        (r#""input_lra"\s*:\s*"?([-\d.]+)"?"#, "input_lra"),
        (r#""input_thresh"\s*:\s*"?([-\d.]+)"?"#, "input_thresh"),
        (r#""output_i"\s*:\s*"?([-\d.]+)"?"#, "output_i"),
        (r#""output_tp"\s*:\s*"?([-\d.]+)"?"#, "output_tp"),
        (r#""output_lra"\s*:\s*"?([-\d.]+)"?"#, "output_lra"),
        (r#""output_thresh"\s*:\s*"?([-\d.]+)"?"#, "output_thresh"),
        (r#""target_offset"\s*:\s*"?([-\d.]+)"?"#, "target_offset"),
    ];

    let mut found_values = false;

    for (pattern_str, key) in &patterns {
        if let Ok(re) = Regex::new(pattern_str) {
            if let Some(caps) = re.captures(stderr_output) {
                if let Some(val_match) = caps.get(1) {
                    if let Ok(val) = val_match.as_str().parse::<f32>() {
                        match *key {
                            "input_i" => analysis.input_i = val,
                            "input_tp" => analysis.input_tp = val,
                            "input_lra" => analysis.input_lra = val,
                            "input_thresh" => analysis.input_thresh = val,
                            "output_i" => analysis.output_i = val,
                            "output_tp" => analysis.output_tp = val,
                            "output_lra" => analysis.output_lra = val,
                            "output_thresh" => analysis.output_thresh = val,
                            "target_offset" => analysis.target_offset_lu = val,
                            _ => {}
                        }
                        found_values = true;
                    }
                }
            }
        }
    }

    // Extract normalization_type
    if let Ok(re) = Regex::new(r#""normalization_type"\s*:\s*"([^"]*)""#) {
        if let Some(caps) = re.captures(stderr_output) {
            if let Some(val_match) = caps.get(1) {
                analysis.normalization_type = val_match.as_str().to_string();
                found_values = true;
            }
        }
    }

    if !found_values {
        return None;
    }

    // Set main analysis values from input measurements
    analysis.integrated_lufs = analysis.input_i;
    analysis.true_peak_dbtp = analysis.input_tp;
    analysis.lra = analysis.input_lra;
    analysis.loudness_range = analysis.input_lra;
    analysis.threshold = analysis.input_thresh;

    Some(analysis)
}

/// Parse key-value pairs from a JSON-like string into a LoudnessAnalysis.
fn parse_kv_pairs(json: &str, analysis: &mut LoudnessAnalysis) -> bool {
    let kv_re = match Regex::new(r#""([^"]+)"\s*:\s*"?([^",}\s]+)"?"#) {
        Ok(re) => re,
        Err(_) => return false,
    };

    let mut found = false;

    for caps in kv_re.captures_iter(json) {
        let key = match caps.get(1) {
            Some(m) => m.as_str(),
            None => continue,
        };
        let val_str = match caps.get(2) {
            Some(m) => m.as_str(),
            None => continue,
        };

        match key {
            "input_i" => {
                if let Ok(v) = val_str.parse::<f32>() {
                    analysis.input_i = v;
                    found = true;
                }
            }
            "input_tp" => {
                if let Ok(v) = val_str.parse::<f32>() {
                    analysis.input_tp = v;
                    found = true;
                }
            }
            "input_lra" => {
                if let Ok(v) = val_str.parse::<f32>() {
                    analysis.input_lra = v;
                    found = true;
                }
            }
            "input_thresh" => {
                if let Ok(v) = val_str.parse::<f32>() {
                    analysis.input_thresh = v;
                    found = true;
                }
            }
            "output_i" => {
                if let Ok(v) = val_str.parse::<f32>() {
                    analysis.output_i = v;
                    found = true;
                }
            }
            "output_tp" => {
                if let Ok(v) = val_str.parse::<f32>() {
                    analysis.output_tp = v;
                    found = true;
                }
            }
            "output_lra" => {
                if let Ok(v) = val_str.parse::<f32>() {
                    analysis.output_lra = v;
                    found = true;
                }
            }
            "output_thresh" => {
                if let Ok(v) = val_str.parse::<f32>() {
                    analysis.output_thresh = v;
                    found = true;
                }
            }
            "target_offset" => {
                if let Ok(v) = val_str.parse::<f32>() {
                    analysis.target_offset_lu = v;
                    found = true;
                }
            }
            "normalization_type" => {
                analysis.normalization_type = val_str.to_string();
                found = true;
            }
            _ => {}
        }
    }

    if found {
        analysis.integrated_lufs = analysis.input_i;
        analysis.true_peak_dbtp = analysis.input_tp;
        analysis.lra = analysis.input_lra;
        analysis.loudness_range = analysis.input_lra;
        analysis.threshold = analysis.input_thresh;
    }

    found
}

/// Try to extract EBU R128 short-term and momentary max from FFmpeg output.
///
/// FFmpeg's ebur128 filter outputs lines like:
///   "Summary: ... Short term: ... Momentary: ..."
/// This function parses those if present.
fn extract_ebur128_stats(stderr: &str, analysis: &mut LoudnessAnalysis) {
    // Short-term max
    if let Ok(re) = Regex::new(r"(?i)short[- ]?term\s*(?:max|loudness)?\s*:\s*([-\d.]+)") {
        if let Some(caps) = re.captures(stderr) {
            if let Some(m) = caps.get(1) {
                if let Ok(v) = m.as_str().parse::<f32>() {
                    analysis.short_term_max = v;
                }
            }
        }
    }

    // Momentary max
    if let Ok(re) = Regex::new(r"(?i)momentary\s*(?:max|loudness)?\s*:\s*([-\d.]+)") {
        if let Some(caps) = re.captures(stderr) {
            if let Some(m) = caps.get(1) {
                if let Ok(v) = m.as_str().parse::<f32>() {
                    analysis.momentary_max = v;
                }
            }
        }
    }
}

/// Find FFmpeg executable in system PATH or known locations.
fn find_ffmpeg_executable(provided_path: &str) -> String {
    if Path::new(provided_path).exists() {
        return provided_path.to_string();
    }

    let candidates = [
        "/usr/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "/opt/homebrew/bin/ffmpeg",
        "/opt/local/bin/ffmpeg",
    ];

    for candidate in &candidates {
        if Path::new(candidate).exists() {
            return candidate.to_string();
        }
    }

    provided_path.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loudness_analysis_default() {
        let analysis = LoudnessAnalysis::default();
        assert_eq!(analysis.integrated_lufs, -24.0);
        assert_eq!(analysis.true_peak_dbtp, -1.0);
        assert_eq!(analysis.loudness_range, 0.0);
        assert_eq!(analysis.short_term_max, -24.0);
        assert_eq!(analysis.momentary_max, -24.0);
    }

    #[test]
    fn test_parse_loudnorm_output_json_block() {
        let stderr = r#"
[Parsed_loudnorm_0 @ 0x7f8b8c004e00]
{
    "input_i" : "-18.53",
    "input_tp" : "-3.21",
    "input_lra" : "7.20",
    "input_thresh" : "-29.18",
    "output_i" : "-14.00",
    "output_tp" : "-0.95",
    "output_lra" : "5.80",
    "output_thresh" : "-24.65",
    "normalization_type" : "dynamic",
    "target_offset" : "0.00"
}
"#;

        let analysis = parse_loudnorm_output(stderr).unwrap();
        assert!((analysis.integrated_lufs - (-18.53)).abs() < 0.01);
        assert!((analysis.true_peak_dbtp - (-3.21)).abs() < 0.01);
        assert!((analysis.lra - 7.20).abs() < 0.01);
        assert!((analysis.loudness_range - 7.20).abs() < 0.01);
        assert!((analysis.input_thresh - (-29.18)).abs() < 0.01);
        assert!((analysis.output_i - (-14.00)).abs() < 0.01);
        assert!((analysis.output_tp - (-0.95)).abs() < 0.01);
        assert_eq!(analysis.normalization_type, "dynamic");
    }

    #[test]
    fn test_parse_loudnorm_output_empty() {
        let result = parse_loudnorm_output("");
        assert!(result.is_none());
    }

    #[test]
    fn test_meets_target() {
        let mut analysis = LoudnessAnalysis::default();
        analysis.integrated_lufs = -14.0;
        analysis.true_peak_dbtp = -1.0;

        assert!(LoudnessMeter::meets_target(&analysis, -14.0, -1.0, 1.0, 0.1));
        assert!(LoudnessMeter::meets_target(&analysis, -14.5, -1.0, 1.0, 0.1));
        assert!(!LoudnessMeter::meets_target(&analysis, -16.0, -1.0, 1.0, 0.1));
    }

    #[test]
    fn test_meets_target_tp_exceeded() {
        let mut analysis = LoudnessAnalysis::default();
        analysis.integrated_lufs = -14.0;
        analysis.true_peak_dbtp = -0.5;

        // TP is -0.5, target is -1.0, so delta is 0.5 which exceeds 0.1 tolerance
        assert!(!LoudnessMeter::meets_target(&analysis, -14.0, -1.0, 1.0, 0.1));
        // With 1.0 tolerance it should pass
        assert!(LoudnessMeter::meets_target(&analysis, -14.0, -1.0, 1.0, 1.0));
    }

    #[test]
    fn test_get_loudnorm_filter() {
        let mut analysis = LoudnessAnalysis::default();
        analysis.input_i = -18.5;
        analysis.input_tp = -3.2;
        analysis.input_lra = 7.2;
        analysis.input_thresh = -29.2;
        analysis.target_offset_lu = 0.0;

        let meter = LoudnessMeter::new("ffmpeg");
        let filter = meter.get_loudnorm_filter(&analysis, -14.0, -1.0, 11.0);

        assert!(filter.contains("loudnorm="));
        assert!(filter.contains("I=-14"));
        assert!(filter.contains("TP=-1"));
        assert!(filter.contains("LRA=11"));
        assert!(filter.contains("measured_I=-18.5"));
        assert!(filter.contains("measured_TP=-3.2"));
        assert!(filter.contains("linear=true"));
    }

    #[test]
    fn test_estimate_from_buffer_empty() {
        let meter = LoudnessMeter::new("ffmpeg");
        let buffer: AudioBuffer = Vec::new();
        let (lufs, tp) = meter.estimate_from_buffer(&buffer);
        assert_eq!(lufs, -70.0);
        assert_eq!(tp, -70.0);
    }

    #[test]
    fn test_estimate_from_buffer_sine() {
        let meter = LoudnessMeter::new("ffmpeg");
        let n = 44100;
        let signal: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();
        let buffer = vec![signal.clone(), signal];
        let (lufs, tp) = meter.estimate_from_buffer(&buffer);
        // RMS of sine is about -3 dB
        assert!(lufs > -5.0);
        assert!(lufs < 0.0);
        // Peak should be near 0 dB
        assert!(tp > -1.0);
    }

    #[test]
    fn test_find_ffmpeg_nonexistent() {
        let path = find_ffmpeg_executable("/nonexistent/path/ffmpeg");
        assert!(!path.is_empty());
    }

    #[test]
    fn test_extract_ebur128_stats() {
        let mut analysis = LoudnessAnalysis::default();
        let stderr = "Summary:\n  Short term max: -12.5 LUFS\n  Momentary max: -10.3 LUFS\n";
        extract_ebur128_stats(stderr, &mut analysis);
        assert!((analysis.short_term_max - (-12.5)).abs() < 0.01);
        assert!((analysis.momentary_max - (-10.3)).abs() < 0.01);
    }
}
