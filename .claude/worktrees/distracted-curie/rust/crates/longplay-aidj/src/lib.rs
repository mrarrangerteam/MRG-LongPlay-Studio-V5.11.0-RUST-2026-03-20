//! AI DJ Module for LongPlay Studio V5
//!
//! Features:
//! 1. Audio analysis (BPM, Key, Energy, Mood)
//! 2. Smart playlist ordering
//! 3. Best #1 track suggestion
//! 4. Multiple shuffle versions
//!
//! Ported from MRG-AI-Studio/src/ai_dj.rs

use std::collections::HashMap;
use std::path::Path;
use std::process::Command;
use std::sync::Mutex;

use ndarray::{Array1, Array2};
use rustfft::{FftPlanner, num_complex::Complex};
use serde::{Deserialize, Serialize};
use regex::Regex;
use rand::seq::SliceRandom;
use rand::Rng;
use rand::thread_rng;

/// Audio analysis result per track
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioAnalysis {
    pub file_path: String,
    pub filename: String,
    pub duration_sec: f64,
    pub bpm: f64,
    pub key: String,
    pub energy: f64,        // 0-1 scale
    pub loudness_db: f64,
    pub intro_score: f64,   // 0-100
}

impl AudioAnalysis {
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
            bpm: 0.0,
            key: String::new(),
            energy: 0.0,
            loudness_db: -14.0,
            intro_score: 0.0,
        }
    }

    /// Visual energy bars
    pub fn energy_bars(&self) -> String {
        let filled = (self.energy * 6.0) as usize;
        let filled = filled.min(6);
        "\u{2588}".repeat(filled) + &"\u{2591}".repeat(6 - filled)
    }

    /// BPM category
    pub fn bpm_category(&self) -> &'static str {
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

/// Playlist statistics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct PlaylistStats {
    pub smoothness: f64,
    pub energy_balance: f64,
    pub avg_bpm: f64,
    pub avg_energy: f64,
    pub total_duration_sec: f64,
    pub track_count: usize,
}

/// AI DJ - analyze and order tracks intelligently
pub struct AiDj {
    analyses: Mutex<HashMap<String, AudioAnalysis>>,
    shuffle_history: Mutex<Vec<Vec<String>>>,
    current_shuffle_index: Mutex<isize>,
}

/// Musical key compatibility (Camelot wheel)
const KEY_COMPATIBILITY: &[(&str, &[&str])] = &[
    ("C", &["C", "G", "F", "Am", "Em", "Dm"]),
    ("G", &["G", "D", "C", "Em", "Bm", "Am"]),
    ("D", &["D", "A", "G", "Bm", "F#m", "Em"]),
    ("A", &["A", "E", "D", "F#m", "C#m", "Bm"]),
    ("E", &["E", "B", "A", "C#m", "G#m", "F#m"]),
    ("B", &["B", "F#", "E", "G#m", "D#m", "C#m"]),
    ("F#", &["F#", "C#", "B", "D#m", "A#m", "G#m"]),
    ("C#", &["C#", "G#", "F#", "A#m", "E#m", "D#m"]),
    ("F", &["F", "C", "Bb", "Dm", "Am", "Gm"]),
    ("Bb", &["Bb", "F", "Eb", "Gm", "Dm", "Cm"]),
    ("Eb", &["Eb", "Bb", "Ab", "Cm", "Gm", "Fm"]),
    ("Ab", &["Ab", "Eb", "Db", "Fm", "Cm", "Bbm"]),
    ("Am", &["Am", "Em", "Dm", "C", "G", "F"]),
    ("Em", &["Em", "Bm", "Am", "G", "D", "C"]),
    ("Bm", &["Bm", "F#m", "Em", "D", "A", "G"]),
    ("F#m", &["F#m", "C#m", "Bm", "A", "E", "D"]),
    ("C#m", &["C#m", "G#m", "F#m", "E", "B", "A"]),
    ("G#m", &["G#m", "D#m", "C#m", "B", "F#", "E"]),
    ("D#m", &["D#m", "A#m", "G#m", "F#", "C#", "B"]),
    ("Dm", &["Dm", "Am", "Gm", "F", "C", "Bb"]),
    ("Gm", &["Gm", "Dm", "Cm", "Bb", "F", "Eb"]),
    ("Cm", &["Cm", "Gm", "Fm", "Eb", "Bb", "Ab"]),
    ("Fm", &["Fm", "Cm", "Bbm", "Ab", "Eb", "Db"]),
    ("Bbm", &["Bbm", "Fm", "Ebm", "Db", "Ab", "Gb"]),
];

impl AiDj {
    pub fn new() -> Self {
        Self {
            analyses: Mutex::new(HashMap::new()),
            shuffle_history: Mutex::new(Vec::new()),
            current_shuffle_index: Mutex::new(-1),
        }
    }

    /// Analyze a single track
    pub fn analyze_track(&self, file_path: &str) -> Result<AudioAnalysis, Box<dyn std::error::Error + Send + Sync>> {
        // Check cache
        {
            let analyses = self.analyses.lock().unwrap();
            if let Some(analysis) = analyses.get(file_path) {
                return Ok(analysis.clone());
            }
        }

        let mut analysis = AudioAnalysis::new(file_path);
        let filename = analysis.filename.clone();

        // Get duration via ffprobe
        match get_audio_duration(file_path) {
            Ok(duration) => {
                analysis.duration_sec = duration;
            }
            Err(e) => {
                eprintln!("ffprobe error for {}: {}", filename, e);
            }
        }

        // Measure loudness via ffmpeg volumedetect
        match measure_loudness(file_path) {
            Ok(loudness) => {
                analysis.loudness_db = loudness;
                let normalized = (loudness + 30.0) / 25.0;
                analysis.energy = normalized.clamp(0.0, 1.0);
            }
            Err(e) => {
                eprintln!("Loudness measure error for {}: {}", filename, e);
            }
        }

        // FFT analysis for BPM and Key
        match analyze_audio_features(file_path) {
            Ok((bpm, key)) => {
                analysis.bpm = bpm;
                analysis.key = key;
            }
            Err(e) => {
                eprintln!("Audio analysis error for {}: {}", filename, e);
                analysis.bpm = 100.0 + analysis.energy * 40.0;
                let keys = ["C", "G", "D", "A", "Am", "Em", "F", "Dm"];
                let key_index = filename.bytes().fold(0u32, |acc, b| acc.wrapping_add(b as u32)) as usize % keys.len();
                analysis.key = keys[key_index].to_string();
            }
        }

        // Compute intro score
        let energy_score = 100.0 - (analysis.energy - 0.6).abs() * 100.0;
        let bpm_score = 100.0 - (analysis.bpm - 100.0).abs() * 0.5;
        let loudness_score = 100.0 - (analysis.loudness_db + 14.0).abs() * 2.0;

        analysis.intro_score = (energy_score * 0.4 + bpm_score * 0.3 + loudness_score * 0.3)
            .clamp(0.0, 100.0);

        // Save to cache
        {
            let mut analyses = self.analyses.lock().unwrap();
            analyses.insert(file_path.to_string(), analysis.clone());
        }

        Ok(analysis)
    }

    /// Suggest playlist order with strategy
    pub fn suggest_order(&self, file_paths: &[String], strategy: &str) -> Vec<String> {
        if file_paths.len() <= 1 {
            return file_paths.to_vec();
        }

        let analyses: Vec<AudioAnalysis> = file_paths
            .iter()
            .filter_map(|p| self.analyze_track(p).ok())
            .collect();

        match strategy {
            "smooth" => self.order_smooth(&analyses),
            "energy_up" => self.order_energy_up(&analyses),
            "energy_down" => self.order_energy_down(&analyses),
            "random_smart" => self.order_random_smart(&analyses),
            _ => file_paths.to_vec(),
        }
    }

    fn order_smooth(&self, analyses: &[AudioAnalysis]) -> Vec<String> {
        if analyses.is_empty() {
            return Vec::new();
        }

        let mut sorted_by_intro: Vec<_> = analyses.to_vec();
        sorted_by_intro.sort_by(|a, b| b.intro_score.partial_cmp(&a.intro_score).unwrap());

        let mut ordered = vec![sorted_by_intro[0].clone()];
        let mut remaining: Vec<_> = sorted_by_intro.into_iter().skip(1).collect();

        while !remaining.is_empty() {
            let current = ordered.last().unwrap();
            let mut best_score = -1.0;
            let mut best_idx = 0;

            for (i, candidate) in remaining.iter().enumerate() {
                let score = transition_score(current, candidate);
                if score > best_score {
                    best_score = score;
                    best_idx = i;
                }
            }

            ordered.push(remaining.remove(best_idx));
        }

        ordered.into_iter().map(|a| a.file_path).collect()
    }

    fn order_energy_up(&self, analyses: &[AudioAnalysis]) -> Vec<String> {
        let mut sorted: Vec<_> = analyses.to_vec();
        sorted.sort_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap());
        sorted.into_iter().map(|a| a.file_path).collect()
    }

    fn order_energy_down(&self, analyses: &[AudioAnalysis]) -> Vec<String> {
        let mut sorted: Vec<_> = analyses.to_vec();
        sorted.sort_by(|a, b| b.energy.partial_cmp(&a.energy).unwrap());
        sorted.into_iter().map(|a| a.file_path).collect()
    }

    fn order_random_smart(&self, analyses: &[AudioAnalysis]) -> Vec<String> {
        if analyses.is_empty() {
            return Vec::new();
        }

        let mut shuffled: Vec<_> = analyses.to_vec();
        let mut rng = thread_rng();
        shuffled.shuffle(&mut rng);

        let opener_idx = shuffled[..3.min(shuffled.len())]
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.intro_score.partial_cmp(&b.intro_score).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        let mut ordered = vec![shuffled.remove(opener_idx)];

        while !shuffled.is_empty() {
            let current = ordered.last().unwrap();

            let compatible: Vec<_> = shuffled
                .iter()
                .enumerate()
                .filter(|(_, a)| keys_compatible(&current.key, &a.key))
                .map(|(i, _)| i)
                .collect();

            let next_idx = if !compatible.is_empty() {
                *compatible.choose(&mut rng).unwrap()
            } else {
                rng.gen_range(0..shuffled.len())
            };

            ordered.push(shuffled.remove(next_idx));
        }

        ordered.into_iter().map(|a| a.file_path).collect()
    }

    /// Get best opener tracks
    pub fn get_best_opener(&self, file_paths: &[String], top_n: usize) -> Vec<(String, f64)> {
        let mut analyses: Vec<_> = file_paths
            .iter()
            .filter_map(|p| self.analyze_track(p).ok())
            .collect();

        analyses.sort_by(|a, b| b.intro_score.partial_cmp(&a.intro_score).unwrap());

        analyses.into_iter()
            .take(top_n)
            .map(|a| (a.file_path, a.intro_score))
            .collect()
    }

    /// Shuffle again (unique from recent history)
    pub fn shuffle_again(&self, file_paths: &[String]) -> Vec<String> {
        let max_attempts = 10;

        for _ in 0..max_attempts {
            let analyses: Vec<_> = file_paths
                .iter()
                .filter_map(|p| self.analyze_track(p).ok())
                .collect();

            let new_order = self.order_random_smart(&analyses);

            let history = self.shuffle_history.lock().unwrap();
            let is_unique = !history.iter().rev().take(5).any(|h| h == &new_order);
            drop(history);

            if is_unique {
                let mut history = self.shuffle_history.lock().unwrap();
                history.push(new_order.clone());
                *self.current_shuffle_index.lock().unwrap() = history.len() as isize - 1;
                return new_order;
            }
        }

        let analyses: Vec<_> = file_paths
            .iter()
            .filter_map(|p| self.analyze_track(p).ok())
            .collect();
        self.order_random_smart(&analyses)
    }

    /// Get playlist statistics
    pub fn get_playlist_stats(&self, file_paths: &[String]) -> PlaylistStats {
        let analyses: Vec<_> = file_paths
            .iter()
            .filter_map(|p| self.analyze_track(p).ok())
            .collect();

        if analyses.is_empty() {
            return PlaylistStats::default();
        }

        let mut total_transition_score = 0.0;
        for i in 0..analyses.len().saturating_sub(1) {
            total_transition_score += transition_score(&analyses[i], &analyses[i + 1]);
        }

        let max_possible = (analyses.len().saturating_sub(1)) as f64 * 100.0;
        let smoothness = if max_possible > 0.0 {
            (total_transition_score / max_possible * 100.0).round()
        } else {
            0.0
        };

        let energies: Vec<_> = analyses.iter().map(|a| a.energy).collect();
        let avg_energy = energies.iter().sum::<f64>() / energies.len() as f64;
        let variance = energies.iter()
            .map(|e| (e - avg_energy).powi(2))
            .sum::<f64>() / energies.len() as f64;
        let energy_balance = (100.0 - variance * 200.0).max(0.0).min(100.0).round();

        let avg_bpm = analyses.iter().map(|a| a.bpm).sum::<f64>() / analyses.len() as f64;
        let total_duration: f64 = analyses.iter().map(|a| a.duration_sec).sum();

        PlaylistStats {
            smoothness,
            energy_balance,
            avg_bpm: (avg_bpm * 10.0).round() / 10.0,
            avg_energy: (avg_energy * 100.0).round() / 100.0,
            total_duration_sec: total_duration,
            track_count: analyses.len(),
        }
    }
}

impl Default for AiDj {
    fn default() -> Self {
        Self::new()
    }
}

// ==================== Helper Functions ====================

/// Check if two keys are compatible (Camelot wheel)
pub fn keys_compatible(key1: &str, key2: &str) -> bool {
    if key1 == key2 {
        return true;
    }
    let compatibility_map: HashMap<_, _> = KEY_COMPATIBILITY.iter().copied().collect();
    if let Some(compatible) = compatibility_map.get(key1) {
        compatible.contains(&key2)
    } else {
        false
    }
}

/// Compute transition score between two tracks
pub fn transition_score(from_track: &AudioAnalysis, to_track: &AudioAnalysis) -> f64 {
    let mut score = 0.0;

    // Key compatibility (0-40 points)
    if keys_compatible(&from_track.key, &to_track.key) {
        score += 40.0;
    } else if from_track.key == to_track.key {
        score += 35.0;
    }

    // Energy flow (0-30 points)
    let energy_diff = (from_track.energy - to_track.energy).abs();
    score += 30.0 * (1.0 - energy_diff);

    // BPM compatibility (0-20 points)
    let bpm_diff = (from_track.bpm - to_track.bpm).abs();
    if bpm_diff < 5.0 {
        score += 20.0;
    } else if bpm_diff < 10.0 {
        score += 15.0;
    } else if bpm_diff < 20.0 {
        score += 10.0;
    }

    // Loudness compatibility (0-10 points)
    let loudness_diff = (from_track.loudness_db - to_track.loudness_db).abs();
    score += (10.0 - loudness_diff).max(0.0);

    score
}

/// Get audio duration via ffprobe
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

/// Measure loudness via ffmpeg volumedetect
fn measure_loudness(file_path: &str) -> Result<f64, Box<dyn std::error::Error + Send + Sync>> {
    let output = Command::new("ffmpeg")
        .args(["-i", file_path, "-af", "volumedetect", "-f", "null", "-"])
        .output()?;

    let stderr = String::from_utf8_lossy(&output.stderr);
    let re = Regex::new(r"mean_volume:\s*([-\d.]+)\s*dB")?;

    if let Some(caps) = re.captures(&stderr) {
        if let Some(val) = caps.get(1) {
            return Ok(val.as_str().parse::<f64>()?);
        }
    }

    Err("Could not parse loudness".into())
}

/// Analyze audio features (BPM, Key) using FFT
fn analyze_audio_features(file_path: &str) -> Result<(f64, String), Box<dyn std::error::Error + Send + Sync>> {
    let temp_dir = tempfile::tempdir()?;
    let temp_file = temp_dir.path().join("temp_audio.raw");

    let output = Command::new("ffmpeg")
        .args([
            "-i", file_path,
            "-ar", "22050",
            "-ac", "1",
            "-t", "60",
            "-f", "s16le",
            temp_file.to_str().unwrap()
        ])
        .output()?;

    if !output.status.success() {
        return Err("ffmpeg conversion failed".into());
    }

    let audio_data = std::fs::read(&temp_file)?;
    let samples: Vec<f64> = audio_data
        .chunks_exact(2)
        .map(|chunk| {
            i16::from_le_bytes([chunk[0], chunk[1]]) as f64 / 32768.0
        })
        .collect();

    let bpm = detect_bpm(&samples, 22050);
    let key = detect_key(&samples, 22050);

    Ok((bpm, key))
}

/// BPM detection using onset detection on energy envelope
fn detect_bpm(samples: &[f64], sample_rate: u32) -> f64 {
    let hop_size = 512;
    let frame_size = 1024;
    let num_frames = samples.len() / hop_size;

    let mut energy_envelope = Vec::with_capacity(num_frames);

    for i in 0..num_frames {
        let start = i * hop_size;
        let end = (start + frame_size).min(samples.len());
        let frame = &samples[start..end];
        let energy: f64 = frame.iter().map(|&s| s * s).sum();
        energy_envelope.push(energy.sqrt());
    }

    let mut peaks = Vec::new();
    for i in 1..energy_envelope.len().saturating_sub(1) {
        if energy_envelope[i] > energy_envelope[i - 1]
            && energy_envelope[i] > energy_envelope[i + 1]
            && energy_envelope[i] > 0.1
        {
            peaks.push(i);
        }
    }

    if peaks.len() >= 2 {
        let mut intervals: Vec<_> = (1..peaks.len())
            .map(|i| peaks[i] - peaks[i - 1])
            .collect();
        intervals.sort();
        let median_interval = intervals[intervals.len() / 2] as f64;
        let bpm = (60.0 * sample_rate as f64) / (median_interval * hop_size as f64);
        bpm.clamp(60.0, 200.0)
    } else {
        120.0
    }
}

/// Key detection using chroma features and Krumhansl-Schmuckler algorithm
fn detect_key(samples: &[f64], sample_rate: u32) -> String {
    let frame_size = 4096;
    let hop_size = 2048;
    let num_frames = samples.len() / hop_size;

    let mut chroma = Array2::zeros((12, num_frames));
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(frame_size);

    for frame_idx in 0..num_frames.saturating_sub(1) {
        let start = frame_idx * hop_size;
        let mut buffer: Vec<Complex<f64>> = samples[start..(start + frame_size).min(samples.len())]
            .iter()
            .map(|&s| Complex::new(s, 0.0))
            .collect();

        while buffer.len() < frame_size {
            buffer.push(Complex::new(0.0, 0.0));
        }

        fft.process(&mut buffer);

        for bin in 1..frame_size / 2 {
            let freq = bin as f64 * sample_rate as f64 / frame_size as f64;
            let magnitude = buffer[bin].norm();
            let pitch_class = freq_to_pitch_class(freq);
            chroma[[pitch_class, frame_idx]] += magnitude;
        }
    }

    let chroma_mean: Array1<f64> = chroma.mean_axis(ndarray::Axis(1)).unwrap();

    let key_index = chroma_mean.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    let key_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

    let major_profile = Array1::from(vec![6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88_f64]);
    let minor_profile = Array1::from(vec![6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17_f64]);

    let major_corr = correlation(&chroma_mean, &rotate_array(&major_profile, key_index));
    let minor_corr = correlation(&chroma_mean, &rotate_array(&minor_profile, key_index));

    if minor_corr > major_corr {
        format!("{}m", key_names[key_index])
    } else {
        key_names[key_index].to_string()
    }
}

fn freq_to_pitch_class(freq: f64) -> usize {
    if freq <= 0.0 {
        return 0;
    }
    let midi_note = 69.0 + 12.0 * (freq / 440.0).log2();
    ((midi_note.round() as i32) % 12).rem_euclid(12) as usize
}

fn rotate_array(arr: &Array1<f64>, shift: usize) -> Array1<f64> {
    let n = arr.len();
    let mut result = Array1::zeros(n);
    for i in 0..n {
        result[i] = arr[(i + shift) % n];
    }
    result
}

fn correlation(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    let mean_a = a.mean().unwrap_or(0.0);
    let mean_b = b.mean().unwrap_or(0.0);

    let num: f64 = a.iter().zip(b.iter())
        .map(|(x, y)| (x - mean_a) * (y - mean_b))
        .sum();

    let den_a: f64 = a.iter().map(|x| (x - mean_a).powi(2)).sum();
    let den_b: f64 = b.iter().map(|y| (y - mean_b).powi(2)).sum();

    if den_a > 0.0 && den_b > 0.0 {
        num / (den_a.sqrt() * den_b.sqrt())
    } else {
        0.0
    }
}

// ==================== YouTube Generator ====================

/// YouTube metadata generator
pub struct YouTubeGenerator {
    channel_name: String,
    max_tags_length: usize,
}

/// Theme data
#[derive(Clone, Copy)]
struct ThemeData {
    emoji: &'static str,
    thai_name: &'static str,
    english_name: &'static str,
    keywords_th: &'static [&'static str],
    keywords_en: &'static [&'static str],
}

/// Track input for timestamp generation
#[derive(Debug, Clone)]
pub struct TrackInput {
    pub name: String,
    pub duration_sec: u64,
}

/// Track info with timestamp
#[derive(Debug, Clone)]
pub struct TrackInfo {
    pub name: String,
    pub timestamp: String,
    pub duration_sec: u64,
}

const THEMES: &[(&str, ThemeData)] = &[
    ("cafe", ThemeData {
        emoji: "\u{2615}",
        thai_name: "\u{0e40}\u{0e1e}\u{0e25}\u{0e07}\u{0e04}\u{0e32}\u{0e40}\u{0e1f}\u{0e48}",
        english_name: "Cafe Music",
        keywords_th: &["\u{0e40}\u{0e1e}\u{0e25}\u{0e07}\u{0e40}\u{0e1b}\u{0e34}\u{0e14}\u{0e23}\u{0e49}\u{0e32}\u{0e19}\u{0e01}\u{0e32}\u{0e41}\u{0e1f}", "\u{0e40}\u{0e1e}\u{0e25}\u{0e07}\u{0e04}\u{0e32}\u{0e40}\u{0e1f}\u{0e48}", "\u{0e1f}\u{0e31}\u{0e07}\u{0e2a}\u{0e1a}\u{0e32}\u{0e22}"],
        keywords_en: &["cafe music", "coffee shop music", "work music", "study music"],
    }),
    ("driving", ThemeData {
        emoji: "\u{1f697}",
        thai_name: "\u{0e40}\u{0e1e}\u{0e25}\u{0e07}\u{0e02}\u{0e31}\u{0e1a}\u{0e23}\u{0e16}",
        english_name: "Driving Music",
        keywords_th: &["\u{0e40}\u{0e1e}\u{0e25}\u{0e07}\u{0e02}\u{0e31}\u{0e1a}\u{0e23}\u{0e16}", "\u{0e40}\u{0e1e}\u{0e25}\u{0e07}\u{0e40}\u{0e14}\u{0e34}\u{0e19}\u{0e17}\u{0e32}\u{0e07}"],
        keywords_en: &["driving music", "road trip", "travel music"],
    }),
    ("sleep", ThemeData {
        emoji: "\u{1f319}",
        thai_name: "\u{0e40}\u{0e1e}\u{0e25}\u{0e07}\u{0e01}\u{0e48}\u{0e2d}\u{0e19}\u{0e19}\u{0e2d}\u{0e19}",
        english_name: "Sleep Music",
        keywords_th: &["\u{0e40}\u{0e1e}\u{0e25}\u{0e07}\u{0e01}\u{0e48}\u{0e2d}\u{0e19}\u{0e19}\u{0e2d}\u{0e19}", "\u{0e40}\u{0e1e}\u{0e25}\u{0e07}\u{0e1c}\u{0e48}\u{0e2d}\u{0e19}\u{0e04}\u{0e25}\u{0e32}\u{0e22}"],
        keywords_en: &["sleep music", "relaxing music", "calm music"],
    }),
    ("workout", ThemeData {
        emoji: "\u{1f4aa}",
        thai_name: "\u{0e40}\u{0e1e}\u{0e25}\u{0e07}\u{0e2d}\u{0e2d}\u{0e01}\u{0e01}\u{0e33}\u{0e25}\u{0e31}\u{0e07}\u{0e01}\u{0e32}\u{0e22}",
        english_name: "Workout Music",
        keywords_th: &["\u{0e40}\u{0e1e}\u{0e25}\u{0e07}\u{0e2d}\u{0e2d}\u{0e01}\u{0e01}\u{0e33}\u{0e25}\u{0e31}\u{0e07}\u{0e01}\u{0e32}\u{0e22}"],
        keywords_en: &["workout music", "gym music", "fitness music"],
    }),
    ("focus", ThemeData {
        emoji: "\u{1f3af}",
        thai_name: "\u{0e40}\u{0e1e}\u{0e25}\u{0e07}\u{0e42}\u{0e1f}\u{0e01}\u{0e31}\u{0e2a}",
        english_name: "Focus Music",
        keywords_th: &["\u{0e40}\u{0e1e}\u{0e25}\u{0e07}\u{0e17}\u{0e33}\u{0e07}\u{0e32}\u{0e19}", "\u{0e40}\u{0e1e}\u{0e25}\u{0e07}\u{0e42}\u{0e1f}\u{0e01}\u{0e31}\u{0e2a}"],
        keywords_en: &["focus music", "concentration", "study music"],
    }),
    ("chill", ThemeData {
        emoji: "\u{1f334}",
        thai_name: "\u{0e40}\u{0e1e}\u{0e25}\u{0e07}\u{0e0a}\u{0e34}\u{0e25}\u{0e46}",
        english_name: "Chill Vibes",
        keywords_th: &["\u{0e40}\u{0e1e}\u{0e25}\u{0e07}\u{0e0a}\u{0e34}\u{0e25}\u{0e46}", "\u{0e1f}\u{0e31}\u{0e07}\u{0e2a}\u{0e1a}\u{0e32}\u{0e22}"],
        keywords_en: &["chill vibes", "lofi", "relaxing", "easy listening"],
    }),
];

impl YouTubeGenerator {
    pub fn new(channel_name: &str) -> Self {
        Self {
            channel_name: channel_name.to_string(),
            max_tags_length: 500,
        }
    }

    pub fn generate_title(&self, volume: i32, theme: &str, duration_str: &str) -> String {
        let theme_data = get_theme(theme);
        format!("Vol.{} {} {} \u{0e1f}\u{0e31}\u{0e07}\u{0e2a}\u{0e1a}\u{0e32}\u{0e22}\u{0e44}\u{0e21}\u{0e48}\u{0e21}\u{0e35}\u{0e2a}\u{0e30}\u{0e14}\u{0e38}\u{0e14} {}", volume, theme_data.emoji, theme_data.thai_name, duration_str)
    }

    pub fn generate_timestamps(&self, tracks: &[TrackInput]) -> Vec<TrackInfo> {
        let mut timestamped = Vec::new();
        let mut current_time_sec: u64 = 0;

        for track in tracks {
            let minutes = current_time_sec / 60;
            let seconds = current_time_sec % 60;
            let timestamp = format!("{}:{:02}", minutes, seconds);

            timestamped.push(TrackInfo {
                name: track.name.clone(),
                timestamp,
                duration_sec: track.duration_sec,
            });

            current_time_sec += track.duration_sec;
        }

        timestamped
    }

    pub fn format_duration(&self, total_seconds: f64) -> String {
        let hours = (total_seconds / 3600.0) as u64;
        let minutes = ((total_seconds % 3600.0) / 60.0) as u64;

        if hours > 0 {
            if minutes == 0 {
                format!("{} \u{0e0a}\u{0e31}\u{0e48}\u{0e27}\u{0e42}\u{0e21}\u{0e07}", hours)
            } else {
                format!("{} \u{0e0a}\u{0e31}\u{0e48}\u{0e27}\u{0e42}\u{0e21}\u{0e07} {} \u{0e19}\u{0e32}\u{0e17}\u{0e35}", hours, minutes)
            }
        } else {
            format!("{} \u{0e19}\u{0e32}\u{0e17}\u{0e35}", minutes)
        }
    }
}

impl Default for YouTubeGenerator {
    fn default() -> Self {
        Self::new("Chillin' Vibes")
    }
}

fn get_theme(theme: &str) -> &'static ThemeData {
    for &(name, ref data) in THEMES {
        if name == theme {
            return data;
        }
    }
    &THEMES[5].1
}

// ==================== Tests ====================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_analysis_new() {
        let analysis = AudioAnalysis::new("/path/to/song.mp3");
        assert_eq!(analysis.filename, "song.mp3");
        assert_eq!(analysis.loudness_db, -14.0);
    }

    #[test]
    fn test_energy_bars() {
        let mut analysis = AudioAnalysis::new("test.mp3");
        analysis.energy = 0.5;
        let bars = analysis.energy_bars();
        assert_eq!(bars.chars().count(), 6);
    }

    #[test]
    fn test_bpm_category() {
        let mut analysis = AudioAnalysis::new("test.mp3");
        analysis.bpm = 70.0;
        assert_eq!(analysis.bpm_category(), "slow");
        analysis.bpm = 100.0;
        assert_eq!(analysis.bpm_category(), "medium");
        analysis.bpm = 120.0;
        assert_eq!(analysis.bpm_category(), "upbeat");
        analysis.bpm = 150.0;
        assert_eq!(analysis.bpm_category(), "fast");
    }

    #[test]
    fn test_keys_compatible() {
        assert!(keys_compatible("C", "C"));
        assert!(keys_compatible("C", "G"));
        assert!(!keys_compatible("C", "F#"));
    }

    #[test]
    fn test_transition_score() {
        let a = AudioAnalysis {
            file_path: "a.mp3".into(),
            filename: "a.mp3".into(),
            duration_sec: 180.0,
            bpm: 120.0,
            key: "C".into(),
            energy: 0.5,
            loudness_db: -14.0,
            intro_score: 50.0,
        };
        let b = AudioAnalysis {
            file_path: "b.mp3".into(),
            filename: "b.mp3".into(),
            duration_sec: 200.0,
            bpm: 122.0,
            key: "G".into(),
            energy: 0.6,
            loudness_db: -13.0,
            intro_score: 60.0,
        };
        let score = transition_score(&a, &b);
        assert!(score > 50.0); // Compatible keys + close BPM + close energy
    }

    #[test]
    fn test_youtube_title() {
        let yt = YouTubeGenerator::new("Test Channel");
        let title = yt.generate_title(24, "cafe", "1 hr");
        assert!(title.contains("Vol.24"));
    }

    #[test]
    fn test_format_duration() {
        let yt = YouTubeGenerator::default();
        let d1 = yt.format_duration(3600.0);
        assert!(d1.contains("1"));
        let d2 = yt.format_duration(1800.0);
        assert!(d2.contains("30"));
    }
}
