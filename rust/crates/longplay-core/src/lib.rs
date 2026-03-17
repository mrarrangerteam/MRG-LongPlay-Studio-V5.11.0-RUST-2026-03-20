pub mod conversions;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub const PI: f64 = std::f64::consts::PI;
pub const TWO_PI: f64 = 2.0 * PI;
pub const EPSILON: f64 = 1e-10;

/// Audio buffer: Vec of channels, each channel is Vec of samples
pub type AudioBuffer = Vec<Vec<f32>>;

/// Audio file metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioInfo {
    pub sample_rate: i32,
    pub channels: i32,
    pub frames: i64,
    pub bit_depth: i32,
    pub format: String,
    pub duration_seconds: f64,
}

impl Default for AudioInfo {
    fn default() -> Self {
        Self {
            sample_rate: 44100,
            channels: 2,
            frames: 0,
            bit_depth: 16,
            format: String::new(),
            duration_seconds: 0.0,
        }
    }
}

/// Settings value that can hold different types (replaces C++ std::variant)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SettingsValue {
    Int(i32),
    Float(f64),
    Float32(f32),
    Bool(bool),
    String(String),
}

impl SettingsValue {
    pub fn as_int(&self) -> Option<i32> {
        match self {
            SettingsValue::Int(v) => Some(*v),
            SettingsValue::Float(v) => Some(*v as i32),
            SettingsValue::Float32(v) => Some(*v as i32),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            SettingsValue::Float(v) => Some(*v),
            SettingsValue::Float32(v) => Some(*v as f64),
            SettingsValue::Int(v) => Some(*v as f64),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            SettingsValue::Float32(v) => Some(*v),
            SettingsValue::Float(v) => Some(*v as f32),
            SettingsValue::Int(v) => Some(*v as f32),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            SettingsValue::Bool(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            SettingsValue::String(v) => Some(v),
            _ => None,
        }
    }
}

impl From<i32> for SettingsValue {
    fn from(v: i32) -> Self { SettingsValue::Int(v) }
}
impl From<f64> for SettingsValue {
    fn from(v: f64) -> Self { SettingsValue::Float(v) }
}
impl From<f32> for SettingsValue {
    fn from(v: f32) -> Self { SettingsValue::Float32(v) }
}
impl From<bool> for SettingsValue {
    fn from(v: bool) -> Self { SettingsValue::Bool(v) }
}
impl From<String> for SettingsValue {
    fn from(v: String) -> Self { SettingsValue::String(v) }
}
impl From<&str> for SettingsValue {
    fn from(v: &str) -> Self { SettingsValue::String(v.to_string()) }
}

/// Settings map type
pub type Settings = HashMap<String, SettingsValue>;

/// Get typed value from Settings with default
pub fn get_setting_f64(s: &Settings, key: &str, default: f64) -> f64 {
    s.get(key).and_then(|v| v.as_f64()).unwrap_or(default)
}

pub fn get_setting_f32(s: &Settings, key: &str, default: f32) -> f32 {
    s.get(key).and_then(|v| v.as_f32()).unwrap_or(default)
}

pub fn get_setting_int(s: &Settings, key: &str, default: i32) -> i32 {
    s.get(key).and_then(|v| v.as_int()).unwrap_or(default)
}

pub fn get_setting_bool(s: &Settings, key: &str, default: bool) -> bool {
    s.get(key).and_then(|v| v.as_bool()).unwrap_or(default)
}

pub fn get_setting_string(s: &Settings, key: &str, default: &str) -> String {
    s.get(key)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| default.to_string())
}

/// Deinterleave audio data from interleaved format to channel-separated
pub fn deinterleave(interleaved: &[f32], channels: usize) -> AudioBuffer {
    let frames = interleaved.len() / channels;
    let mut result = vec![vec![0.0f32; frames]; channels];
    for i in 0..frames {
        for ch in 0..channels {
            result[ch][i] = interleaved[i * channels + ch];
        }
    }
    result
}

/// Interleave audio data from channel-separated to interleaved format
pub fn interleave(buffer: &AudioBuffer) -> Vec<f32> {
    if buffer.is_empty() {
        return Vec::new();
    }
    let channels = buffer.len();
    let frames = buffer[0].len();
    let mut result = vec![0.0f32; frames * channels];
    for i in 0..frames {
        for ch in 0..channels {
            result[i * channels + ch] = buffer[ch][i];
        }
    }
    result
}

/// Common error type for the library
#[derive(Debug, thiserror::Error)]
pub enum LongplayError {
    #[error("Audio I/O error: {0}")]
    AudioIo(String),
    #[error("Processing error: {0}")]
    Processing(String),
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    #[error("FFmpeg error: {0}")]
    Ffmpeg(String),
    #[error("Analysis error: {0}")]
    Analysis(String),
}

pub type Result<T> = std::result::Result<T, LongplayError>;
