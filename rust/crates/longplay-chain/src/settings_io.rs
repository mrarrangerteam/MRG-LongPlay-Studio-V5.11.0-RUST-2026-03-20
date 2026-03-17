//! Settings I/O - JSON serialization for chain settings
//! Replaces C++ nlohmann/json with serde_json

use longplay_core::{Settings, SettingsValue};
use serde_json::{json, Value, Map};
use std::fs;

/// Loaded chain settings from JSON file
pub struct LoadedChainSettings {
    pub intensity: f32,
    pub target_lufs: f32,
    pub target_tp: f32,
    pub equalizer: Option<Settings>,
    pub dynamics: Option<Settings>,
    pub imager: Option<Settings>,
    pub maximizer: Option<Settings>,
}

/// Convert a Settings map to serde_json::Value
fn settings_to_json(s: &Settings) -> Value {
    let mut map = Map::new();
    for (key, val) in s {
        let json_val = match val {
            SettingsValue::Int(v) => json!(*v),
            SettingsValue::Float(v) => json!(*v),
            SettingsValue::Float32(v) => json!(*v),
            SettingsValue::Bool(v) => json!(*v),
            SettingsValue::String(v) => json!(v),
        };
        map.insert(key.clone(), json_val);
    }
    Value::Object(map)
}

/// Convert serde_json::Value to a Settings map
fn json_to_settings(j: &Value) -> Settings {
    let mut s = Settings::new();
    if let Value::Object(map) = j {
        for (key, val) in map {
            let sv = match val {
                Value::Bool(v) => SettingsValue::Bool(*v),
                Value::Number(n) => {
                    if let Some(i) = n.as_i64() {
                        SettingsValue::Int(i as i32)
                    } else if let Some(f) = n.as_f64() {
                        SettingsValue::Float(f)
                    } else {
                        continue;
                    }
                }
                Value::String(v) => SettingsValue::String(v.clone()),
                _ => continue,
            };
            s.insert(key.clone(), sv);
        }
    }
    s
}

/// Save chain settings to a JSON file
pub fn save_chain_settings(
    filepath: &str,
    intensity: f32,
    target_lufs: f32,
    target_tp: f32,
    equalizer: &Settings,
    dynamics: &Settings,
    imager: &Settings,
    maximizer: &Settings,
) -> bool {
    let settings = json!({
        "intensity": intensity,
        "target_lufs": target_lufs,
        "target_tp": target_tp,
        "equalizer": settings_to_json(equalizer),
        "dynamics": settings_to_json(dynamics),
        "imager": settings_to_json(imager),
        "maximizer": settings_to_json(maximizer),
    });

    match serde_json::to_string_pretty(&settings) {
        Ok(json_str) => {
            match fs::write(filepath, json_str) {
                Ok(()) => true,
                Err(e) => {
                    eprintln!("Failed to write settings file: {}", e);
                    false
                }
            }
        }
        Err(e) => {
            eprintln!("Failed to serialize settings: {}", e);
            false
        }
    }
}

/// Load chain settings from a JSON file
pub fn load_chain_settings(filepath: &str) -> Option<LoadedChainSettings> {
    let content = match fs::read_to_string(filepath) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to read settings file: {}", e);
            return None;
        }
    };

    let parsed: Value = match serde_json::from_str(&content) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Failed to parse settings JSON: {}", e);
            return None;
        }
    };

    let intensity = parsed.get("intensity")
        .and_then(|v| v.as_f64())
        .unwrap_or(75.0) as f32;

    let target_lufs = parsed.get("target_lufs")
        .and_then(|v| v.as_f64())
        .unwrap_or(-14.0) as f32;

    let target_tp = parsed.get("target_tp")
        .and_then(|v| v.as_f64())
        .unwrap_or(-1.0) as f32;

    let equalizer = parsed.get("equalizer")
        .filter(|v| v.is_object())
        .map(json_to_settings);

    let dynamics = parsed.get("dynamics")
        .filter(|v| v.is_object())
        .map(json_to_settings);

    let imager = parsed.get("imager")
        .filter(|v| v.is_object())
        .map(json_to_settings);

    let maximizer = parsed.get("maximizer")
        .filter(|v| v.is_object())
        .map(json_to_settings);

    Some(LoadedChainSettings {
        intensity,
        target_lufs,
        target_tp,
        equalizer,
        dynamics,
        imager,
        maximizer,
    })
}
