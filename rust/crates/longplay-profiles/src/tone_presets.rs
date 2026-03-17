#[derive(Debug, Clone)]
pub struct TonePreset {
    pub name: String,
    pub low_gain: f64,
    pub mid_gain: f64,
    pub high_gain: f64,
    pub low_freq: f64,
    pub high_freq: f64,
}

/// Returns all tone presets with exact values from the C++ engine.
/// Presets: Neutral (flat), Warm, Bright, Dark, Presence (Mid-Forward).
pub fn get_tone_presets() -> Vec<TonePreset> {
    vec![
        TonePreset {
            name: "flat".into(),
            low_gain: 0.0,
            mid_gain: 0.0,
            high_gain: 0.0,
            low_freq: 200.0,
            high_freq: 4000.0,
        },
        TonePreset {
            name: "warm".into(),
            low_gain: 1.5,
            mid_gain: -0.5,
            high_gain: -1.0,
            low_freq: 150.0,
            high_freq: 5000.0,
        },
        TonePreset {
            name: "bright".into(),
            low_gain: -0.5,
            mid_gain: 0.0,
            high_gain: 1.5,
            low_freq: 200.0,
            high_freq: 3500.0,
        },
        TonePreset {
            name: "dark".into(),
            low_gain: 1.0,
            mid_gain: 0.0,
            high_gain: -2.0,
            low_freq: 250.0,
            high_freq: 4000.0,
        },
        TonePreset {
            name: "presence".into(),
            low_gain: 0.0,
            mid_gain: 1.0,
            high_gain: 0.5,
            low_freq: 200.0,
            high_freq: 3000.0,
        },
    ]
}
