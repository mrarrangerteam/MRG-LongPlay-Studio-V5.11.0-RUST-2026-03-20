use longplay_core::{Settings, SettingsValue};

pub fn generate_maximizer_recommendations(
    target_lufs: f32, current_lufs: f32, target_tp: f32, irc_mode: &str, intensity: f32,
) -> Settings {
    let mut settings = Settings::new();
    let lufs_diff = target_lufs - current_lufs;
    let mut gain_db = 0.0f32;
    if lufs_diff > 0.0 {
        gain_db = (lufs_diff * 0.8).clamp(0.0, 20.0);
    }
    gain_db *= intensity / 100.0;
    settings.insert("gain_db".into(), SettingsValue::Float32(gain_db));
    settings.insert("ceiling_dbfs".into(), SettingsValue::Float32(target_tp));
    settings.insert("irc_mode".into(), SettingsValue::String(irc_mode.to_string()));
    settings.insert("enabled".into(), SettingsValue::Bool(true));
    settings
}

pub fn generate_equalizer_recommendations(brightness: f32, low_energy: f32, intensity: f32) -> Settings {
    let mut settings = Settings::new();
    let intensity_factor = intensity / 100.0;
    if brightness > 0.4 {
        settings.insert("band_8k_gain_db".into(), SettingsValue::Float32(-1.5 * intensity_factor));
        settings.insert("band_8k_q".into(), SettingsValue::Float32(0.7));
    }
    if brightness < 0.15 {
        settings.insert("band_8k_gain_db".into(), SettingsValue::Float32(1.5 * intensity_factor));
        settings.insert("band_8k_q".into(), SettingsValue::Float32(0.7));
    }
    if low_energy > 0.35 {
        settings.insert("band_200hz_gain_db".into(), SettingsValue::Float32(-2.0 * intensity_factor));
        settings.insert("band_200hz_q".into(), SettingsValue::Float32(0.7));
    }
    settings
}

pub fn generate_dynamics_recommendations(crest_factor: f32, intensity: f32) -> Settings {
    let mut settings = Settings::new();
    let intensity_factor = intensity / 100.0;
    let (ratio, description) = if crest_factor < 6.0 {
        (1.5f32, "Gentle glue")
    } else if crest_factor > 15.0 {
        (3.0f32, "Tame dynamics")
    } else {
        (2.0f32, "Balanced compression")
    };
    let threshold = ((-20.0 + (crest_factor - 10.0) * 0.5) + (5.0 * intensity_factor)).clamp(-40.0, -5.0);
    settings.insert("ratio".into(), SettingsValue::Float32(ratio));
    settings.insert("threshold_db".into(), SettingsValue::Float32(threshold));
    settings.insert("attack_ms".into(), SettingsValue::Float32(10.0));
    settings.insert("release_ms".into(), SettingsValue::Float32(100.0));
    settings.insert("makeup_gain".into(), SettingsValue::String("auto".into()));
    settings.insert("description".into(), SettingsValue::String(description.into()));
    settings.insert("enabled".into(), SettingsValue::Bool(true));
    settings
}

pub fn generate_imager_recommendations(correlation: f32, is_mono: bool, width: f32) -> Settings {
    let mut settings = Settings::new();
    if is_mono {
        settings.insert("width".into(), SettingsValue::Float32(100.0));
        settings.insert("enabled".into(), SettingsValue::Bool(false));
        return settings;
    }
    if correlation > 0.95 {
        settings.insert("width".into(), SettingsValue::Float32(120.0));
        settings.insert("description".into(), SettingsValue::String("Widen stereo".into()));
    } else if correlation < 0.3 {
        settings.insert("width".into(), SettingsValue::Float32(90.0));
        settings.insert("description".into(), SettingsValue::String("Tighten stereo".into()));
    } else {
        settings.insert("width".into(), SettingsValue::Float32(100.0));
    }
    if !is_mono && width < 50.0 {
        settings.insert("mono_bass_freq_hz".into(), SettingsValue::Float32(150.0));
        settings.insert("mono_bass_enabled".into(), SettingsValue::Bool(true));
        settings.insert("description".into(), SettingsValue::String("Mono bass for clarity".into()));
    }
    settings.insert("enabled".into(), SettingsValue::Bool(true));
    settings
}

pub fn calculate_confidence(duration: f32, loudness_available: bool, spectral_centroid: f32) -> f32 {
    let mut confidence = 50.0f32;
    if duration > 30.0 { confidence += 10.0; }
    if loudness_available { confidence += 15.0; }
    if spectral_centroid > 0.0 { confidence += 10.0; }
    confidence.clamp(0.0, 100.0)
}
