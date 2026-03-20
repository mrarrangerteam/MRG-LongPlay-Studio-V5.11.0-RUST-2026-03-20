use std::collections::HashMap;
use std::sync::OnceLock;

use crate::irc_modes::IRCMode;
use crate::tone_presets::TonePreset;

// =====================================================
// Structs
// =====================================================

#[derive(Debug, Clone)]
pub struct EQBandSetting {
    pub frequency: f64,
    pub gain: f64,
    pub q: f64,
    pub filter_type: String,
    pub enabled: bool,
}

#[derive(Debug, Clone)]
pub struct DynamicsSetting {
    pub threshold: f64,
    pub ratio: f64,
    pub attack_ms: f64,
    pub release_ms: f64,
    pub makeup_gain: f64,
    pub knee: f64,
    pub multiband: bool,
    pub crossover_low: f64,
    pub crossover_high: f64,
}

#[derive(Debug, Clone)]
pub struct LimiterSetting {
    pub ceiling_db: f64,
    pub lookahead_ms: f64,
    pub release_ms: f64,
    pub attack_ms: f64,
    pub true_peak: bool,
}

#[derive(Debug, Clone)]
pub struct ImagerSetting {
    pub width: f64,
    pub balance: f64,
    pub mono_bass_freq: f64,
    pub multiband: bool,
}

#[derive(Debug, Clone)]
pub struct MaximizerSetting {
    pub gain_db: f64,
    pub ceiling_db: f64,
    pub character: f64,
    pub irc_mode: String,
    pub irc_sub_mode: String,
}

#[derive(Debug, Clone)]
pub struct GenreProfile {
    pub name: String,
    pub category: String,
    pub description: String,
    pub target_lufs: f64,
    pub target_true_peak: f64,
    pub target_lra: f64,
    pub eq: Vec<EQBandSetting>,
    pub dynamics: DynamicsSetting,
    pub limiter: LimiterSetting,
    pub imager: ImagerSetting,
    pub maximizer: MaximizerSetting,
}

#[derive(Debug, Clone)]
pub struct PlatformTarget {
    pub name: String,
    pub target_lufs: f64,
    pub max_true_peak: f64,
    pub description: String,
}

// =====================================================
// Data Store
// =====================================================

struct ProfileData {
    irc_modes: HashMap<String, IRCMode>,
    tone_presets: HashMap<String, TonePreset>,
    genre_profiles: HashMap<String, GenreProfile>,
    platform_targets: HashMap<String, PlatformTarget>,
}

static DATA: OnceLock<ProfileData> = OnceLock::new();

fn init_data() -> ProfileData {
    let mut irc_modes = HashMap::new();
    let mut tone_presets = HashMap::new();
    let mut genre_profiles = HashMap::new();
    let mut platform_targets = HashMap::new();

    // =====================================================
    // IRC MODES
    // =====================================================

    // Populate from irc_modes.rs (Ozone 12 matching)
    for mode in crate::irc_modes::get_irc_modes() {
        irc_modes.insert(mode.name.clone(), mode);
    }

    // =====================================================
    // TONE PRESETS
    // =====================================================

    tone_presets.insert("flat".to_string(), TonePreset {
        name: "flat".to_string(),
        low_gain: 0.0,
        mid_gain: 0.0,
        high_gain: 0.0,
        low_freq: 200.0,
        high_freq: 4000.0,
    });

    tone_presets.insert("warm".to_string(), TonePreset {
        name: "warm".to_string(),
        low_gain: 1.5,
        mid_gain: -0.5,
        high_gain: -1.0,
        low_freq: 150.0,
        high_freq: 5000.0,
    });

    tone_presets.insert("bright".to_string(), TonePreset {
        name: "bright".to_string(),
        low_gain: -0.5,
        mid_gain: 0.0,
        high_gain: 1.5,
        low_freq: 200.0,
        high_freq: 3500.0,
    });

    tone_presets.insert("dark".to_string(), TonePreset {
        name: "dark".to_string(),
        low_gain: 1.0,
        mid_gain: 0.0,
        high_gain: -2.0,
        low_freq: 250.0,
        high_freq: 4000.0,
    });

    tone_presets.insert("presence".to_string(), TonePreset {
        name: "presence".to_string(),
        low_gain: 0.0,
        mid_gain: 1.0,
        high_gain: 0.5,
        low_freq: 200.0,
        high_freq: 3000.0,
    });

    // =====================================================
    // GENRE PROFILES - ELECTRONIC
    // =====================================================

    genre_profiles.insert("edm".to_string(), GenreProfile {
        name: "EDM".to_string(),
        category: "Electronic".to_string(),
        description: "EDM / Dance Music".to_string(),
        target_lufs: -8.0,
        target_true_peak: -0.5,
        target_lra: 4.0,
        eq: vec![
            EQBandSetting { frequency: 60.0, gain: 2.0, q: 0.7, filter_type: "lowshelf".to_string(), enabled: true },
            EQBandSetting { frequency: 250.0, gain: -1.0, q: 0.8, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 1000.0, gain: 0.5, q: 1.0, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 4000.0, gain: 1.5, q: 0.8, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 12000.0, gain: 1.0, q: 0.7, filter_type: "highshelf".to_string(), enabled: true },
        ],
        dynamics: DynamicsSetting { threshold: -20.0, ratio: 2.5, attack_ms: 2.0, release_ms: 50.0, makeup_gain: 2.0, knee: 0.3, multiband: false, crossover_low: 0.0, crossover_high: 0.0 },
        limiter: LimiterSetting { ceiling_db: -0.3, lookahead_ms: 5.0, release_ms: 40.0, attack_ms: 0.1, true_peak: true },
        imager: ImagerSetting { width: 150.0, balance: 0.0, mono_bass_freq: 60.0, multiband: false },
        maximizer: MaximizerSetting { gain_db: 2.0, ceiling_db: -0.5, character: 6.0, irc_mode: "IRC4".to_string(), irc_sub_mode: "Crisp".to_string() },
    });

    genre_profiles.insert("house".to_string(), GenreProfile {
        name: "House".to_string(),
        category: "Electronic".to_string(),
        description: "House Music".to_string(),
        target_lufs: -9.0,
        target_true_peak: -0.5,
        target_lra: 4.0,
        eq: vec![
            EQBandSetting { frequency: 50.0, gain: 1.5, q: 0.7, filter_type: "lowshelf".to_string(), enabled: true },
            EQBandSetting { frequency: 200.0, gain: -0.5, q: 0.8, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 800.0, gain: 0.3, q: 1.0, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 3000.0, gain: 1.0, q: 0.8, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 10000.0, gain: 0.8, q: 0.7, filter_type: "highshelf".to_string(), enabled: true },
        ],
        dynamics: DynamicsSetting { threshold: -18.0, ratio: 2.2, attack_ms: 1.5, release_ms: 60.0, makeup_gain: 1.8, knee: 0.4, multiband: false, crossover_low: 0.0, crossover_high: 0.0 },
        limiter: LimiterSetting { ceiling_db: -0.3, lookahead_ms: 5.0, release_ms: 40.0, attack_ms: 0.1, true_peak: true },
        imager: ImagerSetting { width: 130.0, balance: 0.0, mono_bass_freq: 60.0, multiband: false },
        maximizer: MaximizerSetting { gain_db: 1.5, ceiling_db: -0.5, character: 5.0, irc_mode: "IRC2".to_string(), irc_sub_mode: "Standard".to_string() },
    });

    genre_profiles.insert("techno".to_string(), GenreProfile {
        name: "Techno".to_string(),
        category: "Electronic".to_string(),
        description: "Techno".to_string(),
        target_lufs: -9.0,
        target_true_peak: -0.5,
        target_lra: 3.5,
        eq: vec![
            EQBandSetting { frequency: 40.0, gain: 2.5, q: 0.7, filter_type: "lowshelf".to_string(), enabled: true },
            EQBandSetting { frequency: 150.0, gain: -1.5, q: 0.8, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 600.0, gain: 0.5, q: 1.0, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 2000.0, gain: 0.8, q: 0.8, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 8000.0, gain: 1.2, q: 0.7, filter_type: "highshelf".to_string(), enabled: true },
        ],
        dynamics: DynamicsSetting { threshold: -22.0, ratio: 3.0, attack_ms: 1.0, release_ms: 40.0, makeup_gain: 2.5, knee: 0.2, multiband: false, crossover_low: 0.0, crossover_high: 0.0 },
        limiter: LimiterSetting { ceiling_db: -0.2, lookahead_ms: 4.0, release_ms: 35.0, attack_ms: 0.1, true_peak: true },
        imager: ImagerSetting { width: 140.0, balance: 0.0, mono_bass_freq: 50.0, multiband: false },
        maximizer: MaximizerSetting { gain_db: 2.5, ceiling_db: -0.3, character: 7.0, irc_mode: "IRC4".to_string(), irc_sub_mode: "Saturated".to_string() },
    });

    genre_profiles.insert("dubstep".to_string(), GenreProfile {
        name: "Dubstep".to_string(),
        category: "Electronic".to_string(),
        description: "Dubstep - Massive Bass".to_string(),
        target_lufs: -8.0,
        target_true_peak: -0.5,
        target_lra: 5.0,
        eq: vec![
            EQBandSetting { frequency: 30.0, gain: 3.5, q: 0.6, filter_type: "lowshelf".to_string(), enabled: true },
            EQBandSetting { frequency: 100.0, gain: 2.0, q: 0.7, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 400.0, gain: -1.0, q: 0.8, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 1500.0, gain: 0.5, q: 1.0, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 6000.0, gain: 1.0, q: 0.7, filter_type: "highshelf".to_string(), enabled: true },
        ],
        dynamics: DynamicsSetting { threshold: -16.0, ratio: 3.5, attack_ms: 1.5, release_ms: 80.0, makeup_gain: 3.0, knee: 0.2, multiband: true, crossover_low: 50.0, crossover_high: 300.0 },
        limiter: LimiterSetting { ceiling_db: -0.2, lookahead_ms: 5.0, release_ms: 50.0, attack_ms: 0.1, true_peak: true },
        imager: ImagerSetting { width: 160.0, balance: 0.0, mono_bass_freq: 80.0, multiband: true },
        maximizer: MaximizerSetting { gain_db: 3.0, ceiling_db: -0.2, character: 8.0, irc_mode: "IRC4".to_string(), irc_sub_mode: "Warm".to_string() },
    });

    genre_profiles.insert("drum_and_bass".to_string(), GenreProfile {
        name: "Drum and Bass".to_string(),
        category: "Electronic".to_string(),
        description: "Drum and Bass".to_string(),
        target_lufs: -9.0,
        target_true_peak: -0.5,
        target_lra: 4.5,
        eq: vec![
            EQBandSetting { frequency: 45.0, gain: 1.8, q: 0.7, filter_type: "lowshelf".to_string(), enabled: true },
            EQBandSetting { frequency: 180.0, gain: -0.8, q: 0.8, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 700.0, gain: 0.3, q: 1.0, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 2500.0, gain: 0.7, q: 0.8, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 9000.0, gain: 1.3, q: 0.7, filter_type: "highshelf".to_string(), enabled: true },
        ],
        dynamics: DynamicsSetting { threshold: -20.0, ratio: 2.8, attack_ms: 0.8, release_ms: 50.0, makeup_gain: 2.2, knee: 0.3, multiband: false, crossover_low: 0.0, crossover_high: 0.0 },
        limiter: LimiterSetting { ceiling_db: -0.3, lookahead_ms: 4.5, release_ms: 38.0, attack_ms: 0.08, true_peak: true },
        imager: ImagerSetting { width: 120.0, balance: 0.0, mono_bass_freq: 60.0, multiband: false },
        maximizer: MaximizerSetting { gain_db: 2.2, ceiling_db: -0.4, character: 6.5, irc_mode: "IRC2".to_string(), irc_sub_mode: "Tight".to_string() },
    });

    genre_profiles.insert("trance".to_string(), GenreProfile {
        name: "Trance".to_string(),
        category: "Electronic".to_string(),
        description: "Trance - Wide Stereo".to_string(),
        target_lufs: -10.0,
        target_true_peak: -0.5,
        target_lra: 4.5,
        eq: vec![
            EQBandSetting { frequency: 55.0, gain: 1.2, q: 0.7, filter_type: "lowshelf".to_string(), enabled: true },
            EQBandSetting { frequency: 220.0, gain: -0.3, q: 0.8, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 900.0, gain: 0.2, q: 1.0, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 3500.0, gain: 0.9, q: 0.8, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 11000.0, gain: 0.7, q: 0.7, filter_type: "highshelf".to_string(), enabled: true },
        ],
        dynamics: DynamicsSetting { threshold: -19.0, ratio: 2.0, attack_ms: 2.0, release_ms: 55.0, makeup_gain: 1.5, knee: 0.4, multiband: false, crossover_low: 0.0, crossover_high: 0.0 },
        limiter: LimiterSetting { ceiling_db: -0.4, lookahead_ms: 5.5, release_ms: 42.0, attack_ms: 0.1, true_peak: true },
        imager: ImagerSetting { width: 160.0, balance: 0.0, mono_bass_freq: 70.0, multiband: false },
        maximizer: MaximizerSetting { gain_db: 1.8, ceiling_db: -0.6, character: 5.5, irc_mode: "IRC1".to_string(), irc_sub_mode: "Modern".to_string() },
    });

    genre_profiles.insert("synthwave".to_string(), GenreProfile {
        name: "Synthwave".to_string(),
        category: "Electronic".to_string(),
        description: "Synthwave - Warm Synths".to_string(),
        target_lufs: -10.0,
        target_true_peak: -0.5,
        target_lra: 4.0,
        eq: vec![
            EQBandSetting { frequency: 70.0, gain: 1.8, q: 0.7, filter_type: "lowshelf".to_string(), enabled: true },
            EQBandSetting { frequency: 300.0, gain: 0.5, q: 0.8, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 1200.0, gain: -0.2, q: 1.0, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 3000.0, gain: 0.6, q: 0.8, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 10000.0, gain: 0.5, q: 0.7, filter_type: "highshelf".to_string(), enabled: true },
        ],
        dynamics: DynamicsSetting { threshold: -21.0, ratio: 1.8, attack_ms: 2.5, release_ms: 60.0, makeup_gain: 1.2, knee: 0.5, multiband: false, crossover_low: 0.0, crossover_high: 0.0 },
        limiter: LimiterSetting { ceiling_db: -0.4, lookahead_ms: 6.0, release_ms: 45.0, attack_ms: 0.12, true_peak: true },
        imager: ImagerSetting { width: 140.0, balance: 0.0, mono_bass_freq: 65.0, multiband: false },
        maximizer: MaximizerSetting { gain_db: 1.5, ceiling_db: -0.7, character: 4.5, irc_mode: "IRC1".to_string(), irc_sub_mode: "Vintage".to_string() },
    });

    // =====================================================
    // GENRE PROFILES - ROCK
    // =====================================================

    genre_profiles.insert("rock".to_string(), GenreProfile {
        name: "Rock".to_string(),
        category: "Rock".to_string(),
        description: "Rock Music".to_string(),
        target_lufs: -10.0,
        target_true_peak: -0.8,
        target_lra: 5.0,
        eq: vec![
            EQBandSetting { frequency: 60.0, gain: 1.0, q: 0.7, filter_type: "lowshelf".to_string(), enabled: true },
            EQBandSetting { frequency: 250.0, gain: -0.5, q: 0.8, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 1000.0, gain: 0.3, q: 1.0, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 3500.0, gain: 0.8, q: 0.8, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 10000.0, gain: 1.0, q: 0.7, filter_type: "highshelf".to_string(), enabled: true },
        ],
        dynamics: DynamicsSetting { threshold: -18.0, ratio: 2.0, attack_ms: 3.0, release_ms: 80.0, makeup_gain: 2.0, knee: 0.4, multiband: false, crossover_low: 0.0, crossover_high: 0.0 },
        limiter: LimiterSetting { ceiling_db: -0.5, lookahead_ms: 4.5, release_ms: 50.0, attack_ms: 0.2, true_peak: true },
        imager: ImagerSetting { width: 120.0, balance: 0.0, mono_bass_freq: 55.0, multiband: false },
        maximizer: MaximizerSetting { gain_db: 1.8, ceiling_db: -0.8, character: 5.0, irc_mode: "IRC2".to_string(), irc_sub_mode: "Standard".to_string() },
    });

    genre_profiles.insert("metal".to_string(), GenreProfile {
        name: "Metal".to_string(),
        category: "Rock".to_string(),
        description: "Metal - Aggressive".to_string(),
        target_lufs: -8.0,
        target_true_peak: -0.3,
        target_lra: 6.0,
        eq: vec![
            EQBandSetting { frequency: 50.0, gain: 2.5, q: 0.7, filter_type: "lowshelf".to_string(), enabled: true },
            EQBandSetting { frequency: 150.0, gain: -2.0, q: 0.8, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 500.0, gain: 0.8, q: 1.0, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 2000.0, gain: 1.2, q: 0.8, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 8000.0, gain: 2.0, q: 0.7, filter_type: "highshelf".to_string(), enabled: true },
        ],
        dynamics: DynamicsSetting { threshold: -16.0, ratio: 3.5, attack_ms: 1.0, release_ms: 40.0, makeup_gain: 3.5, knee: 0.2, multiband: true, crossover_low: 50.0, crossover_high: 400.0 },
        limiter: LimiterSetting { ceiling_db: -0.2, lookahead_ms: 3.5, release_ms: 40.0, attack_ms: 0.08, true_peak: true },
        imager: ImagerSetting { width: 100.0, balance: 0.0, mono_bass_freq: 50.0, multiband: false },
        maximizer: MaximizerSetting { gain_db: 2.8, ceiling_db: -0.2, character: 8.5, irc_mode: "IRC2".to_string(), irc_sub_mode: "Tight".to_string() },
    });

    genre_profiles.insert("punk".to_string(), GenreProfile {
        name: "Punk".to_string(),
        category: "Rock".to_string(),
        description: "Punk Rock".to_string(),
        target_lufs: -9.0,
        target_true_peak: -0.8,
        target_lra: 5.5,
        eq: vec![
            EQBandSetting { frequency: 50.0, gain: 1.2, q: 0.7, filter_type: "lowshelf".to_string(), enabled: true },
            EQBandSetting { frequency: 200.0, gain: -0.8, q: 0.8, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 800.0, gain: 0.5, q: 1.0, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 3000.0, gain: 1.0, q: 0.8, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 9000.0, gain: 1.5, q: 0.7, filter_type: "highshelf".to_string(), enabled: true },
        ],
        dynamics: DynamicsSetting { threshold: -17.0, ratio: 2.5, attack_ms: 2.0, release_ms: 50.0, makeup_gain: 2.5, knee: 0.3, multiband: false, crossover_low: 0.0, crossover_high: 0.0 },
        limiter: LimiterSetting { ceiling_db: -0.6, lookahead_ms: 4.0, release_ms: 45.0, attack_ms: 0.15, true_peak: true },
        imager: ImagerSetting { width: 110.0, balance: 0.0, mono_bass_freq: 50.0, multiband: false },
        maximizer: MaximizerSetting { gain_db: 2.0, ceiling_db: -0.6, character: 6.0, irc_mode: "IRC2".to_string(), irc_sub_mode: "Bouncy".to_string() },
    });

    genre_profiles.insert("alternative".to_string(), GenreProfile {
        name: "Alternative".to_string(),
        category: "Rock".to_string(),
        description: "Alternative Rock".to_string(),
        target_lufs: -11.0,
        target_true_peak: -1.0,
        target_lra: 5.5,
        eq: vec![
            EQBandSetting { frequency: 55.0, gain: 0.8, q: 0.7, filter_type: "lowshelf".to_string(), enabled: true },
            EQBandSetting { frequency: 280.0, gain: -0.3, q: 0.8, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 1100.0, gain: 0.2, q: 1.0, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 3800.0, gain: 0.5, q: 0.8, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 11000.0, gain: 0.7, q: 0.7, filter_type: "highshelf".to_string(), enabled: true },
        ],
        dynamics: DynamicsSetting { threshold: -19.0, ratio: 1.8, attack_ms: 3.5, release_ms: 90.0, makeup_gain: 1.5, knee: 0.5, multiband: false, crossover_low: 0.0, crossover_high: 0.0 },
        limiter: LimiterSetting { ceiling_db: -0.8, lookahead_ms: 5.0, release_ms: 55.0, attack_ms: 0.25, true_peak: true },
        imager: ImagerSetting { width: 130.0, balance: 0.0, mono_bass_freq: 60.0, multiband: false },
        maximizer: MaximizerSetting { gain_db: 1.5, ceiling_db: -1.0, character: 4.5, irc_mode: "IRC3".to_string(), irc_sub_mode: "Clean".to_string() },
    });

    genre_profiles.insert("indie_rock".to_string(), GenreProfile {
        name: "Indie Rock".to_string(),
        category: "Rock".to_string(),
        description: "Indie Rock".to_string(),
        target_lufs: -12.0,
        target_true_peak: -1.0,
        target_lra: 6.0,
        eq: vec![
            EQBandSetting { frequency: 60.0, gain: 0.5, q: 0.7, filter_type: "lowshelf".to_string(), enabled: true },
            EQBandSetting { frequency: 300.0, gain: -0.2, q: 0.8, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 1200.0, gain: 0.1, q: 1.0, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 4000.0, gain: 0.3, q: 0.8, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 12000.0, gain: 0.5, q: 0.7, filter_type: "highshelf".to_string(), enabled: true },
        ],
        dynamics: DynamicsSetting { threshold: -20.0, ratio: 1.5, attack_ms: 4.0, release_ms: 100.0, makeup_gain: 1.2, knee: 0.6, multiband: false, crossover_low: 0.0, crossover_high: 0.0 },
        limiter: LimiterSetting { ceiling_db: -0.9, lookahead_ms: 5.5, release_ms: 60.0, attack_ms: 0.3, true_peak: true },
        imager: ImagerSetting { width: 140.0, balance: 0.0, mono_bass_freq: 65.0, multiband: false },
        maximizer: MaximizerSetting { gain_db: 1.2, ceiling_db: -1.2, character: 3.5, irc_mode: "IRC3".to_string(), irc_sub_mode: "Smooth".to_string() },
    });

    // =====================================================
    // GENRE PROFILES - POP
    // =====================================================

    genre_profiles.insert("pop".to_string(), GenreProfile {
        name: "Pop".to_string(),
        category: "Pop".to_string(),
        description: "Pop Music - Bright".to_string(),
        target_lufs: -10.0,
        target_true_peak: -0.8,
        target_lra: 4.5,
        eq: vec![
            EQBandSetting { frequency: 60.0, gain: 0.8, q: 0.7, filter_type: "lowshelf".to_string(), enabled: true },
            EQBandSetting { frequency: 200.0, gain: -0.3, q: 0.8, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 1000.0, gain: 0.5, q: 1.0, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 3500.0, gain: 1.2, q: 0.8, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 11000.0, gain: 1.0, q: 0.7, filter_type: "highshelf".to_string(), enabled: true },
        ],
        dynamics: DynamicsSetting { threshold: -18.0, ratio: 2.2, attack_ms: 2.5, release_ms: 70.0, makeup_gain: 2.0, knee: 0.4, multiband: false, crossover_low: 0.0, crossover_high: 0.0 },
        limiter: LimiterSetting { ceiling_db: -0.6, lookahead_ms: 4.5, release_ms: 50.0, attack_ms: 0.15, true_peak: true },
        imager: ImagerSetting { width: 140.0, balance: 0.0, mono_bass_freq: 70.0, multiband: false },
        maximizer: MaximizerSetting { gain_db: 1.8, ceiling_db: -0.7, character: 5.0, irc_mode: "IRC2".to_string(), irc_sub_mode: "Standard".to_string() },
    });

    genre_profiles.insert("kpop".to_string(), GenreProfile {
        name: "K-Pop".to_string(),
        category: "Pop".to_string(),
        description: "K-Pop - Bright Wide".to_string(),
        target_lufs: -9.0,
        target_true_peak: -0.5,
        target_lra: 4.0,
        eq: vec![
            EQBandSetting { frequency: 50.0, gain: 1.2, q: 0.7, filter_type: "lowshelf".to_string(), enabled: true },
            EQBandSetting { frequency: 180.0, gain: -0.5, q: 0.8, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 900.0, gain: 0.3, q: 1.0, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 3200.0, gain: 1.5, q: 0.8, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 10000.0, gain: 1.2, q: 0.7, filter_type: "highshelf".to_string(), enabled: true },
        ],
        dynamics: DynamicsSetting { threshold: -17.0, ratio: 2.5, attack_ms: 2.0, release_ms: 60.0, makeup_gain: 2.2, knee: 0.3, multiband: false, crossover_low: 0.0, crossover_high: 0.0 },
        limiter: LimiterSetting { ceiling_db: -0.5, lookahead_ms: 4.0, release_ms: 45.0, attack_ms: 0.1, true_peak: true },
        imager: ImagerSetting { width: 150.0, balance: 0.0, mono_bass_freq: 75.0, multiband: false },
        maximizer: MaximizerSetting { gain_db: 2.0, ceiling_db: -0.5, character: 5.5, irc_mode: "IRC2".to_string(), irc_sub_mode: "Standard".to_string() },
    });

    genre_profiles.insert("jpop".to_string(), GenreProfile {
        name: "J-Pop".to_string(),
        category: "Pop".to_string(),
        description: "J-Pop".to_string(),
        target_lufs: -10.0,
        target_true_peak: -0.8,
        target_lra: 4.5,
        eq: vec![
            EQBandSetting { frequency: 55.0, gain: 0.9, q: 0.7, filter_type: "lowshelf".to_string(), enabled: true },
            EQBandSetting { frequency: 220.0, gain: -0.2, q: 0.8, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 1000.0, gain: 0.4, q: 1.0, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 3500.0, gain: 1.1, q: 0.8, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 10500.0, gain: 0.9, q: 0.7, filter_type: "highshelf".to_string(), enabled: true },
        ],
        dynamics: DynamicsSetting { threshold: -18.0, ratio: 2.1, attack_ms: 2.3, release_ms: 65.0, makeup_gain: 1.9, knee: 0.4, multiband: false, crossover_low: 0.0, crossover_high: 0.0 },
        limiter: LimiterSetting { ceiling_db: -0.7, lookahead_ms: 4.3, release_ms: 48.0, attack_ms: 0.13, true_peak: true },
        imager: ImagerSetting { width: 135.0, balance: 0.0, mono_bass_freq: 68.0, multiband: false },
        maximizer: MaximizerSetting { gain_db: 1.7, ceiling_db: -0.75, character: 5.0, irc_mode: "IRC2".to_string(), irc_sub_mode: "Standard".to_string() },
    });

    genre_profiles.insert("latin_pop".to_string(), GenreProfile {
        name: "Latin Pop".to_string(),
        category: "Pop".to_string(),
        description: "Latin Pop".to_string(),
        target_lufs: -10.0,
        target_true_peak: -0.8,
        target_lra: 5.0,
        eq: vec![
            EQBandSetting { frequency: 65.0, gain: 1.5, q: 0.7, filter_type: "lowshelf".to_string(), enabled: true },
            EQBandSetting { frequency: 250.0, gain: -0.4, q: 0.8, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 1100.0, gain: 0.4, q: 1.0, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 3800.0, gain: 1.0, q: 0.8, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 10500.0, gain: 0.8, q: 0.7, filter_type: "highshelf".to_string(), enabled: true },
        ],
        dynamics: DynamicsSetting { threshold: -18.0, ratio: 2.0, attack_ms: 2.8, release_ms: 75.0, makeup_gain: 1.8, knee: 0.4, multiband: true, crossover_low: 60.0, crossover_high: 300.0 },
        limiter: LimiterSetting { ceiling_db: -0.7, lookahead_ms: 4.5, release_ms: 50.0, attack_ms: 0.15, true_peak: true },
        imager: ImagerSetting { width: 145.0, balance: 0.0, mono_bass_freq: 65.0, multiband: false },
        maximizer: MaximizerSetting { gain_db: 1.6, ceiling_db: -0.8, character: 5.0, irc_mode: "IRC1".to_string(), irc_sub_mode: "Modern".to_string() },
    });

    // =====================================================
    // GENRE PROFILES - HIP-HOP
    // =====================================================

    genre_profiles.insert("hip_hop".to_string(), GenreProfile {
        name: "Hip-Hop".to_string(),
        category: "Hip-Hop".to_string(),
        description: "Hip-Hop - Heavy Bass".to_string(),
        target_lufs: -9.0,
        target_true_peak: -0.5,
        target_lra: 4.5,
        eq: vec![
            EQBandSetting { frequency: 40.0, gain: 2.0, q: 0.7, filter_type: "lowshelf".to_string(), enabled: true },
            EQBandSetting { frequency: 120.0, gain: 1.5, q: 0.8, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 600.0, gain: -0.5, q: 1.0, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 2500.0, gain: 0.6, q: 0.8, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 8000.0, gain: 0.9, q: 0.7, filter_type: "highshelf".to_string(), enabled: true },
        ],
        dynamics: DynamicsSetting { threshold: -17.0, ratio: 2.8, attack_ms: 1.8, release_ms: 70.0, makeup_gain: 2.5, knee: 0.3, multiband: true, crossover_low: 50.0, crossover_high: 250.0 },
        limiter: LimiterSetting { ceiling_db: -0.5, lookahead_ms: 5.0, release_ms: 55.0, attack_ms: 0.12, true_peak: true },
        imager: ImagerSetting { width: 120.0, balance: 0.0, mono_bass_freq: 80.0, multiband: true },
        maximizer: MaximizerSetting { gain_db: 2.2, ceiling_db: -0.5, character: 6.0, irc_mode: "IRC4".to_string(), irc_sub_mode: "Warm".to_string() },
    });

    genre_profiles.insert("trap".to_string(), GenreProfile {
        name: "Trap".to_string(),
        category: "Hip-Hop".to_string(),
        description: "Trap - Sub Bass".to_string(),
        target_lufs: -8.0,
        target_true_peak: -0.3,
        target_lra: 4.0,
        eq: vec![
            EQBandSetting { frequency: 30.0, gain: 3.0, q: 0.6, filter_type: "lowshelf".to_string(), enabled: true },
            EQBandSetting { frequency: 80.0, gain: 2.5, q: 0.7, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 400.0, gain: -1.0, q: 0.8, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 2000.0, gain: 0.7, q: 1.0, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 7000.0, gain: 1.2, q: 0.7, filter_type: "highshelf".to_string(), enabled: true },
        ],
        dynamics: DynamicsSetting { threshold: -16.0, ratio: 3.2, attack_ms: 1.5, release_ms: 50.0, makeup_gain: 3.0, knee: 0.2, multiband: true, crossover_low: 40.0, crossover_high: 200.0 },
        limiter: LimiterSetting { ceiling_db: -0.3, lookahead_ms: 4.5, release_ms: 50.0, attack_ms: 0.1, true_peak: true },
        imager: ImagerSetting { width: 110.0, balance: 0.0, mono_bass_freq: 85.0, multiband: true },
        maximizer: MaximizerSetting { gain_db: 2.8, ceiling_db: -0.2, character: 8.0, irc_mode: "IRC4".to_string(), irc_sub_mode: "Saturated".to_string() },
    });

    genre_profiles.insert("lofi_hiphop".to_string(), GenreProfile {
        name: "Lo-Fi Hip-Hop".to_string(),
        category: "Hip-Hop".to_string(),
        description: "Lo-Fi Hip-Hop - Warm".to_string(),
        target_lufs: -14.0,
        target_true_peak: -2.0,
        target_lra: 6.0,
        eq: vec![
            EQBandSetting { frequency: 70.0, gain: 2.5, q: 0.7, filter_type: "lowshelf".to_string(), enabled: true },
            EQBandSetting { frequency: 200.0, gain: 1.5, q: 0.8, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 1000.0, gain: 0.2, q: 1.0, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 3000.0, gain: -0.5, q: 0.8, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 9000.0, gain: -1.0, q: 0.7, filter_type: "highshelf".to_string(), enabled: true },
        ],
        dynamics: DynamicsSetting { threshold: -24.0, ratio: 1.2, attack_ms: 5.0, release_ms: 100.0, makeup_gain: 0.8, knee: 0.8, multiband: false, crossover_low: 0.0, crossover_high: 0.0 },
        limiter: LimiterSetting { ceiling_db: -1.8, lookahead_ms: 6.5, release_ms: 80.0, attack_ms: 0.4, true_peak: false },
        imager: ImagerSetting { width: 110.0, balance: 0.0, mono_bass_freq: 40.0, multiband: false },
        maximizer: MaximizerSetting { gain_db: 0.8, ceiling_db: -2.0, character: 2.0, irc_mode: "IRC1".to_string(), irc_sub_mode: "Vintage".to_string() },
    });

    genre_profiles.insert("rnb".to_string(), GenreProfile {
        name: "R&B".to_string(),
        category: "Hip-Hop".to_string(),
        description: "R&B - Smooth".to_string(),
        target_lufs: -12.0,
        target_true_peak: -1.0,
        target_lra: 5.0,
        eq: vec![
            EQBandSetting { frequency: 50.0, gain: 1.5, q: 0.7, filter_type: "lowshelf".to_string(), enabled: true },
            EQBandSetting { frequency: 150.0, gain: 0.5, q: 0.8, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 800.0, gain: 0.3, q: 1.0, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 3000.0, gain: 0.8, q: 0.8, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 9500.0, gain: 0.6, q: 0.7, filter_type: "highshelf".to_string(), enabled: true },
        ],
        dynamics: DynamicsSetting { threshold: -19.0, ratio: 1.8, attack_ms: 3.5, release_ms: 80.0, makeup_gain: 1.5, knee: 0.5, multiband: false, crossover_low: 0.0, crossover_high: 0.0 },
        limiter: LimiterSetting { ceiling_db: -0.9, lookahead_ms: 5.5, release_ms: 60.0, attack_ms: 0.2, true_peak: true },
        imager: ImagerSetting { width: 125.0, balance: 0.0, mono_bass_freq: 65.0, multiband: false },
        maximizer: MaximizerSetting { gain_db: 1.5, ceiling_db: -1.0, character: 4.0, irc_mode: "IRC3".to_string(), irc_sub_mode: "Smooth".to_string() },
    });

    // =====================================================
    // GENRE PROFILES - ACOUSTIC/ORCHESTRAL
    // =====================================================

    genre_profiles.insert("acoustic".to_string(), GenreProfile {
        name: "Acoustic".to_string(),
        category: "Acoustic".to_string(),
        description: "Acoustic - Minimal Processing".to_string(),
        target_lufs: -16.0,
        target_true_peak: -1.5,
        target_lra: 7.0,
        eq: vec![
            EQBandSetting { frequency: 80.0, gain: 0.3, q: 0.7, filter_type: "lowshelf".to_string(), enabled: true },
            EQBandSetting { frequency: 250.0, gain: -0.2, q: 0.8, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 1200.0, gain: 0.1, q: 1.0, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 4000.0, gain: 0.2, q: 0.8, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 10000.0, gain: 0.3, q: 0.7, filter_type: "highshelf".to_string(), enabled: true },
        ],
        dynamics: DynamicsSetting { threshold: -22.0, ratio: 1.2, attack_ms: 5.0, release_ms: 120.0, makeup_gain: 0.5, knee: 0.8, multiband: false, crossover_low: 0.0, crossover_high: 0.0 },
        limiter: LimiterSetting { ceiling_db: -1.4, lookahead_ms: 6.0, release_ms: 80.0, attack_ms: 0.3, true_peak: false },
        imager: ImagerSetting { width: 120.0, balance: 0.0, mono_bass_freq: 50.0, multiband: false },
        maximizer: MaximizerSetting { gain_db: 0.5, ceiling_db: -1.5, character: 1.5, irc_mode: "IRC3".to_string(), irc_sub_mode: "Clean".to_string() },
    });

    genre_profiles.insert("jazz".to_string(), GenreProfile {
        name: "Jazz".to_string(),
        category: "Acoustic".to_string(),
        description: "Jazz - Transparent".to_string(),
        target_lufs: -18.0,
        target_true_peak: -1.5,
        target_lra: 8.0,
        eq: vec![
            EQBandSetting { frequency: 90.0, gain: 0.2, q: 0.7, filter_type: "lowshelf".to_string(), enabled: true },
            EQBandSetting { frequency: 280.0, gain: -0.1, q: 0.8, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 1300.0, gain: 0.0, q: 1.0, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 4200.0, gain: 0.1, q: 0.8, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 11000.0, gain: 0.2, q: 0.7, filter_type: "highshelf".to_string(), enabled: true },
        ],
        dynamics: DynamicsSetting { threshold: -23.0, ratio: 1.0, attack_ms: 6.0, release_ms: 130.0, makeup_gain: 0.3, knee: 1.0, multiband: false, crossover_low: 0.0, crossover_high: 0.0 },
        limiter: LimiterSetting { ceiling_db: -1.4, lookahead_ms: 7.0, release_ms: 90.0, attack_ms: 0.4, true_peak: false },
        imager: ImagerSetting { width: 130.0, balance: 0.0, mono_bass_freq: 45.0, multiband: false },
        maximizer: MaximizerSetting { gain_db: 0.3, ceiling_db: -1.5, character: 0.5, irc_mode: "IRC3".to_string(), irc_sub_mode: "Silky".to_string() },
    });

    genre_profiles.insert("classical".to_string(), GenreProfile {
        name: "Classical".to_string(),
        category: "Acoustic".to_string(),
        description: "Classical - Wide Dynamic Range".to_string(),
        target_lufs: -20.0,
        target_true_peak: -1.0,
        target_lra: 10.0,
        eq: vec![
            EQBandSetting { frequency: 100.0, gain: 0.1, q: 0.7, filter_type: "lowshelf".to_string(), enabled: true },
            EQBandSetting { frequency: 300.0, gain: 0.0, q: 0.8, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 1400.0, gain: 0.0, q: 1.0, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 4500.0, gain: 0.1, q: 0.8, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 12000.0, gain: 0.1, q: 0.7, filter_type: "highshelf".to_string(), enabled: true },
        ],
        dynamics: DynamicsSetting { threshold: -25.0, ratio: 1.0, attack_ms: 8.0, release_ms: 150.0, makeup_gain: 0.0, knee: 1.0, multiband: false, crossover_low: 0.0, crossover_high: 0.0 },
        limiter: LimiterSetting { ceiling_db: -0.9, lookahead_ms: 8.0, release_ms: 100.0, attack_ms: 0.5, true_peak: false },
        imager: ImagerSetting { width: 140.0, balance: 0.0, mono_bass_freq: 40.0, multiband: false },
        maximizer: MaximizerSetting { gain_db: 0.2, ceiling_db: -1.0, character: 0.0, irc_mode: "IRC4_Surgical".to_string(), irc_sub_mode: "Invisible".to_string() },
    });

    genre_profiles.insert("folk".to_string(), GenreProfile {
        name: "Folk".to_string(),
        category: "Acoustic".to_string(),
        description: "Folk Music".to_string(),
        target_lufs: -16.0,
        target_true_peak: -1.5,
        target_lra: 7.5,
        eq: vec![
            EQBandSetting { frequency: 85.0, gain: 0.4, q: 0.7, filter_type: "lowshelf".to_string(), enabled: true },
            EQBandSetting { frequency: 260.0, gain: -0.1, q: 0.8, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 1250.0, gain: 0.1, q: 1.0, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 4100.0, gain: 0.2, q: 0.8, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 10200.0, gain: 0.3, q: 0.7, filter_type: "highshelf".to_string(), enabled: true },
        ],
        dynamics: DynamicsSetting { threshold: -21.0, ratio: 1.1, attack_ms: 5.5, release_ms: 125.0, makeup_gain: 0.4, knee: 0.9, multiband: false, crossover_low: 0.0, crossover_high: 0.0 },
        limiter: LimiterSetting { ceiling_db: -1.4, lookahead_ms: 6.5, release_ms: 85.0, attack_ms: 0.35, true_peak: false },
        imager: ImagerSetting { width: 125.0, balance: 0.0, mono_bass_freq: 48.0, multiband: false },
        maximizer: MaximizerSetting { gain_db: 0.4, ceiling_db: -1.5, character: 1.0, irc_mode: "IRC3".to_string(), irc_sub_mode: "Smooth".to_string() },
    });

    genre_profiles.insert("blues".to_string(), GenreProfile {
        name: "Blues".to_string(),
        category: "Acoustic".to_string(),
        description: "Blues".to_string(),
        target_lufs: -14.0,
        target_true_peak: -1.5,
        target_lra: 7.0,
        eq: vec![
            EQBandSetting { frequency: 75.0, gain: 1.0, q: 0.7, filter_type: "lowshelf".to_string(), enabled: true },
            EQBandSetting { frequency: 220.0, gain: 0.3, q: 0.8, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 1000.0, gain: 0.2, q: 1.0, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 3500.0, gain: 0.5, q: 0.8, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 9500.0, gain: 0.4, q: 0.7, filter_type: "highshelf".to_string(), enabled: true },
        ],
        dynamics: DynamicsSetting { threshold: -20.0, ratio: 1.3, attack_ms: 4.5, release_ms: 110.0, makeup_gain: 0.6, knee: 0.7, multiband: false, crossover_low: 0.0, crossover_high: 0.0 },
        limiter: LimiterSetting { ceiling_db: -1.4, lookahead_ms: 6.0, release_ms: 75.0, attack_ms: 0.3, true_peak: false },
        imager: ImagerSetting { width: 120.0, balance: 0.0, mono_bass_freq: 55.0, multiband: false },
        maximizer: MaximizerSetting { gain_db: 0.6, ceiling_db: -1.5, character: 2.0, irc_mode: "IRC1".to_string(), irc_sub_mode: "Modern".to_string() },
    });

    // =====================================================
    // GENRE PROFILES - AMBIENT/OTHER
    // =====================================================

    genre_profiles.insert("ambient".to_string(), GenreProfile {
        name: "Ambient".to_string(),
        category: "Ambient".to_string(),
        description: "Ambient - Very Wide".to_string(),
        target_lufs: -18.0,
        target_true_peak: -2.0,
        target_lra: 8.0,
        eq: vec![
            EQBandSetting { frequency: 95.0, gain: 0.1, q: 0.7, filter_type: "lowshelf".to_string(), enabled: true },
            EQBandSetting { frequency: 300.0, gain: -0.1, q: 0.8, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 1500.0, gain: 0.0, q: 1.0, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 4500.0, gain: 0.0, q: 0.8, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 12000.0, gain: 0.1, q: 0.7, filter_type: "highshelf".to_string(), enabled: true },
        ],
        dynamics: DynamicsSetting { threshold: -23.0, ratio: 1.0, attack_ms: 7.0, release_ms: 140.0, makeup_gain: 0.2, knee: 1.0, multiband: false, crossover_low: 0.0, crossover_high: 0.0 },
        limiter: LimiterSetting { ceiling_db: -1.9, lookahead_ms: 7.5, release_ms: 95.0, attack_ms: 0.45, true_peak: false },
        imager: ImagerSetting { width: 150.0, balance: 0.0, mono_bass_freq: 35.0, multiband: false },
        maximizer: MaximizerSetting { gain_db: 0.2, ceiling_db: -2.0, character: 0.0, irc_mode: "IRC3".to_string(), irc_sub_mode: "Silky".to_string() },
    });

    genre_profiles.insert("chillout".to_string(), GenreProfile {
        name: "Chillout".to_string(),
        category: "Ambient".to_string(),
        description: "Chillout".to_string(),
        target_lufs: -14.0,
        target_true_peak: -1.5,
        target_lra: 6.0,
        eq: vec![
            EQBandSetting { frequency: 70.0, gain: 0.8, q: 0.7, filter_type: "lowshelf".to_string(), enabled: true },
            EQBandSetting { frequency: 240.0, gain: -0.2, q: 0.8, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 1100.0, gain: 0.2, q: 1.0, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 3800.0, gain: 0.3, q: 0.8, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 10000.0, gain: 0.2, q: 0.7, filter_type: "highshelf".to_string(), enabled: true },
        ],
        dynamics: DynamicsSetting { threshold: -20.0, ratio: 1.1, attack_ms: 5.0, release_ms: 100.0, makeup_gain: 0.5, knee: 0.8, multiband: false, crossover_low: 0.0, crossover_high: 0.0 },
        limiter: LimiterSetting { ceiling_db: -1.4, lookahead_ms: 6.5, release_ms: 70.0, attack_ms: 0.3, true_peak: false },
        imager: ImagerSetting { width: 140.0, balance: 0.0, mono_bass_freq: 60.0, multiband: false },
        maximizer: MaximizerSetting { gain_db: 0.8, ceiling_db: -1.5, character: 2.5, irc_mode: "IRC1".to_string(), irc_sub_mode: "Vintage".to_string() },
    });

    genre_profiles.insert("soundtrack".to_string(), GenreProfile {
        name: "Soundtrack".to_string(),
        category: "Ambient".to_string(),
        description: "Soundtrack / Score".to_string(),
        target_lufs: -16.0,
        target_true_peak: -1.5,
        target_lra: 8.0,
        eq: vec![
            EQBandSetting { frequency: 80.0, gain: 0.2, q: 0.7, filter_type: "lowshelf".to_string(), enabled: true },
            EQBandSetting { frequency: 270.0, gain: 0.0, q: 0.8, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 1300.0, gain: 0.1, q: 1.0, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 4200.0, gain: 0.2, q: 0.8, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 11000.0, gain: 0.2, q: 0.7, filter_type: "highshelf".to_string(), enabled: true },
        ],
        dynamics: DynamicsSetting { threshold: -21.0, ratio: 1.0, attack_ms: 6.0, release_ms: 120.0, makeup_gain: 0.4, knee: 0.9, multiband: false, crossover_low: 0.0, crossover_high: 0.0 },
        limiter: LimiterSetting { ceiling_db: -1.4, lookahead_ms: 6.5, release_ms: 85.0, attack_ms: 0.35, true_peak: false },
        imager: ImagerSetting { width: 135.0, balance: 0.0, mono_bass_freq: 50.0, multiband: false },
        maximizer: MaximizerSetting { gain_db: 0.4, ceiling_db: -1.5, character: 1.0, irc_mode: "IRC3".to_string(), irc_sub_mode: "Gentle".to_string() },
    });

    genre_profiles.insert("podcast".to_string(), GenreProfile {
        name: "Podcast".to_string(),
        category: "Ambient".to_string(),
        description: "Podcast / Voice".to_string(),
        target_lufs: -16.0,
        target_true_peak: -1.5,
        target_lra: 4.0,
        eq: vec![
            EQBandSetting { frequency: 80.0, gain: -1.0, q: 0.7, filter_type: "lowshelf".to_string(), enabled: true },
            EQBandSetting { frequency: 200.0, gain: 0.5, q: 0.8, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 1000.0, gain: 1.5, q: 1.0, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 3000.0, gain: 1.0, q: 0.8, filter_type: "peak".to_string(), enabled: true },
            EQBandSetting { frequency: 8000.0, gain: 0.5, q: 0.7, filter_type: "highshelf".to_string(), enabled: true },
        ],
        dynamics: DynamicsSetting { threshold: -18.0, ratio: 3.0, attack_ms: 4.0, release_ms: 100.0, makeup_gain: 3.0, knee: 0.5, multiband: true, crossover_low: 100.0, crossover_high: 2000.0 },
        limiter: LimiterSetting { ceiling_db: -1.4, lookahead_ms: 5.0, release_ms: 60.0, attack_ms: 0.2, true_peak: false },
        imager: ImagerSetting { width: 100.0, balance: 0.0, mono_bass_freq: 40.0, multiband: false },
        maximizer: MaximizerSetting { gain_db: 1.0, ceiling_db: -1.5, character: 3.0, irc_mode: "IRC2".to_string(), irc_sub_mode: "Standard".to_string() },
    });

    // =====================================================
    // PLATFORM TARGETS
    // =====================================================

    platform_targets.insert("spotify".to_string(), PlatformTarget {
        name: "spotify".to_string(),
        target_lufs: -14.0,
        max_true_peak: -1.0,
        description: "Spotify streaming platform".to_string(),
    });

    platform_targets.insert("apple_music".to_string(), PlatformTarget {
        name: "apple_music".to_string(),
        target_lufs: -16.0,
        max_true_peak: -1.0,
        description: "Apple Music streaming platform".to_string(),
    });

    platform_targets.insert("youtube".to_string(), PlatformTarget {
        name: "youtube".to_string(),
        target_lufs: -14.0,
        max_true_peak: -1.0,
        description: "YouTube video platform".to_string(),
    });

    platform_targets.insert("tidal".to_string(), PlatformTarget {
        name: "tidal".to_string(),
        target_lufs: -14.0,
        max_true_peak: -1.0,
        description: "TIDAL streaming platform".to_string(),
    });

    platform_targets.insert("amazon".to_string(), PlatformTarget {
        name: "amazon".to_string(),
        target_lufs: -14.0,
        max_true_peak: -2.0,
        description: "Amazon Music streaming platform".to_string(),
    });

    platform_targets.insert("soundcloud".to_string(), PlatformTarget {
        name: "soundcloud".to_string(),
        target_lufs: -14.0,
        max_true_peak: -1.0,
        description: "SoundCloud streaming platform".to_string(),
    });

    platform_targets.insert("radio".to_string(), PlatformTarget {
        name: "radio".to_string(),
        target_lufs: -10.0,
        max_true_peak: -1.0,
        description: "Radio broadcast".to_string(),
    });

    platform_targets.insert("cd".to_string(), PlatformTarget {
        name: "cd".to_string(),
        target_lufs: -9.0,
        max_true_peak: -0.3,
        description: "Compact Disc".to_string(),
    });

    ProfileData {
        irc_modes,
        tone_presets,
        genre_profiles,
        platform_targets,
    }
}

fn data() -> &'static ProfileData {
    DATA.get_or_init(init_data)
}

// =====================================================
// Public API - GenreProfiles
// =====================================================

pub struct GenreProfiles;

impl GenreProfiles {
    // IRC Modes
    pub fn get_irc_modes() -> &'static HashMap<String, IRCMode> {
        &data().irc_modes
    }

    pub fn get_irc_mode(name: &str) -> Option<&'static IRCMode> {
        data().irc_modes.get(name)
    }

    pub fn get_irc_sub_modes(mode_name: &str) -> Option<Vec<String>> {
        data().irc_modes.get(mode_name).map(|mode| {
            mode.sub_modes.iter().map(|s| s.name.clone()).collect()
        })
    }

    // Tone Presets
    pub fn get_tone_presets() -> &'static HashMap<String, TonePreset> {
        &data().tone_presets
    }

    pub fn get_tone_preset(name: &str) -> Option<&'static TonePreset> {
        data().tone_presets.get(name)
    }

    // Genre Profiles
    pub fn get_genre_profiles() -> &'static HashMap<String, GenreProfile> {
        &data().genre_profiles
    }

    pub fn get_genre_profile(name: &str) -> Option<&'static GenreProfile> {
        data().genre_profiles.get(name)
    }

    pub fn get_genres_by_category(category: &str) -> Vec<String> {
        data().genre_profiles.iter()
            .filter(|(_, profile)| profile.category == category)
            .map(|(key, _)| key.clone())
            .collect()
    }

    pub fn get_all_categories() -> Vec<String> {
        let mut categories: Vec<String> = data().genre_profiles.values()
            .map(|p| p.category.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        categories.sort();
        categories
    }

    // Platform Targets
    pub fn get_platform_targets() -> &'static HashMap<String, PlatformTarget> {
        &data().platform_targets
    }

    pub fn get_platform_target(name: &str) -> Option<&'static PlatformTarget> {
        data().platform_targets.get(name)
    }
}
