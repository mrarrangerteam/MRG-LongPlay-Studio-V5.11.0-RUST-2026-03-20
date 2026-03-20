use longplay_core::Settings;
use longplay_analysis::analyzer::AudioAnalyzer;
use longplay_analysis::loudness::LoudnessMeter;
use longplay_profiles::GenreProfiles;

/// Master recommendation containing genre-specific settings for all processing modules
#[derive(Debug, Clone)]
pub struct MasterRecommendation {
    pub genre: String,
    pub intensity: f32,
    pub platform: String,
    pub confidence: f32,
    pub explanations: Vec<String>,
    pub maximizer_settings: Settings,
    pub equalizer_settings: Settings,
    pub dynamics_settings: Settings,
    pub imager_settings: Settings,
}

/// AI recommendation engine
pub struct AIAssist {
    analyzer: AudioAnalyzer,
    loudness_meter: LoudnessMeter,
}

impl AIAssist {
    pub fn new() -> Self {
        Self {
            analyzer: AudioAnalyzer::new(),
            loudness_meter: LoudnessMeter::new("ffmpeg"),
        }
    }

    /// Analyze audio file and generate recommendations
    pub fn analyze_and_recommend(
        &mut self,
        file_path: &str,
        genre: &str,
        platform: &str,
        intensity: f32,
    ) -> Result<MasterRecommendation, String> {
        let intensity = intensity.clamp(0.0, 100.0);

        // Step 1: Audio analysis
        let analysis = self.analyzer.analyze(file_path)
            .map_err(|e| format!("Analysis failed: {}", e))?;

        // Step 2: Loudness measurement
        let loudness_opt = self.loudness_meter.quick_measure(file_path);
        let current_lufs = loudness_opt.map(|(lufs, _)| lufs).unwrap_or(-24.0);
        let loudness_available = loudness_opt.is_some();

        // Step 3: Genre profile
        let genre_profile = GenreProfiles::get_genre_profile(genre)
            .ok_or_else(|| format!("Genre not found: {}", genre))?;

        // Step 4: Platform target
        let platform_target = GenreProfiles::get_platform_target(platform);
        let plat_target_lufs = platform_target.map(|p| p.target_lufs).unwrap_or(-14.0) as f32;
        let plat_max_tp = platform_target.map(|p| p.max_true_peak).unwrap_or(-1.0) as f32;

        // Generate per-module recommendations
        let irc_mode = if genre_profile.maximizer.irc_mode.is_empty() {
            "A".to_string()
        } else {
            genre_profile.maximizer.irc_mode.clone()
        };

        let maximizer_settings = crate::generate::generate_maximizer_recommendations(
            plat_target_lufs, current_lufs, plat_max_tp, &irc_mode, intensity,
        );
        let equalizer_settings = crate::generate::generate_equalizer_recommendations(
            analysis.spectral.brightness, analysis.spectral.low_energy, intensity,
        );
        let dynamics_settings = crate::generate::generate_dynamics_recommendations(
            analysis.dynamics.crest_factor_db, intensity,
        );
        let imager_settings = crate::generate::generate_imager_recommendations(
            analysis.stereo.correlation, analysis.stereo.is_mono, analysis.stereo.width_pct,
        );

        let confidence = crate::generate::calculate_confidence(
            analysis.duration_seconds, loudness_available, analysis.spectral.spectral_centroid,
        );

        // Build explanations
        let mut explanations = Vec::new();
        if analysis.spectral.brightness > 0.4 { explanations.push("De-harsh high frequencies".into()); }
        if analysis.spectral.brightness < 0.15 { explanations.push("Add presence".into()); }
        if analysis.spectral.low_energy > 0.35 { explanations.push("Reduce muddiness".into()); }
        if analysis.dynamics.crest_factor_db < 6.0 { explanations.push("Gentle glue".into()); }
        if analysis.dynamics.crest_factor_db > 15.0 { explanations.push("Tame dynamics".into()); }
        if analysis.stereo.correlation > 0.95 { explanations.push("Widen stereo".into()); }
        if analysis.stereo.correlation < 0.3 { explanations.push("Tighten stereo".into()); }
        if !analysis.stereo.is_mono && analysis.stereo.width_pct < 50.0 {
            explanations.push("Mono bass for clarity".into());
        }

        Ok(MasterRecommendation {
            genre: genre.to_string(),
            intensity,
            platform: platform.to_string(),
            confidence,
            explanations,
            maximizer_settings,
            equalizer_settings,
            dynamics_settings,
            imager_settings,
        })
    }
}
