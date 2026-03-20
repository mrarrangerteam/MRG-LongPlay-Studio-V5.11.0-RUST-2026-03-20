//! LongPlay Studio V5 - CLI Entry Point
//! Professional mastering tool with AI-driven recommendations
//! Ported from C++ main.cpp

use clap::Parser;
use std::io::Write;
use std::time::Instant;

use longplay_chain::MasterChain;
use longplay_analysis::analyzer::AudioAnalyzer;
use longplay_analysis::loudness::LoudnessMeter;

#[derive(Parser)]
#[command(
    name = "longplay-cli",
    version = "5.0.0",
    about = "LongPlay Studio V5 - Professional Audio Mastering"
)]
struct Cli {
    /// Input audio file
    input: Option<String>,

    /// Output audio file
    output: Option<String>,

    /// Genre for AI recommendations (e.g., "Pop", "EDM", "Hip Hop")
    #[arg(short = 'g', long)]
    genre: Option<String>,

    /// Platform target (spotify, apple_music, youtube, tidal, amazon, soundcloud, radio, cd)
    #[arg(short = 'p', long)]
    platform: Option<String>,

    /// Processing intensity (0-100)
    #[arg(short = 'i', long, default_value = "75")]
    intensity: u32,

    /// Use AI auto-recommendation
    #[arg(long)]
    ai: bool,

    /// Analyze input file only (no processing)
    #[arg(short = 'a', long)]
    analyze: bool,

    /// Measure LUFS only
    #[arg(long)]
    lufs: bool,

    /// Suppress progress output
    #[arg(short = 'q', long)]
    quiet: bool,

    /// Maximizer gain push (0-20 dB)
    #[arg(long)]
    gain: Option<f64>,

    /// Output ceiling (-3.0 to -0.1 dBTP, default: -1.0)
    #[arg(long)]
    ceiling: Option<f64>,

    /// Load settings from JSON file
    #[arg(long)]
    load: Option<String>,

    /// Save settings to JSON file
    #[arg(long)]
    save: Option<String>,

    /// Output format (wav or flac)
    #[arg(long, default_value = "wav")]
    format: String,

    /// Bit depth (16, 24, or 32)
    #[arg(long = "bit-depth", default_value = "24")]
    bit_depth: u32,
}

// ============================================================================
// Progress Display
// ============================================================================

struct ProgressBar {
    width: usize,
    quiet: bool,
}

impl ProgressBar {
    fn new(width: usize, quiet: bool) -> Self {
        Self { width, quiet }
    }

    fn update(&self, percent: u32, message: &str) {
        if self.quiet {
            return;
        }
        let filled = (percent as usize * self.width) / 100;
        let mut bar = String::with_capacity(self.width + 2);
        bar.push('[');
        for i in 0..self.width {
            if i < filled {
                bar.push('=');
            } else if i == filled {
                bar.push('>');
            } else {
                bar.push(' ');
            }
        }
        bar.push(']');
        eprint!("\r{} {:3}% {}", bar, percent, message);
        let _ = std::io::stderr().flush();
    }

    fn complete(&self) {
        if self.quiet {
            return;
        }
        self.update(100, "Complete!");
        eprintln!();
    }
}

// ============================================================================
// Analysis & Loudness
// ============================================================================

fn perform_analysis(input_file: &str) -> anyhow::Result<()> {
    let mut analyzer = AudioAnalyzer::new();
    let progress = ProgressBar::new(30, false);

    progress.update(25, "Analyzing file...");
    let analysis = analyzer.analyze(input_file)
        .map_err(|e| anyhow::anyhow!("Analysis failed: {}", e))?;
    progress.update(50, "Analyzing spectral content...");
    progress.complete();

    println!("\n=== Audio Analysis ===");
    println!("\nDuration: {:.2} seconds", analysis.duration_seconds);

    println!("\nSpectral Analysis:");
    println!("  Brightness: {:.1}%", analysis.spectral.brightness * 100.0);
    println!("  Spectral Centroid: {:.1} Hz", analysis.spectral.spectral_centroid);
    println!("  Sub-bass Energy: {:.1}%", analysis.spectral.sub_energy * 100.0);
    println!("  Low Frequency Energy: {:.1}%", analysis.spectral.low_energy * 100.0);
    println!("  Midrange Energy: {:.1}%", analysis.spectral.mid_energy * 100.0);
    println!("  High Frequency Energy: {:.1}%", analysis.spectral.high_energy * 100.0);

    println!("\nDynamic Analysis:");
    println!("  Peak Level: {:.2} dB", analysis.dynamics.peak_db);
    println!("  RMS Level: {:.2} dB", analysis.dynamics.rms_db);
    println!("  Crest Factor: {:.2} dB", analysis.dynamics.crest_factor_db);
    println!("  Dynamic Range: {:.2} dB", analysis.dynamics.dynamic_range_db);

    println!("\nStereo Analysis:");
    println!("  Mono: {}", if analysis.stereo.is_mono { "Yes" } else { "No" });
    println!("  Correlation: {:.2}", analysis.stereo.correlation);
    println!("  Width: {:.1}%", analysis.stereo.width_pct);
    println!("  Balance: {:.2}", analysis.stereo.balance_lr);

    Ok(())
}

fn perform_loudness_analysis(input_file: &str) -> anyhow::Result<()> {
    let meter = LoudnessMeter::new("ffmpeg");
    let progress = ProgressBar::new(30, false);

    progress.update(50, "Measuring loudness...");
    let result = meter.analyze(input_file)
        .ok_or_else(|| anyhow::anyhow!("Loudness measurement failed"))?;
    progress.complete();

    println!("\n=== Loudness Measurement ===");
    println!("Integrated LUFS: {:.2} LUFS", result.integrated_lufs);
    println!("True Peak: {:.2} dBTP", result.true_peak_dbtp);
    println!("Loudness Range: {:.2} LU", result.lra);
    println!("Duration: {:.2} seconds", result.duration_sec);
    println!("Sample Rate: {} Hz", result.sample_rate);
    println!("Channels: {}", result.channels);

    Ok(())
}

// ============================================================================
// AI Recommendation
// ============================================================================

fn print_recommendations(rec: &longplay_ai::MasterRecommendation) {
    println!("\n=== AI Recommendations ===");
    println!("Genre: {}", rec.genre);
    println!("Platform: {}", rec.platform);
    println!("Intensity: {:.0}/100", rec.intensity);
    println!("Confidence: {:.0}%", rec.confidence);

    if !rec.explanations.is_empty() {
        println!("\nExplanations:");
        for exp in &rec.explanations {
            println!("  - {}", exp);
        }
    }

    println!("\nRecommended Settings:");
    println!("  Maximizer: {} settings", rec.maximizer_settings.len());
    println!("  Equalizer: {} settings", rec.equalizer_settings.len());
    println!("  Dynamics: {} settings", rec.dynamics_settings.len());
    println!("  Imager: {} settings", rec.imager_settings.len());
}

// ============================================================================
// Main Processing
// ============================================================================

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Validate required arguments
    if !cli.analyze && !cli.lufs && cli.input.is_none() {
        anyhow::bail!("Input file is required. Use --help for usage information.");
    }

    // Analysis-only mode
    if cli.analyze {
        let input = cli.input
            .ok_or_else(|| anyhow::anyhow!("Input file is required for analysis"))?;
        return perform_analysis(&input);
    }

    // Loudness-only mode
    if cli.lufs {
        let input = cli.input
            .ok_or_else(|| anyhow::anyhow!("Input file is required for loudness measurement"))?;
        return perform_loudness_analysis(&input);
    }

    // Full mastering workflow
    let input_file = cli.input
        .ok_or_else(|| anyhow::anyhow!("Input file is required"))?;
    let output_file = cli.output
        .ok_or_else(|| anyhow::anyhow!("Output file is required"))?;

    let progress = ProgressBar::new(40, cli.quiet);
    let start_time = Instant::now();

    progress.update(5, "Initializing MasterChain...");

    // Create and configure master chain
    let mut chain = MasterChain::new();
    chain.load_audio(&input_file);

    // Set platform target (defaults to spotify if not specified)
    let platform = cli.platform.as_deref().unwrap_or("spotify");
    chain.set_platform(platform)
        .map_err(|e| anyhow::anyhow!("Platform error: {}", e))?;

    // Set intensity
    chain.set_intensity(cli.intensity as f32)
        .map_err(|e| anyhow::anyhow!("Intensity error: {}", e))?;

    // Load settings from file if specified
    if let Some(ref load_path) = cli.load {
        progress.update(8, "Loading settings...");
        if !chain.load_settings(load_path) {
            eprintln!("Warning: Could not load settings from {}", load_path);
        }
    }

    // Apply manual ceiling override
    if let Some(ceiling) = cli.ceiling {
        chain.set_target_tp(ceiling as f32);
    }

    progress.update(15, "Analyzing input...");

    // AI recommendation if requested
    let mut recommendation = None;
    if cli.ai {
        progress.update(20, "AI recommendation...");

        let genre = cli.genre.as_deref().unwrap_or("pop");
        match chain.ai_recommend(genre, platform, cli.intensity as f32) {
            Ok(rec) => {
                chain.apply_recommendation(&rec);
                if !cli.quiet {
                    print_recommendations(&rec);
                }
                recommendation = Some(rec);
            }
            Err(e) => {
                eprintln!("Warning: AI recommendation failed: {}", e);
            }
        }
    }

    // Apply manual gain override to maximizer if specified
    if let Some(gain) = cli.gain {
        let mut max_override = longplay_core::Settings::new();
        max_override.insert(
            "gain_db".to_string(),
            longplay_core::SettingsValue::Float32(gain as f32),
        );
        max_override.insert(
            "enabled".to_string(),
            longplay_core::SettingsValue::Bool(true),
        );
        chain.get_maximizer_mut().from_settings(&max_override);
    }

    progress.update(30, "Processing audio...");

    // Render through the complete mastering chain with progress callback
    let render_ok = chain.render(
        &output_file,
        Some(Box::new(move |pct: f32, stage: &str| {
            // Map render progress (0-1) to display range (30-95%)
            let display_pct = 30 + (pct * 65.0) as u32;
            if !cli.quiet {
                let filled = (display_pct as usize * 40) / 100;
                let mut bar = String::with_capacity(42);
                bar.push('[');
                for i in 0..40 {
                    if i < filled {
                        bar.push('=');
                    } else if i == filled {
                        bar.push('>');
                    } else {
                        bar.push(' ');
                    }
                }
                bar.push(']');
                eprint!("\r{} {:3}% {}", bar, display_pct, stage);
                let _ = std::io::stderr().flush();
            }
        })),
    );

    if !render_ok {
        anyhow::bail!("Render failed");
    }

    progress.update(95, "Finalizing...");

    // Save settings if requested
    if let Some(ref save_path) = cli.save {
        chain.save_settings(save_path);
    }

    let duration = start_time.elapsed();

    if !cli.quiet {
        // Print completion
        progress.complete();
    }

    // Print summary with before/after comparison
    println!("\n=== Mastering Summary ===");

    let ab = chain.get_ab_comparison();

    println!("\nLoudness (Before -> After):");
    println!(
        "  LUFS: {:.2} -> {:.2} ({:+.2} LU)",
        ab.before_lufs, ab.after_lufs, ab.lufs_change
    );
    println!(
        "  True Peak: {:.2} -> {:.2} dBTP",
        ab.before_tp, ab.after_tp
    );

    println!("\nOutput Settings:");
    println!("  Format: {}", cli.format);
    println!("  Bit Depth: {}-bit", cli.bit_depth);
    println!("  Platform: {}", platform);
    println!("  Target LUFS: {:.1}", chain.get_target_lufs());
    println!("  Target TP: {:.1} dBTP", chain.get_target_tp());

    if let Some(ref rec) = recommendation {
        println!("\nAI Processing:");
        println!("  Genre: {}", rec.genre);
        println!("  Confidence: {:.0}%", rec.confidence);
        println!("  Intensity: {:.0}/100", rec.intensity);
    }

    println!("\nProcessing Time: {:.2} seconds", duration.as_secs_f64());
    println!("Output: {}", output_file);

    if let Some(ref save_path) = cli.save {
        println!("Settings saved to: {}", save_path);
    }

    Ok(())
}
