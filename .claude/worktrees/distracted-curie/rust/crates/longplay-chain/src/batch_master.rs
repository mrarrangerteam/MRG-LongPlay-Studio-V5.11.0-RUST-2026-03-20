//! BatchMasterLite — Parallel batch mastering for Content Factory
//!
//! Masters N songs in parallel using rayon, applying ONLY:
//!   - Dynamics (compression) — even out dynamic range
//!   - Imager (stereo width) — consistent stereo image
//!   - Maximizer (loudness push + ceiling)
//!   - Loudness normalization (RMS-based LUFS approximation)
//!   - True peak hard clip
//!
//! Does NOT apply EQ — different songs have different tonal balance.
//! Applying one EQ curve to many songs would destroy their character.

use longplay_core::AudioBuffer;
use longplay_core::conversions::db_to_linear_f;
use longplay_dsp::dynamics::Dynamics;
use longplay_dsp::imager::Imager;
use longplay_dsp::maximizer::Maximizer;

use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

/// Configuration for batch mastering
#[derive(Debug, Clone)]
pub struct BatchMasterConfig {
    pub dynamics_enabled: bool,
    pub imager_enabled: bool,
    pub maximizer_enabled: bool,
    pub target_lufs: f32,
    pub true_peak_limit: f32,
    pub dynamics_threshold: f64,
    pub dynamics_ratio: f64,
    pub imager_width: f32,
    pub maximizer_ceiling: f32,
}

impl Default for BatchMasterConfig {
    fn default() -> Self {
        Self {
            dynamics_enabled: true,
            imager_enabled: true,
            maximizer_enabled: true,
            target_lufs: -14.0,
            true_peak_limit: -1.0,
            dynamics_threshold: -18.0,
            dynamics_ratio: 2.5,
            imager_width: 100.0,   // 100 = original width (0-200 scale)
            maximizer_ceiling: -0.3,
        }
    }
}

/// Result of mastering a single song
#[derive(Debug, Clone)]
pub struct BatchMasterResult {
    pub input_path: String,
    pub output_path: String,
    pub success: bool,
    pub error: Option<String>,
    pub duration_sec: f64,
}

/// Progress callback type for batch mastering
/// (completed_count, total_count, current_file)
pub type BatchProgressCallback = Box<dyn Fn(usize, usize, &str) + Send + Sync>;

/// Master a single audio buffer through the lite pipeline (no EQ).
///
/// Signal flow:
///   1. Dynamics (compression)
///   2. Imager (stereo width)
///   3. Maximizer (gain push + ceiling)
///   4. Loudness normalization (RMS → target LUFS)
///   5. True peak hard clip
fn process_buffer_lite(
    buffer: &AudioBuffer,
    sample_rate: u32,
    config: &BatchMasterConfig,
) -> AudioBuffer {
    if buffer.is_empty() || buffer[0].is_empty() {
        return buffer.clone();
    }

    let mut buf = buffer.clone();
    let sr = sample_rate as i32;

    // Stage 1: Dynamics (compression)
    if config.dynamics_enabled {
        let mut dynamics = Dynamics::new();
        dynamics.set_bypass(false);
        dynamics.set_multiband(false);
        dynamics.single_band_mut().set_threshold(config.dynamics_threshold);
        dynamics.single_band_mut().set_ratio(config.dynamics_ratio);
        dynamics.single_band_mut().set_attack(10.0);   // ms
        dynamics.single_band_mut().set_release(100.0);  // ms
        buf = dynamics.process(&buf, sr);
    }

    // Stage 2: Imager (stereo width)
    if config.imager_enabled && buf.len() >= 2 {
        let mut imager = Imager::new();
        imager.set_bypass(false);
        imager.set_width(config.imager_width);
        imager.process(&mut buf, sr);
    }

    // Stage 3: Maximizer (gain push + ceiling)
    if config.maximizer_enabled {
        let mut maximizer = Maximizer::new();
        maximizer.set_bypass(false);
        maximizer.set_ceiling(config.maximizer_ceiling);
        maximizer.process(&mut buf, sr);
    }

    // Stage 4: Loudness normalization (RMS-based LUFS approximation)
    let loudness_gain = estimate_rms_gain(&buf, config.target_lufs);
    for channel in &mut buf {
        for sample in channel.iter_mut() {
            *sample *= loudness_gain;
        }
    }

    // Stage 5: True peak hard clip
    let ceiling_linear = db_to_linear_f(config.true_peak_limit);
    for channel in &mut buf {
        for sample in channel.iter_mut() {
            if *sample > ceiling_linear {
                *sample = ceiling_linear;
            } else if *sample < -ceiling_linear {
                *sample = -ceiling_linear;
            }
        }
    }

    buf
}

/// RMS-based gain estimation to approximate target LUFS.
fn estimate_rms_gain(buffer: &AudioBuffer, target_lufs: f32) -> f32 {
    if buffer.is_empty() || buffer[0].is_empty() {
        return 1.0;
    }

    let mut sum_sq = 0.0f64;
    let mut total_samples = 0usize;

    for channel in buffer {
        for &sample in channel {
            sum_sq += (sample as f64) * (sample as f64);
        }
        total_samples += channel.len();
    }

    if total_samples == 0 {
        return 1.0;
    }

    let rms = (sum_sq / total_samples as f64).sqrt();
    if rms <= 1e-10 {
        return 1.0;
    }

    let rms_db = 20.0 * rms.log10();
    let gain_db = (target_lufs as f64) - rms_db;
    let gain_db = gain_db.clamp(-20.0, 20.0);

    10.0_f64.powf(gain_db / 20.0) as f32
}

/// Master a single file: read → process → write.
fn master_single_file(
    input_path: &str,
    output_path: &str,
    config: &BatchMasterConfig,
) -> BatchMasterResult {
    // Read input (supports WAV, MP3, FLAC, OGG via longplay_io)
    let (buffer, info) = match longplay_io::read_audio_any(input_path) {
        Ok(result) => result,
        Err(e) => {
            return BatchMasterResult {
                input_path: input_path.to_string(),
                output_path: output_path.to_string(),
                success: false,
                error: Some(format!("Read error: {}", e)),
                duration_sec: 0.0,
            };
        }
    };

    // Ensure stereo
    let buffer = if buffer.len() == 1 {
        longplay_io::mono_to_stereo(&buffer)
    } else {
        buffer
    };

    // Process through lite pipeline
    let processed = process_buffer_lite(&buffer, info.sample_rate as u32, config);

    // Write output as 24-bit WAV
    let bit_depth = if info.bit_depth > 0 && info.bit_depth <= 32 {
        info.bit_depth
    } else {
        24
    };

    match longplay_io::write_audio(output_path, &processed, info.sample_rate, bit_depth) {
        Ok(_) => BatchMasterResult {
            input_path: input_path.to_string(),
            output_path: output_path.to_string(),
            success: true,
            error: None,
            duration_sec: info.duration_seconds,
        },
        Err(e) => BatchMasterResult {
            input_path: input_path.to_string(),
            output_path: output_path.to_string(),
            success: false,
            error: Some(format!("Write error: {}", e)),
            duration_sec: info.duration_seconds,
        },
    }
}

/// Batch master multiple songs in parallel using rayon.
///
/// # Arguments
/// * `input_paths` — List of input audio file paths
/// * `output_paths` — Corresponding output file paths (must be same length)
/// * `config` — Batch mastering configuration
/// * `cancelled` — Atomic flag to signal cancellation
/// * `progress_callback` — Optional progress callback (completed, total, current_file)
///
/// # Returns
/// Vector of results, one per input file.
pub fn batch_master(
    input_paths: &[String],
    output_paths: &[String],
    config: &BatchMasterConfig,
    cancelled: Option<Arc<AtomicBool>>,
    progress_callback: Option<&BatchProgressCallback>,
) -> Vec<BatchMasterResult> {
    assert_eq!(
        input_paths.len(),
        output_paths.len(),
        "Input and output path counts must match"
    );

    let total = input_paths.len();
    let completed = Arc::new(AtomicUsize::new(0));
    let cancel_flag = cancelled.unwrap_or_else(|| Arc::new(AtomicBool::new(false)));

    // Pair input/output paths for parallel iteration
    let pairs: Vec<(&String, &String)> = input_paths.iter().zip(output_paths.iter()).collect();

    let results: Vec<BatchMasterResult> = pairs
        .par_iter()
        .map(|(input, output)| {
            // Check cancellation
            if cancel_flag.load(Ordering::Relaxed) {
                return BatchMasterResult {
                    input_path: input.to_string(),
                    output_path: output.to_string(),
                    success: false,
                    error: Some("Cancelled".to_string()),
                    duration_sec: 0.0,
                };
            }

            let result = master_single_file(input, output, config);

            // Update progress
            let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
            if let Some(cb) = progress_callback {
                cb(done, total, input);
            }

            result
        })
        .collect();

    results
}

/// Get the number of available CPU threads for parallel processing
pub fn available_parallelism() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}
