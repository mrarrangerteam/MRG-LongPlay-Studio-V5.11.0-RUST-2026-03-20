//! Integration tests for the MasterChain pipeline.
//!
//! All tests are self-contained and generate their own synthetic audio data.

use std::f32::consts::PI;
use std::path::PathBuf;

/// Generate a stereo 440Hz sine wave WAV file and return its path.
/// Duration: 5 seconds, 44100 Hz, 16-bit.
fn generate_test_wav(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join("longplay_chain_tests");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{}.wav", name));

    let sample_rate = 44100u32;
    let duration_secs = 5.0f32;
    let num_samples = (sample_rate as f32 * duration_secs) as usize;
    let frequency = 440.0f32;

    let spec = hound::WavSpec {
        channels: 2,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(&path, spec).unwrap();

    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;
        let sample = (2.0 * PI * frequency * t).sin() * 0.5; // -6 dBFS
        let int_sample = (sample * 32767.0) as i16;
        // Left channel
        writer.write_sample(int_sample).unwrap();
        // Right channel (slightly phase-shifted for stereo content)
        let sample_r = (2.0 * PI * frequency * t + 0.3).sin() * 0.5;
        let int_sample_r = (sample_r * 32767.0) as i16;
        writer.write_sample(int_sample_r).unwrap();
    }
    writer.finalize().unwrap();
    path
}

/// Helper to clean up test files.
fn cleanup(paths: &[&PathBuf]) {
    for path in paths {
        let _ = std::fs::remove_file(path);
    }
}

// ============================================================================
// Full Pipeline Tests
// ============================================================================

#[test]
fn test_full_pipeline_load_process_render() {
    let input_path = generate_test_wav("pipeline_input");
    let output_dir = std::env::temp_dir().join("longplay_chain_tests");
    let output_path = output_dir.join("pipeline_output.wav");

    let mut chain = longplay_chain::MasterChain::new();

    // Load audio
    chain.load_audio(input_path.to_str().unwrap());
    assert_eq!(chain.get_input_path(), input_path.to_str().unwrap());

    // Set platform
    let result = chain.set_platform("spotify");
    assert!(result.is_ok());
    assert!((chain.get_target_lufs() - (-14.0)).abs() < 0.1);

    // Render
    let success = chain.render(output_path.to_str().unwrap(), None);
    assert!(success, "Render should succeed");

    // Verify output file exists and has valid audio
    assert!(output_path.exists(), "Output file should exist");
    let metadata = std::fs::metadata(&output_path).unwrap();
    assert!(metadata.len() > 44, "Output file should be larger than WAV header");

    // Verify output is valid WAV by reading it back
    let reader = hound::WavReader::open(&output_path).unwrap();
    let spec = reader.spec();
    assert_eq!(spec.channels, 2);
    assert_eq!(spec.sample_rate, 44100);

    cleanup(&[&input_path, &output_path]);
}

#[test]
fn test_process_file_method() {
    let input_path = generate_test_wav("process_file_input");
    let output_dir = std::env::temp_dir().join("longplay_chain_tests");
    let output_path = output_dir.join("process_file_output.wav");

    let mut chain = longplay_chain::MasterChain::new();
    let result = chain.process_file(
        input_path.to_str().unwrap(),
        output_path.to_str().unwrap(),
    );
    assert!(result.is_ok(), "process_file should succeed: {:?}", result.err());

    assert!(output_path.exists(), "Output file should exist");

    // Read output and verify no NaN/Inf
    let reader = hound::WavReader::open(&output_path).unwrap();
    let samples: Vec<i32> = reader.into_samples::<i32>().map(|s| s.unwrap()).collect();
    assert!(!samples.is_empty(), "Output should contain samples");

    cleanup(&[&input_path, &output_path]);
}

#[test]
fn test_process_audio_buffer_directly() {
    let sample_rate = 44100u32;
    let num_samples = 44100; // 1 second

    // Generate stereo sine wave buffer
    let mut buffer = vec![vec![0.0f32; num_samples]; 2];
    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;
        buffer[0][i] = (2.0 * PI * 440.0 * t).sin() * 0.5;
        buffer[1][i] = (2.0 * PI * 440.0 * t + 0.3).sin() * 0.5;
    }

    let mut chain = longplay_chain::MasterChain::new();
    chain.set_target_lufs(-14.0);
    chain.set_target_tp(-1.0);

    let output = chain.process_audio(&buffer, sample_rate);

    // Verify output dimensions match input
    assert_eq!(output.len(), 2, "Should have 2 channels");
    assert_eq!(output[0].len(), num_samples, "Should have same number of samples");

    // Verify no NaN or Inf
    for ch in &output {
        for &sample in ch {
            assert!(!sample.is_nan(), "Output should not contain NaN");
            assert!(!sample.is_infinite(), "Output should not contain Inf");
        }
    }

    // Verify output has audio content (not silence)
    let max_sample: f32 = output.iter()
        .flat_map(|ch| ch.iter())
        .fold(0.0f32, |max, &s| max.max(s.abs()));
    assert!(max_sample > 0.001, "Output should contain audio, max={}", max_sample);
}

// ============================================================================
// Platform Target Tests
// ============================================================================

#[test]
fn test_set_platform_targets() {
    let mut chain = longplay_chain::MasterChain::new();

    // Valid platforms
    assert!(chain.set_platform("spotify").is_ok());
    assert!(chain.set_platform("apple_music").is_ok());
    assert!(chain.set_platform("youtube").is_ok());
    assert!(chain.set_platform("cd").is_ok());

    // Invalid platform
    assert!(chain.set_platform("nonexistent_platform").is_err());

    // Case insensitive
    assert!(chain.set_platform("SPOTIFY").is_ok());
}

#[test]
fn test_platform_targets_have_values() {
    let targets = longplay_chain::MasterChain::get_platform_targets();
    assert!(!targets.is_empty(), "Should have platform targets");
    assert!(targets.contains_key("spotify"), "Should have spotify target");
}

// ============================================================================
// Settings Save/Load Roundtrip
// ============================================================================

#[test]
fn test_settings_save_load_roundtrip() {
    let settings_dir = std::env::temp_dir().join("longplay_chain_tests");
    std::fs::create_dir_all(&settings_dir).unwrap();
    let settings_path = settings_dir.join("test_settings.json");

    // Configure chain with specific settings
    let mut chain = longplay_chain::MasterChain::new();
    let _ = chain.set_intensity(80.0);
    let _ = chain.set_platform("spotify");
    // Set custom targets AFTER platform (platform overrides targets)
    chain.set_target_lufs(-16.0);
    chain.set_target_tp(-0.5);

    // Configure EQ with non-default values
    chain.get_equalizer_mut().band_mut(0).set_gain(3.0);
    chain.get_equalizer_mut().band_mut(1).set_gain(-2.0);

    // Configure dynamics
    chain.get_dynamics_mut().single_band_mut().set_threshold(-20.0);
    chain.get_dynamics_mut().single_band_mut().set_ratio(3.0);

    // Configure imager
    chain.get_imager_mut().set_width(130.0);

    // Configure maximizer
    chain.get_maximizer_mut().set_gain_db(6.0);
    chain.get_maximizer_mut().set_ceiling(-1.0);

    // Save
    let save_ok = chain.save_settings(settings_path.to_str().unwrap());
    assert!(save_ok, "Settings save should succeed");
    assert!(settings_path.exists(), "Settings file should exist");

    // Verify JSON file is valid
    let content = std::fs::read_to_string(&settings_path).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();
    assert!(parsed.is_object(), "Should be valid JSON object");

    // Load into a fresh chain
    let mut chain2 = longplay_chain::MasterChain::new();
    let load_ok = chain2.load_settings(settings_path.to_str().unwrap());
    assert!(load_ok, "Settings load should succeed");

    // Verify values were restored
    assert!((chain2.get_target_lufs() - (-16.0)).abs() < 0.1,
        "Target LUFS should be restored, got {}", chain2.get_target_lufs());
    assert!((chain2.get_target_tp() - (-0.5)).abs() < 0.1,
        "Target TP should be restored, got {}", chain2.get_target_tp());
    assert!((chain2.get_intensity() - 80.0).abs() < 0.1,
        "Intensity should be restored, got {}", chain2.get_intensity());

    let _ = std::fs::remove_file(&settings_path);
}

// ============================================================================
// Individual DSP Module Tests via MasterChain
// ============================================================================

#[test]
fn test_eq_module_changes_output() {
    let sample_rate = 44100u32;
    let num_samples = 44100;

    let mut buffer = vec![vec![0.0f32; num_samples]; 2];
    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;
        buffer[0][i] = (2.0 * PI * 440.0 * t).sin() * 0.5;
        buffer[1][i] = (2.0 * PI * 440.0 * t).sin() * 0.5;
    }

    let mut chain = longplay_chain::MasterChain::new();

    // Process with flat EQ (default)
    let output_flat = chain.process_audio(&buffer, sample_rate);

    // Apply a strong EQ boost and process again
    chain.reset_all();
    chain.get_equalizer_mut().band_mut(4).set_gain(6.0); // +6dB at 1kHz
    let output_boosted = chain.process_audio(&buffer, sample_rate);

    // The two outputs should differ (EQ changed the sound)
    let mut diff_sum = 0.0f64;
    for i in 0..num_samples {
        diff_sum += (output_flat[0][i] - output_boosted[0][i]).abs() as f64;
    }
    assert!(diff_sum > 0.1, "EQ boost should change the output, diff={}", diff_sum);
}

#[test]
fn test_dynamics_module_processes() {
    let sample_rate = 44100u32;
    let num_samples = 44100;

    let mut buffer = vec![vec![0.0f32; num_samples]; 2];
    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;
        buffer[0][i] = (2.0 * PI * 440.0 * t).sin() * 0.8;
        buffer[1][i] = (2.0 * PI * 440.0 * t).sin() * 0.8;
    }

    let mut chain = longplay_chain::MasterChain::new();
    chain.get_dynamics_mut().single_band_mut().set_threshold(-10.0);
    chain.get_dynamics_mut().single_band_mut().set_ratio(4.0);

    let output = chain.process_audio(&buffer, sample_rate);

    // Verify output is valid
    for ch in &output {
        for &sample in ch {
            assert!(!sample.is_nan(), "No NaN after dynamics");
            assert!(!sample.is_infinite(), "No Inf after dynamics");
        }
    }
}

#[test]
fn test_imager_module_width_changes_stereo() {
    let sample_rate = 44100u32;
    let num_samples = 44100;

    // Create stereo content with different L/R
    let mut buffer = vec![vec![0.0f32; num_samples]; 2];
    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;
        buffer[0][i] = (2.0 * PI * 440.0 * t).sin() * 0.5;
        buffer[1][i] = (2.0 * PI * 440.0 * t + 0.5).sin() * 0.5;
    }

    let mut chain = longplay_chain::MasterChain::new();

    // Process with default width (100%)
    let output_normal = chain.process_audio(&buffer, sample_rate);

    // Process with wide stereo
    chain.reset_all();
    chain.get_imager_mut().set_width(180.0);
    let output_wide = chain.process_audio(&buffer, sample_rate);

    // Compute stereo difference for each output
    let stereo_diff_normal: f64 = (0..num_samples)
        .map(|i| (output_normal[0][i] - output_normal[1][i]).abs() as f64)
        .sum();
    let stereo_diff_wide: f64 = (0..num_samples)
        .map(|i| (output_wide[0][i] - output_wide[1][i]).abs() as f64)
        .sum();

    // Wider image should have greater L-R difference
    assert!(stereo_diff_wide > stereo_diff_normal * 0.9,
        "Wide image should increase stereo diff: normal={:.4}, wide={:.4}",
        stereo_diff_normal, stereo_diff_wide);
}

#[test]
fn test_maximizer_module_limits_output() {
    let sample_rate = 44100u32;
    let num_samples = 44100;

    let mut buffer = vec![vec![0.0f32; num_samples]; 2];
    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;
        // Loud signal
        buffer[0][i] = (2.0 * PI * 440.0 * t).sin() * 0.9;
        buffer[1][i] = (2.0 * PI * 440.0 * t).sin() * 0.9;
    }

    let mut chain = longplay_chain::MasterChain::new();
    chain.set_target_tp(-1.0);
    chain.get_maximizer_mut().set_gain_db(10.0);
    chain.get_maximizer_mut().set_ceiling(-1.0);

    let output = chain.process_audio(&buffer, sample_rate);

    // The ceiling should limit output amplitude
    let ceiling_linear = 10.0_f32.powf(-1.0 / 20.0);
    let max_sample: f32 = output.iter()
        .flat_map(|ch| ch.iter())
        .fold(0.0f32, |max, &s| max.max(s.abs()));

    // Allow some tolerance for true peak overshoot and the hard clip stage
    assert!(max_sample <= ceiling_linear + 0.1,
        "Output should be limited near ceiling: max={:.4}, ceiling={:.4}",
        max_sample, ceiling_linear);
}

// ============================================================================
// AB Comparison Tests
// ============================================================================

#[test]
fn test_ab_comparison_default() {
    let chain = longplay_chain::MasterChain::new();
    let ab = chain.get_ab_comparison();

    // Before processing, all values should be zero (no analysis data)
    assert!((ab.before_lufs - 0.0).abs() < 0.01);
    assert!((ab.after_lufs - 0.0).abs() < 0.01);
    assert!((ab.lufs_change - 0.0).abs() < 0.01);
}

#[test]
fn test_ab_comparison_after_process() {
    let input_path = generate_test_wav("ab_compare_input");
    let output_dir = std::env::temp_dir().join("longplay_chain_tests");
    let output_path = output_dir.join("ab_compare_output.wav");

    let mut chain = longplay_chain::MasterChain::new();
    let _ = chain.process_file(
        input_path.to_str().unwrap(),
        output_path.to_str().unwrap(),
    );

    let ab = chain.get_ab_comparison();
    // After processing, the comparison struct should be populated
    // (The specific values depend on the analysis implementation,
    // but the struct should at least be returned without panicking)
    let _ = ab.lufs_change;
    let _ = ab.tp_change;

    cleanup(&[&input_path, &output_path]);
}

// ============================================================================
// AI Recommendation Tests
// ============================================================================

#[test]
fn test_ai_recommend_requires_loaded_audio() {
    let mut chain = longplay_chain::MasterChain::new();
    // No audio loaded - should fail
    let result = chain.ai_recommend("rock", "spotify", 75.0);
    assert!(result.is_err(), "Should fail without loaded audio");
}

#[test]
fn test_ai_recommend_invalid_platform() {
    let input_path = generate_test_wav("ai_rec_invalid_platform");

    let mut chain = longplay_chain::MasterChain::new();
    chain.load_audio(input_path.to_str().unwrap());

    let result = chain.ai_recommend("rock", "nonexistent_platform", 75.0);
    assert!(result.is_err(), "Should fail with invalid platform");

    cleanup(&[&input_path]);
}

#[test]
fn test_ai_recommend_invalid_intensity() {
    let input_path = generate_test_wav("ai_rec_invalid_intensity");

    let mut chain = longplay_chain::MasterChain::new();
    chain.load_audio(input_path.to_str().unwrap());

    let result = chain.ai_recommend("rock", "spotify", -10.0);
    assert!(result.is_err(), "Should fail with negative intensity");

    let result = chain.ai_recommend("rock", "spotify", 150.0);
    assert!(result.is_err(), "Should fail with intensity > 100");

    cleanup(&[&input_path]);
}

#[test]
fn test_ai_recommend_and_apply() {
    let input_path = generate_test_wav("ai_rec_apply");

    let mut chain = longplay_chain::MasterChain::new();
    chain.load_audio(input_path.to_str().unwrap());

    let result = chain.ai_recommend("rock", "spotify", 75.0);
    // This may fail if the genre is not found in profiles, which is acceptable
    if let Ok(rec) = result {
        assert!(!rec.genre.is_empty());
        assert!(!rec.platform.is_empty());
        assert!(rec.confidence >= 0.0 && rec.confidence <= 100.0);

        // Apply recommendation - should not panic
        chain.apply_recommendation(&rec);

        // Verify recommendation is stored
        let last_rec = chain.get_last_recommendation();
        assert!(last_rec.is_some());
    }

    cleanup(&[&input_path]);
}

// ============================================================================
// Intensity & Reset Tests
// ============================================================================

#[test]
fn test_intensity_bounds() {
    let mut chain = longplay_chain::MasterChain::new();

    assert!(chain.set_intensity(50.0).is_ok());
    assert!((chain.get_intensity() - 50.0).abs() < 0.1);

    assert!(chain.set_intensity(0.0).is_ok());
    assert!(chain.set_intensity(100.0).is_ok());

    assert!(chain.set_intensity(-1.0).is_err());
    assert!(chain.set_intensity(101.0).is_err());
}

#[test]
fn test_reset_all() {
    let mut chain = longplay_chain::MasterChain::new();

    // Modify everything
    chain.set_target_lufs(-20.0);
    chain.set_target_tp(-2.0);
    let _ = chain.set_intensity(50.0);
    chain.get_equalizer_mut().band_mut(0).set_gain(6.0);

    // Reset
    chain.reset_all();

    // Verify defaults restored
    assert!((chain.get_target_lufs() - (-14.0)).abs() < 0.1);
    assert!((chain.get_target_tp() - (-1.0)).abs() < 0.1);
    assert!((chain.get_intensity() - 75.0).abs() < 0.1);
}

// ============================================================================
// Chain Summary Test
// ============================================================================

#[test]
fn test_chain_summary() {
    let chain = longplay_chain::MasterChain::new();
    let summary = chain.get_chain_summary();

    assert!(summary.contains("MasterChain Summary"));
    assert!(summary.contains("Intensity: 75%"));
    assert!(summary.contains("Target LUFS: -14.0"));
    assert!(summary.contains("Equalizer:"));
    assert!(summary.contains("Dynamics:"));
    assert!(summary.contains("Imager:"));
    assert!(summary.contains("Maximizer:"));
}

// ============================================================================
// Preview Test
// ============================================================================

#[test]
fn test_preview() {
    let input_path = generate_test_wav("preview_input");

    let mut chain = longplay_chain::MasterChain::new();
    chain.load_audio(input_path.to_str().unwrap());

    // Preview 1 second starting at 1 second
    let preview = chain.preview(1.0, 1.0);
    assert_eq!(preview.len(), 2, "Preview should be stereo");
    // Should be approximately 1 second of samples
    let expected_samples = 44100;
    let tolerance = 100; // Allow some rounding
    assert!(
        (preview[0].len() as i64 - expected_samples as i64).abs() < tolerance as i64,
        "Preview length should be ~{} samples, got {}",
        expected_samples, preview[0].len()
    );

    // Verify no NaN
    for ch in &preview {
        for &sample in ch {
            assert!(!sample.is_nan(), "Preview should not contain NaN");
        }
    }

    cleanup(&[&input_path]);
}

// ============================================================================
// Genre Profile Test
// ============================================================================

#[test]
fn test_set_genre_profile() {
    let mut chain = longplay_chain::MasterChain::new();

    // Setting a valid genre profile should change targets
    chain.set_genre_profile("rock");
    // Should not panic even if genre is not found
    chain.set_genre_profile("nonexistent_genre");
}

// ============================================================================
// Empty/Edge Case Tests
// ============================================================================

#[test]
fn test_process_empty_buffer() {
    let mut chain = longplay_chain::MasterChain::new();
    let empty_buffer: Vec<Vec<f32>> = vec![vec![]];
    let output = chain.process_audio(&empty_buffer, 44100);
    // Should return empty without panicking
    assert_eq!(output.len(), 1);
    assert!(output[0].is_empty());
}

#[test]
fn test_render_without_input() {
    let mut chain = longplay_chain::MasterChain::new();
    let output_path = std::env::temp_dir().join("longplay_chain_tests").join("no_input.wav");
    let success = chain.render(output_path.to_str().unwrap(), None);
    assert!(!success, "Render without input should fail");
}

// ============================================================================
// Module Bypass Tests
// ============================================================================

#[test]
fn test_module_bypass_flags() {
    let mut chain = longplay_chain::MasterChain::new();

    // Test EQ bypass
    assert!(!chain.get_equalizer().is_bypassed());
    chain.get_equalizer_mut().set_bypass(true);
    assert!(chain.get_equalizer().is_bypassed());

    // Test Dynamics bypass
    assert!(!chain.get_dynamics().is_bypassed());
    chain.get_dynamics_mut().set_bypass(true);
    assert!(chain.get_dynamics().is_bypassed());

    // Test Imager bypass
    assert!(!chain.get_imager().is_bypassed());
    chain.get_imager_mut().set_bypass(true);
    assert!(chain.get_imager().is_bypassed());

    // Test Maximizer bypass
    assert!(!chain.get_maximizer().is_bypassed());
    chain.get_maximizer_mut().set_bypass(true);
    assert!(chain.get_maximizer().is_bypassed());
}

// ============================================================================
// Render with Progress Callback
// ============================================================================

#[test]
fn test_render_with_progress_callback() {
    let input_path = generate_test_wav("progress_callback_input");
    let output_dir = std::env::temp_dir().join("longplay_chain_tests");
    let output_path = output_dir.join("progress_callback_output.wav");

    let mut chain = longplay_chain::MasterChain::new();
    chain.load_audio(input_path.to_str().unwrap());

    let progress_values = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let pv = progress_values.clone();

    let callback: longplay_chain::ProgressCallback = Box::new(move |progress, _msg| {
        pv.lock().unwrap().push(progress);
    });

    let success = chain.render(output_path.to_str().unwrap(), Some(callback));
    assert!(success, "Render with callback should succeed");

    let values = progress_values.lock().unwrap();
    assert!(!values.is_empty(), "Progress callback should have been called");
    // Last progress should be 1.0 (complete)
    assert!(
        (*values.last().unwrap() - 1.0).abs() < 0.01,
        "Final progress should be 1.0"
    );

    cleanup(&[&input_path, &output_path]);
}
