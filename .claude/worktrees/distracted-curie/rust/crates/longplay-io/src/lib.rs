use longplay_core::{AudioBuffer, AudioInfo, LongplayError, Result};
use longplay_core::conversions::{db_to_linear_f, linear_to_db_f, linear_to_db};

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// Read a WAV audio file and return deinterleaved buffer + info
pub fn read_audio(path: &str) -> Result<(AudioBuffer, AudioInfo)> {
    let reader = hound::WavReader::open(path)
        .map_err(|e| LongplayError::AudioIo(format!("Failed to open {}: {}", path, e)))?;

    let spec = reader.spec();
    let num_channels = spec.channels as usize;
    let sample_rate = spec.sample_rate as i32;
    let bit_depth = spec.bits_per_sample as i32;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max_val))
                .collect::<std::result::Result<Vec<f32>, _>>()
                .map_err(|e| LongplayError::AudioIo(format!("Failed to read samples: {}", e)))?
        }
        hound::SampleFormat::Float => {
            reader
                .into_samples::<f32>()
                .collect::<std::result::Result<Vec<f32>, _>>()
                .map_err(|e| LongplayError::AudioIo(format!("Failed to read samples: {}", e)))?
        }
    };

    let frames = samples.len() / num_channels;
    let buffer = longplay_core::deinterleave(&samples, num_channels);

    let info = AudioInfo {
        sample_rate,
        channels: num_channels as i32,
        frames: frames as i64,
        bit_depth,
        format: "wav".to_string(),
        duration_seconds: frames as f64 / sample_rate as f64,
    };

    Ok((buffer, info))
}

/// Write audio buffer to WAV file
pub fn write_audio(
    path: &str,
    buffer: &AudioBuffer,
    sample_rate: i32,
    bit_depth: i32,
) -> Result<()> {
    if buffer.is_empty() {
        return Err(LongplayError::AudioIo("Empty buffer".to_string()));
    }

    let channels = buffer.len() as u16;
    let spec = hound::WavSpec {
        channels,
        sample_rate: sample_rate as u32,
        bits_per_sample: if bit_depth == 32 { 32 } else { bit_depth as u16 },
        sample_format: if bit_depth == 32 {
            hound::SampleFormat::Float
        } else {
            hound::SampleFormat::Int
        },
    };

    let mut writer = hound::WavWriter::create(path, spec)
        .map_err(|e| LongplayError::AudioIo(format!("Failed to create {}: {}", path, e)))?;

    let interleaved = longplay_core::interleave(buffer);

    match spec.sample_format {
        hound::SampleFormat::Float => {
            for sample in &interleaved {
                writer.write_sample(*sample)
                    .map_err(|e| LongplayError::AudioIo(format!("Write error: {}", e)))?;
            }
        }
        hound::SampleFormat::Int => {
            let max_val = (1i64 << (bit_depth - 1)) as f32;
            for sample in &interleaved {
                let int_sample = (*sample * max_val).round() as i32;
                writer.write_sample(int_sample)
                    .map_err(|e| LongplayError::AudioIo(format!("Write error: {}", e)))?;
            }
        }
    }

    writer.finalize()
        .map_err(|e| LongplayError::AudioIo(format!("Finalize error: {}", e)))?;

    Ok(())
}

/// Get audio file info without reading data
pub fn get_info(path: &str) -> Result<AudioInfo> {
    let reader = hound::WavReader::open(path)
        .map_err(|e| LongplayError::AudioIo(format!("Failed to open {}: {}", path, e)))?;

    let spec = reader.spec();
    let frames = reader.len() as i64 / spec.channels as i64;

    Ok(AudioInfo {
        sample_rate: spec.sample_rate as i32,
        channels: spec.channels as i32,
        frames,
        bit_depth: spec.bits_per_sample as i32,
        format: "wav".to_string(),
        duration_seconds: frames as f64 / spec.sample_rate as f64,
    })
}

/// Resample audio buffer (simple linear interpolation)
pub fn resample(input: &AudioBuffer, from_rate: i32, to_rate: i32) -> AudioBuffer {
    if from_rate == to_rate {
        return input.clone();
    }
    let ratio = to_rate as f64 / from_rate as f64;
    let new_frames = (input[0].len() as f64 * ratio) as usize;

    input
        .iter()
        .map(|channel| {
            (0..new_frames)
                .map(|i| {
                    let src_idx = i as f64 / ratio;
                    let idx0 = src_idx as usize;
                    let frac = src_idx - idx0 as f64;
                    let idx1 = (idx0 + 1).min(channel.len() - 1);
                    ((1.0 - frac) * channel[idx0] as f64 + frac * channel[idx1] as f64) as f32
                })
                .collect()
        })
        .collect()
}

/// Convert mono to stereo (duplicate channel)
pub fn mono_to_stereo(input: &AudioBuffer) -> AudioBuffer {
    if input.len() >= 2 {
        return input.clone();
    }
    vec![input[0].clone(), input[0].clone()]
}

/// Convert stereo to mono (average channels)
pub fn stereo_to_mono(input: &AudioBuffer) -> AudioBuffer {
    if input.len() == 1 {
        return input.clone();
    }
    let frames = input[0].len();
    let mono: Vec<f32> = (0..frames)
        .map(|i| (input[0][i] + input[1][i]) * 0.5)
        .collect();
    vec![mono]
}

/// Normalize audio to target peak level in dB
pub fn normalize_peak(input: &AudioBuffer, target_db: f32) -> AudioBuffer {
    let current = peak_db(input);
    let gain = target_db - current;
    apply_gain(input, gain)
}

/// Apply gain in dB to all samples
pub fn apply_gain(input: &AudioBuffer, gain_db: f32) -> AudioBuffer {
    let gain_linear = db_to_linear_f(gain_db);
    input
        .iter()
        .map(|ch| ch.iter().map(|s| s * gain_linear).collect())
        .collect()
}

/// Compute peak level in dB
pub fn peak_db(input: &AudioBuffer) -> f32 {
    let peak = input
        .iter()
        .flat_map(|ch| ch.iter())
        .fold(0.0f32, |max, &s| max.max(s.abs()));
    linear_to_db_f(peak)
}

/// Read any supported audio format (MP3, FLAC, OGG, WAV) using symphonia.
/// Falls back to ffmpeg subprocess for exotic formats.
pub fn read_audio_any(path: &str) -> Result<(AudioBuffer, AudioInfo)> {
    // Try WAV first (fastest path)
    if path.to_lowercase().ends_with(".wav") {
        return read_audio(path);
    }

    // Try symphonia for MP3, FLAC, OGG, AAC
    match read_audio_symphonia(path) {
        Ok(result) => return Ok(result),
        Err(e) => {
            eprintln!("Symphonia decode failed for {}: {}, trying ffmpeg fallback", path, e);
        }
    }

    // FFmpeg subprocess fallback for exotic formats
    read_audio_ffmpeg(path)
}

/// Decode audio using symphonia (MP3, FLAC, OGG, AAC)
fn read_audio_symphonia(path: &str) -> Result<(AudioBuffer, AudioInfo)> {
    let file = std::fs::File::open(path)
        .map_err(|e| LongplayError::AudioIo(format!("Failed to open {}: {}", path, e)))?;

    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = std::path::Path::new(path).extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .map_err(|e| LongplayError::AudioIo(format!("Probe failed: {}", e)))?;

    let mut format = probed.format;

    let track = format.default_track()
        .ok_or_else(|| LongplayError::AudioIo("No audio track found".to_string()))?;

    let track_id = track.id;
    let codec_params = track.codec_params.clone();

    let sample_rate = codec_params.sample_rate
        .ok_or_else(|| LongplayError::AudioIo("Unknown sample rate".to_string()))? as i32;
    let num_channels = codec_params.channels
        .map(|c| c.count())
        .unwrap_or(2);

    let mut decoder = symphonia::default::get_codecs()
        .make(&codec_params, &DecoderOptions::default())
        .map_err(|e| LongplayError::AudioIo(format!("Decoder creation failed: {}", e)))?;

    let mut all_samples: Vec<Vec<f32>> = vec![Vec::new(); num_channels];

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(symphonia::core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => {
                eprintln!("Packet read warning: {}", e);
                break;
            }
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(buf) => buf,
            Err(e) => {
                eprintln!("Decode warning: {}", e);
                continue;
            }
        };

        let spec = *decoded.spec();
        let duration = decoded.capacity();

        let mut sample_buf = SampleBuffer::<f32>::new(duration as u64, spec);
        sample_buf.copy_interleaved_ref(decoded);

        let interleaved = sample_buf.samples();
        let ch = spec.channels.count();
        let frames = interleaved.len() / ch;

        for frame in 0..frames {
            for c in 0..ch.min(num_channels) {
                all_samples[c].push(interleaved[frame * ch + c]);
            }
        }
    }

    let frames = if all_samples.is_empty() || all_samples[0].is_empty() {
        0
    } else {
        all_samples[0].len()
    };

    let info = AudioInfo {
        sample_rate,
        channels: num_channels as i32,
        frames: frames as i64,
        bit_depth: 32,
        format: std::path::Path::new(path)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("unknown")
            .to_string(),
        duration_seconds: frames as f64 / sample_rate as f64,
    };

    Ok((all_samples, info))
}

/// FFmpeg subprocess fallback for exotic formats
fn read_audio_ffmpeg(path: &str) -> Result<(AudioBuffer, AudioInfo)> {
    use std::process::Command;

    // Get info first
    let probe_output = Command::new("ffprobe")
        .args(["-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", path])
        .output()
        .map_err(|e| LongplayError::Ffmpeg(format!("ffprobe failed: {}", e)))?;

    let probe_json: serde_json::Value = serde_json::from_slice(&probe_output.stdout)
        .unwrap_or_default();

    let sample_rate = probe_json["streams"][0]["sample_rate"]
        .as_str()
        .and_then(|s| s.parse::<i32>().ok())
        .unwrap_or(44100);

    let channels = probe_json["streams"][0]["channels"]
        .as_i64()
        .unwrap_or(2) as i32;

    // Decode to raw PCM
    let output = Command::new("ffmpeg")
        .args([
            "-i", path,
            "-f", "f32le",
            "-acodec", "pcm_f32le",
            "-ar", &sample_rate.to_string(),
            "-ac", &channels.to_string(),
            "-"
        ])
        .output()
        .map_err(|e| LongplayError::Ffmpeg(format!("ffmpeg decode failed: {}", e)))?;

    if !output.status.success() {
        return Err(LongplayError::Ffmpeg("ffmpeg decode failed".to_string()));
    }

    let raw_data = &output.stdout;
    let samples: Vec<f32> = raw_data
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    let frames = samples.len() / channels as usize;
    let buffer = longplay_core::deinterleave(&samples, channels as usize);

    let info = AudioInfo {
        sample_rate,
        channels,
        frames: frames as i64,
        bit_depth: 32,
        format: std::path::Path::new(path)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("unknown")
            .to_string(),
        duration_seconds: frames as f64 / sample_rate as f64,
    };

    Ok((buffer, info))
}

/// Compute RMS level in dB
pub fn rms_db(input: &AudioBuffer) -> f32 {
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for ch in input {
        for &s in ch {
            sum += (s as f64) * (s as f64);
            count += 1;
        }
    }
    if count == 0 {
        return -200.0;
    }
    linear_to_db((sum / count as f64).sqrt()) as f32
}
