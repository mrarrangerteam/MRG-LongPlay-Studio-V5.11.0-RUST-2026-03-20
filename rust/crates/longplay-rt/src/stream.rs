//! cpal audio stream callback and DSP processing.
//!
//! The audio callback reads samples from the source buffer,
//! processes them through the DSP chain, and writes to the output device.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleFormat, Stream, StreamConfig};
use crossbeam_channel::Sender;

use longplay_dsp::equalizer::Equalizer;
use longplay_dsp::dynamics::Dynamics;
use longplay_dsp::imager::Imager;
use longplay_dsp::maximizer::Maximizer;
use longplay_dsp::limiter::LookAheadLimiter;
use longplay_dsp::resonance_suppressor::ResonanceSuppressor;

use crate::params::RtParams;

/// Block size for DSP processing (512 samples = ~10.7ms @ 48kHz).
pub const BLOCK_SIZE: usize = 512;

/// Meter data sent from audio thread to Python at ~30Hz.
#[derive(Debug, Clone)]
pub struct MeterData {
    pub peak_l: f32,
    pub peak_r: f32,
    pub rms_l: f32,
    pub rms_r: f32,
    pub gain_reduction_db: f32,
    pub position_frames: u64,
}

impl Default for MeterData {
    fn default() -> Self {
        Self {
            peak_l: -200.0,
            peak_r: -200.0,
            rms_l: -200.0,
            rms_r: -200.0,
            gain_reduction_db: 0.0,
            position_frames: 0,
        }
    }
}

/// Shared state between the control thread and the audio callback.
pub struct StreamState {
    /// Deinterleaved audio data: [left_channel, right_channel]
    pub audio_data: Vec<Vec<f32>>,
    /// Sample rate of the loaded audio
    pub sample_rate: u32,
    /// Total number of frames
    pub total_frames: u64,
}

/// The audio output stream manager.
pub struct AudioStream {
    stream: Option<Stream>,
    /// Current playback position in frames (atomic for lock-free access).
    pub position: Arc<AtomicU64>,
    /// Whether playback is active.
    pub playing: Arc<AtomicBool>,
    /// Whether the stream has been created.
    pub active: bool,
    /// Meter data sender (audio thread → main thread).
    pub meter_tx: Sender<MeterData>,
    /// Shared parameters (Python → audio thread).
    pub params: Arc<RtParams>,
}

impl AudioStream {
    /// Create a new audio stream for the given audio data.
    ///
    /// `meter_tx` receives MeterData at ~30Hz from the audio callback.
    /// `params` are lock-free atomic parameters controlled by the GUI.
    pub fn new(
        state: StreamState,
        params: Arc<RtParams>,
        meter_tx: Sender<MeterData>,
    ) -> Result<Self, String> {
        let position = Arc::new(AtomicU64::new(0));
        let playing = Arc::new(AtomicBool::new(false));

        let mut stream_obj = Self {
            stream: None,
            position: position.clone(),
            playing: playing.clone(),
            active: false,
            meter_tx: meter_tx.clone(),
            params: params.clone(),
        };

        stream_obj.build_stream(state, params, meter_tx, position, playing)?;

        Ok(stream_obj)
    }

    fn build_stream(
        &mut self,
        state: StreamState,
        params: Arc<RtParams>,
        meter_tx: Sender<MeterData>,
        position: Arc<AtomicU64>,
        playing: Arc<AtomicBool>,
    ) -> Result<(), String> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| "No audio output device found".to_string())?;

        // Try to match the audio file's sample rate
        let target_sample_rate = state.sample_rate;
        let config = find_best_config(&device, target_sample_rate)?;

        let output_sample_rate = config.sample_rate.0;
        let output_channels = config.channels as usize;

        // Clone audio data into Arc for the callback
        let audio_l = Arc::new(state.audio_data[0].clone());
        let audio_r = if state.audio_data.len() > 1 {
            Arc::new(state.audio_data[1].clone())
        } else {
            audio_l.clone() // mono → duplicate
        };
        let total_frames = state.total_frames;

        // DSP chain lives inside the audio callback closure
        let mut resonance_suppressor = ResonanceSuppressor::new();
        let mut equalizer = Equalizer::new();
        let mut dynamics = Dynamics::new();
        let mut imager = Imager::new();
        let mut maximizer = Maximizer::new();
        let mut limiter = LookAheadLimiter::new();

        // Meter throttle: send every N callbacks (~30Hz)
        let callbacks_per_meter = (output_sample_rate as f64 / BLOCK_SIZE as f64 / 30.0).max(1.0) as u32;
        let mut callback_count: u32 = 0;

        // Simple sample rate conversion ratio (if needed)
        let src_ratio = target_sample_rate as f64 / output_sample_rate as f64;

        let stream = device
            .build_output_stream(
                &config,
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    let is_playing = playing.load(Ordering::Relaxed);
                    if !is_playing {
                        // Output silence
                        for sample in data.iter_mut() {
                            *sample = 0.0;
                        }
                        return;
                    }

                    let mut current_pos = position.load(Ordering::Relaxed);
                    let num_output_frames = data.len() / output_channels;

                    // Check if params changed and apply to DSP modules
                    if params.take_dirty() {
                        apply_params_to_dsp(
                            &params,
                            &mut resonance_suppressor,
                            &mut equalizer,
                            &mut dynamics,
                            &mut imager,
                            &mut maximizer,
                            &mut limiter,
                        );
                    }

                    let volume = params.volume.load();

                    // Metering accumulators
                    let mut peak_l: f32 = 0.0;
                    let mut peak_r: f32 = 0.0;
                    let mut sum_sq_l: f32 = 0.0;
                    let mut sum_sq_r: f32 = 0.0;

                    // Process multiple DSP blocks to fill the entire output buffer.
                    // Previously only one block was processed, causing sample repetition
                    // when cpal requested more frames than BLOCK_SIZE.
                    let mut frames_written: usize = 0;
                    let sr = target_sample_rate as i32;

                    // Pre-allocate block buffers outside loop (avoid heap alloc per block)
                    let mut block_l = vec![0.0f32; BLOCK_SIZE];
                    let mut block_r = vec![0.0f32; BLOCK_SIZE];

                    while frames_written < num_output_frames {
                        let remaining = num_output_frames - frames_written;
                        let block_frames = remaining.min(BLOCK_SIZE);

                        // Zero and fill block from source audio
                        for s in block_l[..block_frames].iter_mut() { *s = 0.0; }
                        for s in block_r[..block_frames].iter_mut() { *s = 0.0; }
                        for i in 0..block_frames {
                            let src_pos = ((current_pos as f64 + i as f64 * src_ratio) as u64)
                                .min(total_frames.saturating_sub(1));
                            let idx = src_pos as usize;
                            if idx < audio_l.len() {
                                block_l[i] = audio_l[idx];
                                block_r[i] = audio_r[idx];
                            }
                        }

                        // Build AudioBuffer (deinterleaved) — slice to actual block size
                        let mut buffer = vec![
                            block_l[..block_frames].to_vec(),
                            block_r[..block_frames].to_vec(),
                        ];

                        // Pre-gain headroom: -3 dB (matches V2 Python chain)
                        // Prevents clipping before the DSP chain on hot input signals
                        // (e.g. Suno/Udio tracks at -8.6 dB RMS)
                        const PRE_GAIN_LINEAR: f32 = 0.7079; // 10^(-3/20)
                        for ch in buffer.iter_mut() {
                            for s in ch.iter_mut() {
                                *s *= PRE_GAIN_LINEAR;
                            }
                        }

                        // DSP chain: ResSup → EQ → Dynamics → Imager → Maximizer → Limiter
                        resonance_suppressor.process(&mut buffer, sr);
                        equalizer.process_in_place(&mut buffer, sr);
                        let buffer_dyn = dynamics.process(&buffer, sr);
                        buffer = buffer_dyn;
                        imager.process(&mut buffer, sr);
                        maximizer.process(&mut buffer, sr);

                        if !limiter.is_bypassed() {
                            let limited = limiter.process(&buffer, sr);
                            buffer = limited;
                        }

                        // Write processed block to output + compute meters
                        for i in 0..block_frames {
                            let l = buffer[0][i] * volume;
                            let r = buffer[1][i] * volume;

                            peak_l = peak_l.max(l.abs());
                            peak_r = peak_r.max(r.abs());
                            sum_sq_l += l * l;
                            sum_sq_r += r * r;

                            let base = (frames_written + i) * output_channels;
                            if base < data.len() {
                                data[base] = l;
                                if output_channels > 1 && base + 1 < data.len() {
                                    data[base + 1] = r;
                                }
                            }
                        }

                        // Advance position per block
                        let advance = (block_frames as f64 * src_ratio) as u64;
                        current_pos += advance;
                        frames_written += block_frames;

                        if current_pos >= total_frames {
                            // Fill remaining output with silence
                            for j in frames_written..num_output_frames {
                                let base = j * output_channels;
                                if base < data.len() {
                                    data[base] = 0.0;
                                    if output_channels > 1 && base + 1 < data.len() {
                                        data[base + 1] = 0.0;
                                    }
                                }
                            }
                            position.store(0, Ordering::Relaxed);
                            playing.store(false, Ordering::Relaxed);
                            frames_written = num_output_frames; // exit loop
                            break;
                        }
                    }

                    // Update final position
                    if current_pos < total_frames {
                        position.store(current_pos, Ordering::Relaxed);
                    }

                    // Send meter data at ~30Hz
                    callback_count += 1;
                    if callback_count >= callbacks_per_meter {
                        callback_count = 0;
                        let n = num_output_frames as f32;
                        let rms_l = (sum_sq_l / n).sqrt();
                        let rms_r = (sum_sq_r / n).sqrt();

                        let gr = maximizer.peak_reduction_db() as f32;

                        let _ = meter_tx.try_send(MeterData {
                            peak_l: to_db(peak_l),
                            peak_r: to_db(peak_r),
                            rms_l: to_db(rms_l),
                            rms_r: to_db(rms_r),
                            gain_reduction_db: gr,
                            position_frames: position.load(Ordering::Relaxed),
                        });
                    }
                },
                move |err| {
                    eprintln!("[longplay-rt] Audio stream error: {}", err);
                },
                None, // timeout
            )
            .map_err(|e| format!("Failed to build audio stream: {}", e))?;

        self.stream = Some(stream);
        self.active = true;
        Ok(())
    }

    pub fn play(&self) {
        self.playing.store(true, Ordering::Relaxed);
        if let Some(ref stream) = self.stream {
            let _ = stream.play();
        }
    }

    pub fn pause(&self) {
        self.playing.store(false, Ordering::Relaxed);
        if let Some(ref stream) = self.stream {
            let _ = stream.pause();
        }
    }

    pub fn stop(&self) {
        self.playing.store(false, Ordering::Relaxed);
        self.position.store(0, Ordering::Relaxed);
        if let Some(ref stream) = self.stream {
            let _ = stream.pause();
        }
    }

    pub fn seek(&self, frame: u64) {
        self.position.store(frame, Ordering::Relaxed);
    }

    pub fn is_playing(&self) -> bool {
        self.playing.load(Ordering::Relaxed)
    }

    pub fn current_position(&self) -> u64 {
        self.position.load(Ordering::Relaxed)
    }
}

/// Apply atomic parameters to the mutable DSP modules.
/// Called in the audio callback when `dirty` flag is set.
fn apply_params_to_dsp(
    params: &RtParams,
    resonance_suppressor: &mut ResonanceSuppressor,
    equalizer: &mut Equalizer,
    dynamics: &mut Dynamics,
    imager: &mut Imager,
    maximizer: &mut Maximizer,
    limiter: &mut LookAheadLimiter,
) {
    // Resonance Suppressor
    resonance_suppressor.set_bypass(params.res_bypass.load(Ordering::Relaxed));
    resonance_suppressor.set_depth(params.res_depth.load());
    resonance_suppressor.set_sharpness(params.res_sharpness.load());
    resonance_suppressor.set_selectivity(params.res_selectivity.load());
    resonance_suppressor.set_attack(params.res_attack_ms.load());
    resonance_suppressor.set_release(params.res_release_ms.load());
    resonance_suppressor.set_mix(params.res_mix.load());
    resonance_suppressor.set_trim(params.res_trim_db.load());
    resonance_suppressor.set_delta(params.res_delta.load(Ordering::Relaxed));
    let res_mode = params.res_mode.load(Ordering::Relaxed);
    resonance_suppressor.set_mode_str(if res_mode == 1 { "hard" } else { "soft" });

    // Dynamics (Compressor)
    dynamics.set_bypass(params.dyn_bypass.load(Ordering::Relaxed));
    dynamics.single_band_mut().set_threshold(params.dyn_threshold.load() as f64);
    dynamics.single_band_mut().set_ratio(params.dyn_ratio.load() as f64);
    dynamics.single_band_mut().set_attack(params.dyn_attack_ms.load() as f64);
    dynamics.single_band_mut().set_release(params.dyn_release_ms.load() as f64);
    dynamics.single_band_mut().set_makeup_gain(params.dyn_makeup_db.load() as f64);
    dynamics.single_band_mut().set_knee(params.dyn_knee.load() as f64);

    // EQ
    equalizer.set_bypass(params.eq_bypass.load(Ordering::Relaxed));
    for i in 0..8 {
        equalizer.band_mut(i).set_gain(params.eq_gains[i].load() as f64);
    }

    // Imager
    let multiband = params.imager_multiband.load(Ordering::Relaxed);
    imager.set_multiband(multiband);
    if multiband {
        imager.set_low_width(params.low_width.load());
        imager.set_mid_width(params.mid_width.load());
        imager.set_high_width(params.high_width.load());
    } else {
        imager.set_width(params.width_pct.load());
    }

    // Maximizer
    maximizer.set_gain_db(params.gain_db.load());
    maximizer.set_ceiling(params.ceiling_db.load());
    let irc = params.irc_mode.load(Ordering::Relaxed);
    maximizer.set_irc_mode_int(irc);

    // Limiter
    limiter.set_bypass(params.limiter_bypass.load(Ordering::Relaxed));
    limiter.set_ceiling(params.limiter_ceiling_db.load() as f64);
}

/// Find the best output config matching the target sample rate.
fn find_best_config(device: &Device, target_sr: u32) -> Result<StreamConfig, String> {
    let supported = device
        .supported_output_configs()
        .map_err(|e| format!("Failed to query audio configs: {}", e))?;

    // Try to find a config that supports our target sample rate with f32 format
    let mut best: Option<StreamConfig> = None;
    for range in supported {
        if range.sample_format() != SampleFormat::F32 {
            continue;
        }
        let min_sr = range.min_sample_rate().0;
        let max_sr = range.max_sample_rate().0;
        if target_sr >= min_sr && target_sr <= max_sr {
            let config = range.with_sample_rate(cpal::SampleRate(target_sr)).config();
            return Ok(config);
        }
        // Fallback: pick any f32 config at 48kHz or 44100
        if best.is_none() {
            let fallback_sr = if 48000 >= min_sr && 48000 <= max_sr {
                48000
            } else if 44100 >= min_sr && 44100 <= max_sr {
                44100
            } else {
                max_sr
            };
            best = Some(range.with_sample_rate(cpal::SampleRate(fallback_sr)).config());
        }
    }

    best.ok_or_else(|| "No suitable audio output configuration found".to_string())
}

#[inline]
fn to_db(linear: f32) -> f32 {
    if linear < 1e-10 {
        -200.0
    } else {
        20.0 * linear.log10()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_db() {
        assert!((to_db(1.0) - 0.0).abs() < 1e-4);
        assert!(to_db(0.0) < -100.0);
        assert!((to_db(0.5) - (-6.0206)).abs() < 0.01);
    }

    #[test]
    fn test_meter_data_default() {
        let m = MeterData::default();
        assert!(m.peak_l < -100.0);
        assert_eq!(m.position_frames, 0);
    }
}
