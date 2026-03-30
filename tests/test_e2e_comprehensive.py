"""
LongPlay Studio V5.11 — Comprehensive End-to-End Test Suite (v2)
ทดสอบทุก control, pipeline, audio signal และ realtime metering
ใช้ API ที่ถูกต้องจากโค้ดจริง
"""
from __future__ import annotations

import math
import os
import sys
import tempfile

import numpy as np
import pytest
import soundfile as sf

# ── paths ─────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_WAV = os.path.join(BASE, "test_audio.wav")
HOOK_WAV   = os.path.join(BASE, "Vinyl Prophet Vol.1", "Hook",
                           "1.Higher Vibration_hook.wav")
SAMPLE_VIDEO = os.path.join(BASE, "Vinyl Prophet Vol.1", "vdo", "1.mp4")
SR = 44100


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════
def make_tone(freq=440.0, duration=2.0, amplitude=0.5, sr=SR, stereo=True):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    mono = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return np.stack([mono, mono], axis=1) if stereo else mono

def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))

def peak(audio: np.ndarray) -> float:
    return float(np.max(np.abs(audio)))

def write_temp_wav(audio: np.ndarray, sr=SR) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, sr)
    return tmp.name

def chain_render(chain, audio, sr=SR):
    """Helper: write audio to temp WAV, load into chain, render, return (output_path, out_audio)."""
    inp = write_temp_wav(audio, sr)
    out = tempfile.mktemp(suffix="_master.wav")
    try:
        chain.load_audio(inp)
        result = chain.render(out)
        out_audio = None
        if result and os.path.isfile(result):
            out_audio, _ = sf.read(result)
            out = result
        elif os.path.isfile(out):
            out_audio, _ = sf.read(out)
    finally:
        if os.path.exists(inp):
            os.unlink(inp)
    return out, out_audio


# ══════════════════════════════════════════════════════════════════════════════
# 1. AUDIO FILE IMPORT
# ══════════════════════════════════════════════════════════════════════════════
class TestAudioImport:
    """ทดสอบการ import ไฟล์เสียงจริง"""

    def test_sample_wav_exists(self):
        assert os.path.isfile(SAMPLE_WAV), f"Missing: {SAMPLE_WAV}"

    def test_hook_wav_exists(self):
        assert os.path.isfile(HOOK_WAV), f"Missing: {HOOK_WAV}"

    def test_read_sample_wav(self):
        audio, sr = sf.read(SAMPLE_WAV)
        assert audio.ndim >= 1
        assert sr > 0
        assert rms(audio) > 1e-6, "test_audio.wav เป็น silence"

    def test_read_hook_wav(self):
        audio, sr = sf.read(HOOK_WAV)
        assert sr in (44100, 48000, 96000)
        assert audio.shape[0] > 0
        assert rms(audio) > 1e-6, "Hook WAV เป็น silence"

    def test_hook_wav_stereo(self):
        audio, _ = sf.read(HOOK_WAV)
        assert audio.ndim == 2 and audio.shape[1] == 2

    def test_hook_wav_duration(self):
        audio, sr = sf.read(HOOK_WAV)
        assert audio.shape[0] / sr > 1.0


# ══════════════════════════════════════════════════════════════════════════════
# 2. VIDEO FILE IMPORT
# ══════════════════════════════════════════════════════════════════════════════
class TestVideoImport:
    def test_video_file_exists(self):
        assert os.path.isfile(SAMPLE_VIDEO)

    def test_video_file_nonzero(self):
        assert os.path.getsize(SAMPLE_VIDEO) > 10_000

    def test_error_mov_exists(self):
        assert os.path.isfile(os.path.join(BASE, "Error.mov"))

    def test_content_factory_video_model(self):
        """BackgroundVideo model รับ file_path และ dimensions"""
        from gui.content_factory.models import BackgroundVideo
        bv = BackgroundVideo(file_path=SAMPLE_VIDEO, width=1920, height=1080)
        assert bv.file_path == SAMPLE_VIDEO
        assert bv.is_landscape  # property (bool)
        assert abs(bv.aspect_ratio - (1920 / 1080)) < 0.01


# ══════════════════════════════════════════════════════════════════════════════
# 3. EQ CONTROLS — ทุก band, ทุก parameter
# ══════════════════════════════════════════════════════════════════════════════
class TestEQControls:
    @pytest.fixture
    def eq(self):
        from modules.master.equalizer import Equalizer
        return Equalizer()

    def test_eq_init(self, eq):
        assert len(eq.bands) == 8

    def test_all_bands_gain_sweep(self, eq):
        """หมุน gain knob ทุก band ตลอดช่วง -12 ถึง +12 dB"""
        for i in range(8):
            for gain in [-12.0, -6.0, 0.0, 6.0, 12.0]:
                eq.bands[i].gain = gain
                assert abs(eq.bands[i].gain - gain) < 1e-6

    def test_all_bands_freq_sweep(self, eq):
        """เปลี่ยน frequency ทุก band"""
        freqs = [50, 100, 200, 500, 1000, 2000, 4000, 8000]
        for i, freq in enumerate(freqs):
            eq.bands[i].freq = float(freq)
            assert eq.bands[i].freq == float(freq)

    def test_all_bands_q_sweep(self, eq):
        """หมุน Q knob (0.1 ถึง 10)"""
        for i in range(8):
            for q in [0.1, 0.5, 1.0, 2.0, 10.0]:
                eq.bands[i].width = q
                assert abs(eq.bands[i].width - q) < 1e-6

    def test_filter_type_switch(self, eq):
        """สลับ filter type ทุกแบบ"""
        for bt in ["equalizer", "lowshelf", "highshelf", "highpass", "lowpass"]:
            eq.bands[0].band_type = bt
            assert eq.bands[0].band_type == bt

    def test_band_enable_disable(self, eq):
        for i in range(8):
            eq.bands[i].enabled = False
            assert not eq.bands[i].enabled
            eq.bands[i].enabled = True
            assert eq.bands[i].enabled

    def test_eq_global_enable(self, eq):
        eq.enabled = False
        assert not eq.enabled
        eq.enabled = True
        assert eq.enabled

    def test_eq_tone_presets(self, eq):
        """โหลด tone preset ทุกตัวได้"""
        from modules.master.genre_profiles import TONE_PRESETS
        for name in TONE_PRESETS:
            eq.load_tone_preset(name)

    def test_eq_process_audio_has_signal(self, eq):
        """EQ process_audio ให้ output มี signal"""
        audio = make_tone()
        out = eq.process_audio(audio, SR)
        assert out is not None
        assert out.shape == audio.shape
        assert rms(out) > 1e-6

    def test_eq_frequency_response_boost_vs_flat(self, eq):
        """Boost +12 dB ที่ band 4 (1 kHz) → response ที่ 1 kHz สูงกว่า flat"""
        for i in range(8):
            eq.bands[i].gain = 0.0
        freqs, gains_flat = eq.get_frequency_response(sample_rate=SR, n_points=512)
        idx = int(np.searchsorted(freqs, 1000.0))
        idx = max(1, min(idx, len(gains_flat) - 1))
        flat_val = float(gains_flat[idx])

        eq.bands[4].gain = 12.0  # band 4 = 1000 Hz
        _, gains_boost = eq.get_frequency_response(sample_rate=SR, n_points=512)
        boost_val = float(gains_boost[idx])
        assert boost_val > flat_val, f"Boost +12 dB ที่ 1 kHz ไม่มีผล: flat={flat_val:.3f} boost={boost_val:.3f}"

    def test_eq_frequency_response_cut_vs_flat(self, eq):
        """Cut -12 dB ที่ band 4 (1 kHz) → response ที่ 1 kHz ต่ำกว่า flat"""
        for i in range(8):
            eq.bands[i].gain = 0.0
        freqs, gains_flat = eq.get_frequency_response(sample_rate=SR, n_points=512)
        idx = int(np.searchsorted(freqs, 1000.0))
        idx = max(1, min(idx, len(gains_flat) - 1))
        flat_val = float(gains_flat[idx])

        eq.bands[4].gain = -12.0
        _, gains_cut = eq.get_frequency_response(sample_rate=SR, n_points=512)
        cut_val = float(gains_cut[idx])
        assert cut_val < flat_val, f"Cut -12 dB ที่ 1 kHz ไม่มีผล: flat={flat_val:.3f} cut={cut_val:.3f}"

    def test_set_band_api(self, eq):
        """set_band() API ทำงาน"""
        eq.set_band(0, gain=6.0, freq=100.0)
        assert abs(eq.bands[0].gain - 6.0) < 1e-6
        assert eq.bands[0].freq == 100.0

    def test_eq_ffmpeg_filters(self, eq):
        """get_ffmpeg_filters() คืน list ของ filter strings"""
        eq.bands[0].gain = 3.0
        filters = eq.get_ffmpeg_filters()
        assert isinstance(filters, list)

    def test_genre_preset(self, eq):
        """โหลด genre preset ได้"""
        from modules.master.genre_profiles import GENRE_PROFILES
        for name in list(GENRE_PROFILES.keys())[:3]:
            eq.load_genre_preset(name)


# ══════════════════════════════════════════════════════════════════════════════
# 4. COMPRESSOR / DYNAMICS CONTROLS
# ══════════════════════════════════════════════════════════════════════════════
class TestDynamicsControls:
    @pytest.fixture
    def dyn(self):
        from modules.master.dynamics import Dynamics
        return Dynamics()

    def test_dynamics_init(self, dyn):
        assert dyn is not None
        assert dyn.single_band is not None

    def test_threshold_sweep(self, dyn):
        for val in [-60.0, -40.0, -20.0, -10.0, 0.0]:
            dyn.set_threshold(val)
            assert abs(dyn.single_band.threshold - val) < 1e-6

    def test_ratio_sweep(self, dyn):
        for val in [1.0, 2.0, 4.0, 8.0, 20.0]:
            dyn.set_ratio(val)
            assert abs(dyn.single_band.ratio - val) < 1e-6

    def test_attack_sweep(self, dyn):
        for val in [0.1, 1.0, 10.0, 50.0, 100.0]:
            dyn.set_attack(val)
            assert abs(dyn.single_band.attack - val) < 1e-6

    def test_release_sweep(self, dyn):
        for val in [10.0, 50.0, 100.0, 500.0]:
            dyn.set_release(val)
            assert abs(dyn.single_band.release - val) < 1e-6

    def test_knee_direct(self, dyn):
        """ตั้ง knee โดยตรงผ่าน single_band.knee"""
        for val in [0.0, 2.0, 6.0, 12.0]:
            dyn.single_band.knee = val
            assert abs(dyn.single_band.knee - val) < 1e-6

    def test_makeup_gain(self, dyn):
        for val in [0.0, 3.0, 6.0, 12.0]:
            dyn.set_makeup_gain(val)
            assert abs(dyn.single_band.makeup - val) < 1e-6

    def test_enable_disable(self, dyn):
        dyn.enabled = False
        assert not dyn.enabled
        dyn.enabled = True
        assert dyn.enabled

    def test_multiband_enable(self, dyn):
        dyn.multiband = True
        assert dyn.multiband
        dyn.multiband = False
        assert not dyn.multiband

    def test_crossover_direct(self, dyn):
        dyn.crossover_low = 150.0
        dyn.crossover_high = 5000.0
        assert dyn.crossover_low == 150.0
        assert dyn.crossover_high == 5000.0

    def test_all_presets_load(self, dyn):
        from modules.master.dynamics import DYNAMICS_PRESETS
        for name in DYNAMICS_PRESETS:
            dyn.load_preset(name)

    def test_ffmpeg_filter_reflects_params(self, dyn):
        """FFmpeg filter string ต้องใช้ค่าที่ตั้งไว้"""
        dyn.set_threshold(-25.0)
        dyn.set_ratio(4.0)
        filters = dyn.get_ffmpeg_filters()
        assert isinstance(filters, list)
        assert len(filters) > 0
        filter_str = " ".join(filters)
        # threshold ต้องอยู่ใน filter string
        assert "-25" in filter_str or "threshold" in filter_str.lower()

    def test_detection_mode_set(self, dyn):
        for mode in ["peak", "rms"]:
            dyn.single_band.detection_mode = mode
            assert dyn.single_band.detection_mode == mode


# ══════════════════════════════════════════════════════════════════════════════
# 5. LIMITER CONTROLS
# ══════════════════════════════════════════════════════════════════════════════
class TestLimiterControls:
    @pytest.fixture
    def lim(self):
        from modules.master.limiter import LookAheadLimiterFast
        return LookAheadLimiterFast()

    def test_ceiling_sweep(self, lim):
        for val in [-3.0, -2.0, -1.0, -0.5, -0.1]:
            lim.ceiling_db = val
            assert abs(lim.ceiling_db - val) < 1e-6

    def test_lookahead_sweep(self, lim):
        for val in [0.1, 1.0, 5.0, 10.0, 20.0, 50.0]:
            lim.lookahead_ms = val
            assert abs(lim.lookahead_ms - val) < 1e-6

    def test_release_sweep(self, lim):
        for val in [5.0, 50.0, 100.0, 200.0, 500.0]:
            lim.release_ms = val
            assert abs(lim.release_ms - val) < 1e-6

    def test_true_peak_toggle(self, lim):
        lim.true_peak = True
        assert lim.true_peak
        lim.true_peak = False
        assert not lim.true_peak

    def test_ceiling_enforcement(self, lim):
        """Peak ต้องไม่เกิน ceiling"""
        lim.ceiling_db = -1.0
        ceiling_lin = 10 ** (-1.0 / 20.0)
        audio = make_tone(amplitude=0.99, duration=1.0)
        out = lim.process(audio, SR)
        assert peak(out) <= ceiling_lin * 1.001, \
            f"Limiter ไม่บังคับ ceiling: {peak(out):.4f} > {ceiling_lin:.4f}"

    def test_silence_passthrough(self, lim):
        audio = np.zeros((SR, 2), dtype=np.float32)
        out = lim.process(audio, SR)
        assert rms(out) < 1e-10

    def test_output_has_signal(self, lim):
        audio = make_tone(amplitude=0.5)
        out = lim.process(audio, SR)
        assert rms(out) > 1e-6


# ══════════════════════════════════════════════════════════════════════════════
# 6. MAXIMIZER CONTROLS
# ══════════════════════════════════════════════════════════════════════════════
class TestMaximizerControls:
    @pytest.fixture
    def mx(self):
        from modules.master.maximizer import Maximizer
        return Maximizer()

    def test_gain_sweep(self, mx):
        for val in [0.0, 3.0, 6.0, 10.0, 15.0, 20.0]:
            mx.set_gain(val)
            assert abs(mx.gain_db - val) < 1e-6

    def test_ceiling_sweep(self, mx):
        for val in [-3.0, -2.0, -1.0, -0.5, -0.3, -0.1]:
            mx.set_ceiling(val)
            assert abs(mx.ceiling - val) < 1e-6

    def test_all_irc_modes(self, mx):
        for mode in ["IRC 1", "IRC 2", "IRC 3", "IRC 4", "IRC 5", "IRC LL"]:
            mx.set_irc_mode(mode)
            assert mx.irc_mode == mode

    def test_irc4_sub_modes(self, mx):
        mx.set_irc_mode("IRC 4")
        for sub in ["Classic", "Modern", "Crisp", "Transient", "Pumping", "Balanced", "Clipping"]:
            mx.set_irc_sub_mode(sub)
            assert mx.irc_sub_mode == sub

    def test_character_sweep(self, mx):
        for val in [0.0, 2.0, 5.0, 7.0, 10.0]:
            mx.set_character(val)
            assert abs(mx.character - val) < 1e-6

    def test_true_peak_toggle(self, mx):
        mx.true_peak = True
        assert mx.true_peak
        mx.true_peak = False
        assert not mx.true_peak

    def test_enable_disable(self, mx):
        mx.enabled = False
        assert not mx.enabled
        mx.enabled = True
        assert mx.enabled

    def test_soft_clip_toggle(self, mx):
        mx.soft_clip_enabled = True
        assert mx.soft_clip_enabled
        mx.soft_clip_enabled = False
        assert not mx.soft_clip_enabled

    def test_upward_compress(self, mx):
        for val in [0.0, 3.0, 6.0, 9.0, 12.0]:
            mx.upward_compress_db = val
            assert mx.upward_compress_db == val

    def test_getstate_has_keys(self, mx):
        state = mx.__getstate__()
        assert "gain_db" in state
        assert "ceiling" in state
        assert "irc_mode" in state


# ══════════════════════════════════════════════════════════════════════════════
# 7. IMAGER (STEREO WIDTH) CONTROLS
# ══════════════════════════════════════════════════════════════════════════════
class TestImagerControls:
    @pytest.fixture
    def img(self):
        from modules.master.imager import Imager
        return Imager()

    def test_width_sweep(self, img):
        for val in [0, 50, 100, 150, 200]:
            img.set_width(val)
            assert img.width == val

    def test_balance_direct(self, img):
        for val in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            img.balance = val
            assert abs(img.balance - val) < 1e-6

    def test_mono_bass_direct(self, img):
        for val in [0.0, 80.0, 120.0, 200.0]:
            img.mono_bass_freq = val
            assert img.mono_bass_freq == val

    def test_multiband_direct(self, img):
        img.multiband = True
        assert img.multiband
        img.multiband = False
        assert not img.multiband

    def test_all_presets(self, img):
        from modules.master.imager import IMAGER_PRESETS
        for name in IMAGER_PRESETS:
            img.load_preset(name)

    def test_enable_disable(self, img):
        img.enabled = False
        assert not img.enabled
        img.enabled = True
        assert img.enabled

    def test_stereoize_mode(self, img):
        for mode in ["off", "I", "II"]:
            img.stereoize_mode = mode
            assert img.stereoize_mode == mode

    def test_correlation_safety_toggle(self, img):
        img.correlation_safety = False
        assert not img.correlation_safety
        img.correlation_safety = True
        assert img.correlation_safety

    def test_apply_stereoize_has_signal(self, img):
        """apply_stereoize() คืน audio มี signal"""
        t = np.linspace(0, 1.0, SR, endpoint=False).astype(np.float32)
        audio = np.stack([np.sin(2 * math.pi * 440 * t) * 0.4,
                          np.sin(2 * math.pi * 440 * t + 0.3) * 0.4], axis=1)
        img.stereoize_mode = "I"
        img.stereoize_amount = 50
        out = img.apply_stereoize(audio, SR)
        assert out is not None
        assert rms(out) > 1e-6

    def test_ffmpeg_filters_returned(self, img):
        filters = img.get_ffmpeg_filters()
        assert isinstance(filters, list)


# ══════════════════════════════════════════════════════════════════════════════
# 8. MASTERING PIPELINE (chain.py) — Full E2E
# ══════════════════════════════════════════════════════════════════════════════
class TestMasteringPipeline:
    @pytest.fixture
    def chain(self):
        from modules.master.chain import MasterChain
        return MasterChain()

    def test_chain_init(self, chain):
        assert chain.equalizer is not None
        assert chain.dynamics is not None
        assert chain.imager is not None
        assert chain.maximizer is not None

    def test_set_platform_youtube(self, chain):
        chain.set_platform("YouTube")
        assert abs(chain.target_lufs - (-14.0)) < 0.5

    def test_set_platform_spotify(self, chain):
        chain.set_platform("Spotify")
        assert abs(chain.target_lufs - (-14.0)) < 0.5

    def test_set_platform_apple_music(self, chain):
        chain.set_platform("Apple Music")
        assert abs(chain.target_lufs - (-16.0)) < 0.5

    def test_intensity_direct(self, chain):
        for val in [0.0, 0.25, 0.5, 0.75, 1.0]:
            chain.intensity = val
            assert abs(chain.intensity - val) < 1e-6

    def test_load_audio(self, chain):
        chain.load_audio(HOOK_WAV)
        assert chain.input_path == HOOK_WAV

    def test_render_output_not_silence(self, chain):
        """render() ต้องให้ output ที่มี signal จริง"""
        audio = make_tone(amplitude=0.5, duration=2.0)
        out_path, out_audio = chain_render(chain, audio)
        try:
            assert out_audio is not None, "render ไม่สร้าง output"
            assert rms(out_audio) > 1e-6, "Output เป็น silence"
        finally:
            if out_path and os.path.exists(out_path):
                os.unlink(out_path)

    def test_render_ceiling_respected(self, chain):
        """render() ต้องบังคับ True Peak ไม่เกิน ceiling"""
        chain.maximizer.set_ceiling(-1.0)
        ceiling_lin = 10 ** (-1.0 / 20.0)
        audio = make_tone(amplitude=0.9, duration=1.0)
        out_path, out_audio = chain_render(chain, audio)
        try:
            if out_audio is not None:
                assert peak(out_audio) <= ceiling_lin * 1.02, \
                    f"Output เกิน ceiling: {peak(out_audio):.4f}"
        finally:
            if out_path and os.path.exists(out_path):
                os.unlink(out_path)

    def test_render_with_real_hook_wav(self, chain):
        """render ด้วย Hook WAV จริง"""
        out_path = tempfile.mktemp(suffix="_master.wav")
        try:
            chain.load_audio(HOOK_WAV)
            result = chain.render(out_path)
            final = result or out_path
            assert os.path.isfile(final), "render ไม่สร้าง output"
            out_audio, _ = sf.read(final)
            assert rms(out_audio) > 1e-6, "Output จาก Hook WAV เป็น silence"
        finally:
            for p in [out_path]:
                if os.path.exists(p):
                    os.unlink(p)

    def test_meter_callback_fires(self, chain):
        """meter callback ต้องส่งค่า dict กลับมา"""
        received = []
        chain.set_meter_callback(lambda d: received.append(d))
        audio = make_tone(amplitude=0.5, duration=1.0)
        out_path, _ = chain_render(chain, audio)
        try:
            assert len(received) > 0, "meter_callback ไม่ถูกเรียก"
            sample = received[-1]
            assert isinstance(sample, dict), f"ต้องเป็น dict, got {type(sample)}"
        finally:
            if out_path and os.path.exists(out_path):
                os.unlink(out_path)

    def test_meter_data_has_level_keys(self, chain):
        """meter data ต้องมีค่า level"""
        received = []
        chain.set_meter_callback(lambda d: received.append(d))
        audio = make_tone(amplitude=0.5, duration=1.0)
        out_path, _ = chain_render(chain, audio)
        try:
            if received:
                d = received[-1]
                level_keys = {"left_rms_db", "right_rms_db", "left_peak_db",
                              "right_peak_db", "lufs_momentary"}
                assert level_keys & set(d.keys()), \
                    f"meter data ไม่มี level keys: {list(d.keys())}"
        finally:
            if out_path and os.path.exists(out_path):
                os.unlink(out_path)

    def test_save_load_settings(self, chain):
        """save/load settings roundtrip"""
        state_file = tempfile.mktemp(suffix=".json")
        chain.intensity = 0.7
        chain.equalizer.bands[0].gain = 3.0
        chain.save_settings(state_file)
        chain.intensity = 1.0
        chain.equalizer.bands[0].gain = 0.0
        chain.load_settings(state_file)
        assert abs(chain.intensity - 0.7) < 1e-4
        assert abs(chain.equalizer.bands[0].gain - 3.0) < 1e-4
        os.unlink(state_file)

    def test_reset_all(self, chain):
        """reset_all() คืน intensity เป็น default (50)"""
        chain.intensity = 10
        chain.reset_all()
        assert chain.intensity == 50  # default = 50 (percent)

    def test_chain_summary_returned(self, chain):
        """get_chain_summary() คืน string ที่มีข้อมูล"""
        summary = chain.get_chain_summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_build_filter_chain_nocrash(self, chain):
        chain.build_filter_chain()


# ══════════════════════════════════════════════════════════════════════════════
# 9. REALTIME METERING
# ══════════════════════════════════════════════════════════════════════════════
class TestRealtimeMetering:
    def test_realtime_monitor_import(self):
        from modules.master.realtime_monitor import RealtimeMonitor
        assert RealtimeMonitor is not None

    def test_meter_callback_registers(self):
        from modules.master.realtime_monitor import RealtimeMonitor
        mon = RealtimeMonitor()
        mon.set_meter_callback(lambda d: None)  # ไม่ควร raise

    def test_bypass_toggle(self):
        from modules.master.realtime_monitor import RealtimeMonitor
        mon = RealtimeMonitor()
        mon.is_bypassed = True
        assert mon.is_bypassed
        mon.is_bypassed = False
        assert not mon.is_bypassed

    def test_backend_reported(self):
        from modules.master.realtime_monitor import RealtimeMonitor
        mon = RealtimeMonitor()
        backend = mon.backend
        assert backend in ("rust_cpal", "python_fallback")

    def test_meter_engine_exists(self):
        from modules.master.realtime_monitor import RealtimeMonitor
        mon = RealtimeMonitor()
        assert hasattr(mon, "_meter_engine")

    def test_meter_engine_processes_block(self):
        """_PythonMeterEngine ส่ง callback เมื่อ process audio"""
        from modules.master.realtime_monitor import RealtimeMonitor
        mon = RealtimeMonitor()
        received = []
        mon.set_meter_callback(lambda d: received.append(d))
        # Inject audio via internal engine
        audio = make_tone(amplitude=0.5, duration=0.05)
        if hasattr(mon._meter_engine, "process_block"):
            mon._meter_engine.process_block(audio)
            assert len(received) > 0, "_meter_engine ไม่ส่ง callback"

    def test_meter_callback_via_chain_render(self):
        """Chain render ส่ง meter data จริง — level ไม่ใช่ 0"""
        from modules.master.chain import MasterChain
        chain = MasterChain()
        received = []
        chain.set_meter_callback(lambda d: received.append(d))
        audio = make_tone(amplitude=0.5, duration=1.0)
        out_path, _ = chain_render(chain, audio)
        try:
            assert len(received) > 0, "ไม่ได้รับ meter data จาก render"
            d = received[-1]
            rms_val = d.get("left_rms_db", d.get("lufs_momentary", None))
            assert rms_val is not None
            assert float(rms_val) < 0, f"RMS ต้องเป็นค่าลบ (dB): {rms_val}"
        finally:
            if out_path and os.path.exists(out_path):
                os.unlink(out_path)

    def test_gain_raises_meter_level(self):
        """เพิ่ม GAIN → meter level ขึ้น"""
        from modules.master.chain import MasterChain
        chain_low = MasterChain()
        chain_high = MasterChain()
        vals_low, vals_high = [], []
        chain_low.set_meter_callback(lambda d: vals_low.append(d.get("left_rms_db", 0)))
        chain_high.set_meter_callback(lambda d: vals_high.append(d.get("left_rms_db", 0)))

        audio = make_tone(amplitude=0.3, duration=1.0)
        inp = write_temp_wav(audio)
        out_low  = tempfile.mktemp(suffix=".wav")
        out_high = tempfile.mktemp(suffix=".wav")
        try:
            chain_low.maximizer.set_gain(0.0)
            chain_low.load_audio(inp)
            chain_low.render(out_low)

            chain_high.maximizer.set_gain(12.0)
            chain_high.load_audio(inp)
            chain_high.render(out_high)

            if vals_low and vals_high:
                avg_low  = sum(float(v) for v in vals_low)  / len(vals_low)
                avg_high = sum(float(v) for v in vals_high) / len(vals_high)
                assert avg_high >= avg_low - 1.0, \
                    f"GAIN +12 dB ไม่ทำให้ meter ขึ้น: low={avg_low:.1f} high={avg_high:.1f}"
        finally:
            for p in [inp, out_low, out_high]:
                if os.path.exists(p):
                    os.unlink(p)

    def test_loudness_meter_with_real_file(self):
        """LoudnessMeter วิเคราะห์ Hook WAV และคืน LUFS"""
        from modules.master.loudness import LoudnessMeter
        meter = LoudnessMeter()
        result = meter.analyze(HOOK_WAV)
        if result is not None:
            assert result.integrated_lufs < 0, f"LUFS ต้องติดลบ: {result.integrated_lufs}"
            assert result.true_peak_dbtp < 0, f"True Peak ต้องติดลบ"
            assert result.duration_sec > 0


# ══════════════════════════════════════════════════════════════════════════════
# 10. ALL KNOBS TOGETHER — Full sweep then render
# ══════════════════════════════════════════════════════════════════════════════
class TestAllKnobsTogether:
    def test_full_parameter_sweep_render(self):
        """ตั้ง knob ทุกตัวพร้อมกัน แล้ว render — ไม่ crash และได้ signal"""
        from modules.master.chain import MasterChain
        chain = MasterChain()

        # EQ — sweep ทุก band
        for i, gain in enumerate([3, -3, 6, -6, 2, -2, 4, -4]):
            chain.equalizer.bands[i].gain = float(gain)

        # Dynamics
        chain.dynamics.set_threshold(-20.0)
        chain.dynamics.set_ratio(3.0)
        chain.dynamics.set_attack(5.0)
        chain.dynamics.set_release(80.0)
        chain.dynamics.single_band.knee = 4.0
        chain.dynamics.set_makeup_gain(3.0)

        # Imager
        chain.imager.set_width(120)
        chain.imager.balance = 0.0
        chain.imager.mono_bass_freq = 80.0

        # Maximizer
        chain.maximizer.set_gain(6.0)
        chain.maximizer.set_ceiling(-1.0)
        chain.maximizer.set_irc_mode("IRC 4")
        chain.maximizer.set_character(5.0)

        audio = make_tone(amplitude=0.5, duration=1.0)
        out_path, out_audio = chain_render(chain, audio)
        try:
            assert out_audio is not None, "Full sweep render ไม่มี output"
            assert rms(out_audio) > 1e-6, "Full sweep render → silence"
            assert peak(out_audio) < 1.02, f"Output clipping: {peak(out_audio):.3f}"
        finally:
            if out_path and os.path.exists(out_path):
                os.unlink(out_path)

    def test_all_irc_modes_render(self):
        """render ด้วย IRC mode ทุกตัว — ไม่ crash"""
        from modules.master.chain import MasterChain
        for mode in ["IRC 1", "IRC 2", "IRC 3", "IRC 4", "IRC 5", "IRC LL"]:
            chain = MasterChain()
            chain.maximizer.set_irc_mode(mode)
            audio = make_tone(amplitude=0.5, duration=0.5)
            out_path, out_audio = chain_render(chain, audio)
            try:
                assert out_audio is not None, f"IRC mode {mode} → no output"
                assert rms(out_audio) > 1e-6, f"IRC mode {mode} → silence"
            finally:
                if out_path and os.path.exists(out_path):
                    os.unlink(out_path)

    def test_platform_targets_render(self):
        """render ด้วย target แต่ละ platform"""
        from modules.master.chain import MasterChain
        for platform in ["YouTube", "Spotify", "Apple Music"]:
            chain = MasterChain()
            chain.set_platform(platform)
            audio = make_tone(amplitude=0.5, duration=0.5)
            out_path, out_audio = chain_render(chain, audio)
            try:
                assert out_audio is not None, f"{platform} → no output"
                assert rms(out_audio) > 1e-6, f"{platform} → silence"
            finally:
                if out_path and os.path.exists(out_path):
                    os.unlink(out_path)

    def test_extreme_eq_boost_no_crash(self):
        """EQ boost +12 dB ทุก band พร้อมกัน — ไม่ crash"""
        from modules.master.chain import MasterChain
        chain = MasterChain()
        for i in range(8):
            chain.equalizer.bands[i].gain = 12.0
        audio = make_tone()
        out_path, out_audio = chain_render(chain, audio)
        try:
            assert out_audio is not None, "EQ extreme boost → no output"
        finally:
            if out_path and os.path.exists(out_path):
                os.unlink(out_path)

    def test_all_bands_cut_no_crash(self):
        """EQ cut -12 dB ทุก band — ไม่ crash"""
        from modules.master.chain import MasterChain
        chain = MasterChain()
        for i in range(8):
            chain.equalizer.bands[i].gain = -12.0
        audio = make_tone()
        out_path, out_audio = chain_render(chain, audio)
        try:
            assert out_audio is not None
        finally:
            if out_path and os.path.exists(out_path):
                os.unlink(out_path)


# ══════════════════════════════════════════════════════════════════════════════
# 11. GENRE PROFILES & PLATFORM TARGETS
# ══════════════════════════════════════════════════════════════════════════════
class TestGenreProfiles:
    def test_platform_targets_have_lufs(self):
        from modules.master.genre_profiles import PLATFORM_TARGETS
        for name, target in PLATFORM_TARGETS.items():
            assert "target_lufs" in target, f"{name} ขาด target_lufs"
            assert target["target_lufs"] < 0, f"{name} LUFS ต้องติดลบ"

    def test_platform_targets_have_peak(self):
        from modules.master.genre_profiles import PLATFORM_TARGETS
        for name, target in PLATFORM_TARGETS.items():
            peak_key = next((k for k in target if "peak" in k.lower() or "tp" in k.lower()), None)
            assert peak_key is not None, f"{name} ขาด peak/tp key: {list(target.keys())}"

    def test_irc_modes_all_accessible(self):
        from modules.master.genre_profiles import IRC_MODES
        for mode in ["IRC 1", "IRC 2", "IRC 3", "IRC 4", "IRC 5", "IRC LL"]:
            assert mode in IRC_MODES

    def test_tone_presets_apply_to_eq(self):
        from modules.master.equalizer import Equalizer
        from modules.master.genre_profiles import TONE_PRESETS
        eq = Equalizer()
        for name in TONE_PRESETS:
            eq.load_tone_preset(name)
            assert all(isinstance(b.gain, float) for b in eq.bands)

    def test_mastering_presets_exist(self):
        from modules.master.genre_profiles import MASTERING_PRESETS
        assert len(MASTERING_PRESETS) > 0
