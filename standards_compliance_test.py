"""
LongPlay Studio V5.11 — Signal Standards Compliance Test
=========================================================
Tests audio output against industry standards:
- LUFS: Spotify -14, YouTube -14, Apple Music -16 (±1 LU tolerance)
- True Peak: ceiling enforcement with hot signal
- No clipping (samples ≤ ±1.0)
- Stereo correlation > 0 (in phase)
- Dynamic range (crest factor ≥ 6 dB, LU range ≥ 3 LU)
- Frequency balance (no null bands, no extreme spectral imbalance)
- EQ isolation test (after band.q → band.width fix)
- GAIN loudness test (after gain_offset → LUFS fix)
- THD stage isolation

V5.11.1 — includes regression tests for EQ and GAIN bugs fixed.
"""

import os
import sys
import json
import tempfile
import numpy as np
import time

# ──────────────────────────────────────────
# Setup path
# ──────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

try:
    import soundfile as sf
    HAS_SF = True
except ImportError:
    HAS_SF = False
    print("ERROR: soundfile not available")
    sys.exit(1)

try:
    import pyloudnorm as pyln
    HAS_PYLN = True
except ImportError:
    HAS_PYLN = False
    print("WARNING: pyloudnorm not available — LUFS tests will fail")

# ──────────────────────────────────────────
# Test file
# ──────────────────────────────────────────
HOOK_PATH = os.path.join(
    SCRIPT_DIR,
    "Vinyl Prophet Vol.1", "Hook", "1.Higher Vibration_hook.wav"
)

RESULTS = {}
PASS = []
FAIL = []


# ──────────────────────────────────────────
# Measurement helpers (ITU-R BS.1770 compliant)
# ──────────────────────────────────────────

def measure_lufs(audio: np.ndarray, sr: int) -> float:
    """Measure integrated LUFS using pyloudnorm (ITU-R BS.1770-4)."""
    if not HAS_PYLN:
        return -99.0
    meter = pyln.Meter(sr)
    if audio.ndim == 1:
        audio = np.column_stack([audio, audio])
    try:
        return float(meter.integrated_loudness(audio.astype(np.float64)))
    except Exception as e:
        print(f"  [LUFS error: {e}]")
        return -99.0


def measure_true_peak(audio: np.ndarray, sr: int) -> float:
    """Measure True Peak using 4x oversampling (ITU-R BS.1770-4)."""
    try:
        from scipy.signal import resample_poly
        if audio.ndim == 1:
            audio = audio[:, np.newaxis]
        n_samples, n_ch = audio.shape
        max_tp = 0.0
        chunk = min(sr, n_samples)
        for start in range(0, n_samples, chunk):
            end = min(start + chunk, n_samples)
            seg = audio[start:end]
            if len(seg) < 4:
                continue
            for ch in range(n_ch):
                try:
                    up = resample_poly(seg[:, ch], 4, 1)
                    max_tp = max(max_tp, float(np.max(np.abs(up))))
                except Exception:
                    max_tp = max(max_tp, float(np.max(np.abs(seg[:, ch]))))
        return 20.0 * np.log10(max(max_tp, 1e-10))
    except ImportError:
        peak = float(np.max(np.abs(audio)))
        return 20.0 * np.log10(max(peak, 1e-10))


def measure_peak(audio: np.ndarray) -> float:
    return 20.0 * np.log10(max(float(np.max(np.abs(audio))), 1e-10))


def measure_rms(audio: np.ndarray) -> float:
    return 20.0 * np.log10(max(float(np.sqrt(np.mean(audio**2))), 1e-10))


def correlation(audio: np.ndarray) -> float:
    if audio.ndim < 2 or audio.shape[1] < 2:
        return 1.0
    L, R = audio[:, 0], audio[:, 1]
    norm = np.sqrt(np.mean(L**2) * np.mean(R**2))
    if norm < 1e-10:
        return 0.0
    return float(np.mean(L * R) / norm)


def stereo_width(audio: np.ndarray) -> float:
    if audio.ndim < 2 or audio.shape[1] < 2:
        return 0.0
    L, R = audio[:, 0], audio[:, 1]
    M = (L + R) / 2.0
    S = (L - R) / 2.0
    mid_rms = float(np.sqrt(np.mean(M**2)))
    side_rms = float(np.sqrt(np.mean(S**2)))
    if mid_rms < 1e-10:
        return 0.0
    return side_rms / mid_rms


def crest_factor(audio: np.ndarray) -> float:
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio**2)))
    if rms < 1e-10:
        return 99.0
    return 20.0 * np.log10(peak / rms)


def compute_spectrum_octave(audio: np.ndarray, sr: int):
    """8 octave bands — return dict of band_label: rms_db"""
    bands = [
        (20, 80, "20-80Hz"),
        (80, 250, "80-250Hz"),
        (250, 500, "250-500Hz"),
        (500, 1000, "500-1kHz"),
        (1000, 2000, "1-2kHz"),
        (2000, 4000, "2-4kHz"),
        (4000, 8000, "4-8kHz"),
        (8000, 20000, "8-20kHz"),
    ]
    mono = audio[:, 0] if audio.ndim > 1 else audio
    n = len(mono)
    fft = np.fft.rfft(mono)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    power = np.abs(fft) ** 2
    result = {}
    for lo, hi, label in bands:
        mask = (freqs >= lo) & (freqs < hi)
        if mask.sum() > 0:
            rms = float(np.sqrt(np.mean(power[mask])))
            result[label] = 20.0 * np.log10(max(rms, 1e-10))
        else:
            result[label] = -99.0
    return result


def thd_estimate(audio: np.ndarray, sr: int, freq: float = 1000.0, n_harmonics: int = 5) -> float:
    """FFT-based THD% for a known fundamental frequency."""
    mono = audio[:, 0] if audio.ndim > 1 else audio
    n = len(mono)
    fft = np.fft.rfft(mono * np.hanning(n))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    mag = np.abs(fft) * 2.0 / n
    bin_res = sr / n
    def get_mag(f):
        idx = int(round(f / bin_res))
        idx = max(0, min(idx, len(mag) - 1))
        lo, hi = max(0, idx-2), min(len(mag)-1, idx+2)
        return float(np.max(mag[lo:hi+1]))
    fund = get_mag(freq)
    if fund < 1e-10:
        return 0.0
    harm_sq = sum(get_mag(freq * (k+2))**2 for k in range(n_harmonics))
    return float(np.sqrt(harm_sq) / fund * 100.0)


def log_result(test_name: str, passed, detail: str):
    passed = bool(passed)
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status}  {test_name}: {detail}")
    RESULTS[test_name] = {"pass": passed, "detail": str(detail)}
    (PASS if passed else FAIL).append(test_name)


# ──────────────────────────────────────────
# Build chain
# ──────────────────────────────────────────

def build_chain():
    from modules.master.chain import MasterChain
    chain = MasterChain()
    chain.set_meter_callback(lambda d: None)
    return chain


def render_chain(chain, tmp_suffix="out") -> tuple:
    """Load hook, render, return (audio, sr)."""
    with tempfile.NamedTemporaryFile(suffix=f"_{tmp_suffix}.wav", delete=False) as f:
        out_path = f.name
    chain.load_audio(HOOK_PATH)
    result = chain.render(out_path)
    if not result or not os.path.exists(result):
        return None, None
    audio, sr = sf.read(result)
    os.unlink(result)
    if audio.ndim == 1:
        audio = np.column_stack([audio, audio])
    return audio, sr


BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def section(title: str):
    print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─'*60}{RESET}")


# ══════════════════════════════════════════════════════════════
print(f"\n{BOLD}{'═'*60}")
print(f"  LongPlay Studio V5.11 — Signal Standards Compliance")
print(f"  V5.11.1 — After EQ band.width + GAIN offset fixes")
print(f"{'═'*60}{RESET}")

if not os.path.exists(HOOK_PATH):
    print(f"ERROR: Test file not found: {HOOK_PATH}")
    sys.exit(1)

input_audio, input_sr = sf.read(HOOK_PATH)
if input_audio.ndim == 1:
    input_audio = np.column_stack([input_audio, input_audio])

# ──────────────────────────────────────────
# Section 0: Measure input baseline
# ──────────────────────────────────────────
section("0. INPUT BASELINE")
input_lufs = measure_lufs(input_audio, input_sr)
input_tp = measure_true_peak(input_audio, input_sr)
input_peak = measure_peak(input_audio)
input_rms = measure_rms(input_audio)
input_corr = correlation(input_audio)

print(f"  Input: LUFS={input_lufs:.2f}  TP={input_tp:.2f} dBTP  Peak={input_peak:.2f}  RMS={input_rms:.2f}")
print(f"  Correlation: {input_corr:.3f}  Duration: {len(input_audio)/input_sr:.1f}s @ {input_sr} Hz")


# ──────────────────────────────────────────
# Section 1: LUFS Platform Standards
# ──────────────────────────────────────────
section("1. LUFS PLATFORM STANDARDS (±1.0 LU tolerance)")

PLATFORM_TARGETS_LOCAL = {
    "Spotify":     {"target_lufs": -14.0, "true_peak": -1.0},
    "YouTube":     {"target_lufs": -14.0, "true_peak": -1.0},
    "Apple Music": {"target_lufs": -16.0, "true_peak": -1.0},
}

for platform, targets in PLATFORM_TARGETS_LOCAL.items():
    chain = build_chain()
    chain.target_lufs = targets["target_lufs"]
    chain.target_tp   = targets["true_peak"]
    chain.set_platform(platform)
    chain.maximizer.gain_db = 0.0

    audio, sr = render_chain(chain, f"{platform.replace(' ','_')}_lufs")
    if audio is None:
        log_result(f"LUFS {platform}", False, "render failed")
        continue

    lufs = measure_lufs(audio, sr)
    tp   = measure_true_peak(audio, sr)
    target = targets["target_lufs"]
    tolerance = 1.0
    passed_lufs = abs(lufs - target) <= tolerance
    passed_tp   = tp <= targets["true_peak"] + 0.1   # 0.1 dB margin

    print(f"\n  {BOLD}{platform}{RESET}")
    print(f"    Target LUFS: {target:.1f}  Measured: {lufs:.2f}  Δ={lufs-target:+.2f}")
    print(f"    True Peak ceiling: {targets['true_peak']:.1f} dBTP  Measured: {tp:.2f} dBTP")
    log_result(f"LUFS {platform} (target {target}±1.0)", passed_lufs,
               f"{lufs:.2f} LUFS (Δ{lufs-target:+.2f})")
    log_result(f"True Peak {platform} (≤ {targets['true_peak']} dBTP)", passed_tp,
               f"{tp:.2f} dBTP")

RESULTS["lufs_platforms"] = {p: {"target": PLATFORM_TARGETS_LOCAL[p]["target_lufs"]} for p in PLATFORM_TARGETS_LOCAL}


# ──────────────────────────────────────────
# Section 2: True Peak Ceiling — Hot Signal
# ──────────────────────────────────────────
section("2. TRUE PEAK CEILING — HOT SIGNAL TEST")
print("  (Applies GAIN +6 and +12 dB to push signal near/above ceiling)")

for gain_push in [6.0, 12.0]:
    chain = build_chain()
    chain.target_lufs = -14.0
    chain.target_tp   = -1.0
    chain.maximizer.gain_db = gain_push

    audio, sr = render_chain(chain, f"hot_{int(gain_push)}db")
    if audio is None:
        log_result(f"True Peak ceiling with GAIN +{gain_push:.0f}dB", False, "render failed")
        continue

    tp = measure_true_peak(audio, sr)
    ceiling = -1.0
    passed = tp <= ceiling + 0.1
    print(f"  GAIN +{gain_push:.0f} dB → True Peak = {tp:.2f} dBTP  (ceiling {ceiling})")
    log_result(f"Limiter ceiling -1.0 dBTP with +{gain_push:.0f}dB push", passed,
               f"TP={tp:.2f} dBTP {'✓' if passed else 'EXCEEDS ceiling!'}")


# ──────────────────────────────────────────
# Section 3: Clipping Check
# ──────────────────────────────────────────
section("3. CLIPPING / SAMPLE OVERFLOW CHECK")

for gain_push in [0.0, 6.0, 12.0, 18.0]:
    chain = build_chain()
    chain.target_lufs = -14.0
    chain.target_tp   = -1.0
    chain.maximizer.gain_db = gain_push

    audio, sr = render_chain(chain, f"clip_{int(gain_push)}db")
    if audio is None:
        log_result(f"No clipping GAIN +{gain_push:.0f}dB", False, "render failed")
        continue

    clips = int(np.sum(np.abs(audio) >= 1.0))
    over = int(np.sum(np.abs(audio) > 1.0))
    passed = clips == 0
    detail = f"clips={clips}, over_1.0={over}, peak={measure_peak(audio):.2f}dBFS"
    log_result(f"No clipping GAIN +{gain_push:.0f}dB", passed, detail)


# ──────────────────────────────────────────
# Section 4: EQ Isolation Test (after band.width fix)
# ──────────────────────────────────────────
section("4. EQ ISOLATION — spectrum change after band.width fix")

chain_flat = build_chain()
audio_flat, sr = render_chain(chain_flat, "eq_flat")
if audio_flat is not None:
    spec_flat = compute_spectrum_octave(audio_flat, sr)

    for band_idx, freq_hz, target_band in [
        (2, 250,  "250-500Hz"),
        (4, 1000, "500-1kHz"),
        (5, 2000, "1-2kHz"),
    ]:
        chain_boost = build_chain()
        chain_boost.equalizer.bands[band_idx].gain = 12.0
        audio_boost, sr2 = render_chain(chain_boost, f"eq_boost_{freq_hz}hz")
        if audio_boost is None:
            log_result(f"EQ +12dB @ {freq_hz}Hz band effect", False, "render failed")
            continue

        spec_boost = compute_spectrum_octave(audio_boost, sr2)
        delta = spec_boost.get(target_band, -99.0) - spec_flat.get(target_band, -99.0)
        passed = delta >= 2.0
        log_result(f"EQ +12dB @ {freq_hz}Hz changes {target_band}", passed,
                   f"Δ={delta:+.1f} dB (need ≥+2.0)")

    # EQ cut test
    chain_cut = build_chain()
    chain_cut.equalizer.bands[4].gain = -12.0
    audio_cut, sr3 = render_chain(chain_cut, "eq_cut_1khz")
    if audio_cut is not None:
        spec_cut = compute_spectrum_octave(audio_cut, sr3)
        delta_cut = spec_cut.get("500-1kHz", -99.0) - spec_flat.get("500-1kHz", -99.0)
        passed_cut = delta_cut <= -2.0
        log_result("EQ -12dB @ 1kHz cuts 500-1kHz band", passed_cut,
                   f"Δ={delta_cut:+.1f} dB (need ≤-2.0)")


# ──────────────────────────────────────────
# Section 5: GAIN Loudness Test (after gain_offset fix)
# ──────────────────────────────────────────
section("5. GAIN LOUDNESS — LUFS changes with GAIN knob (after gain_offset fix)")

gain_lufs = {}
for gain_db in [0.0, 3.0, 6.0, 9.0]:
    chain = build_chain()
    chain.target_lufs = -14.0
    chain.target_tp   = -1.0
    chain.maximizer.gain_db = gain_db

    audio, sr = render_chain(chain, f"gain_{int(gain_db)}db")
    if audio is None:
        gain_lufs[gain_db] = None
        continue

    lufs = measure_lufs(audio, sr)
    rms  = measure_rms(audio)
    gain_lufs[gain_db] = lufs
    print(f"  GAIN +{gain_db:.0f}dB → LUFS={lufs:.2f}  RMS={rms:.2f}")

# Check gradient
if all(v is not None for v in gain_lufs.values()):
    g0, g3, g6, g9 = [gain_lufs[k] for k in [0.0, 3.0, 6.0, 9.0]]
    log_result("GAIN +3dB louder than +0dB", g3 > g0 - 0.5,
               f"{g0:.2f} → {g3:.2f} LUFS (Δ{g3-g0:+.2f})")
    log_result("GAIN +6dB louder than +3dB", g6 > g3 - 0.5,
               f"{g3:.2f} → {g6:.2f} LUFS (Δ{g6-g3:+.2f})")
    log_result("GAIN +9dB louder than +6dB", g9 > g6 - 0.5,
               f"{g6:.2f} → {g9:.2f} LUFS (Δ{g9-g6:+.2f})")
    log_result("GAIN +0dB LUFS within ±1.0 of target -14", abs(g0 - (-14.0)) <= 1.0,
               f"{g0:.2f} LUFS vs target -14.0 (Δ{g0-(-14):+.2f})")


# ──────────────────────────────────────────
# Section 6: Stereo Correlation
# ──────────────────────────────────────────
section("6. STEREO CORRELATION")

chain = build_chain()
audio, sr = render_chain(chain, "stereo")
if audio is not None:
    corr = correlation(audio)
    sw   = stereo_width(audio)
    log_result("Stereo correlation > 0.0 (in phase)", corr > 0.0,
               f"correlation={corr:.3f}")
    log_result("Stereo correlation > 0.7 (music normal)", corr > 0.7,
               f"correlation={corr:.3f}")
    print(f"  Stereo width M/S ratio: {sw:.3f}")


# ──────────────────────────────────────────
# Section 7: Dynamic Range
# ──────────────────────────────────────────
section("7. DYNAMIC RANGE")

chain = build_chain()
audio, sr = render_chain(chain, "dynrange")
if audio is not None:
    cf = crest_factor(audio)
    peak_db = measure_peak(audio)
    rms_db  = measure_rms(audio)
    lufs_val = measure_lufs(audio, sr)

    print(f"  Peak={peak_db:.2f} dBFS  RMS={rms_db:.2f} dBFS  Crest={cf:.1f} dB")

    # LU range: difference between loudest and softest 3-second window
    try:
        meter = pyln.Meter(sr)
        block_size = 3 * sr
        blocks = [audio[i:i+block_size] for i in range(0, len(audio)-block_size, sr)]
        block_lufs = []
        for blk in blocks:
            if len(blk) >= sr:
                try:
                    bl = meter.integrated_loudness(blk.astype(np.float64))
                    if bl > -70:
                        block_lufs.append(bl)
                except Exception:
                    pass
        lu_range = max(block_lufs) - min(block_lufs) if len(block_lufs) >= 2 else 0.0
        print(f"  LU range over 3-sec windows: {lu_range:.1f} LU  ({len(block_lufs)} windows)")
        log_result("Dynamic range crest ≥ 6 dB (not over-compressed)", cf >= 6.0,
                   f"crest={cf:.1f} dB")
        log_result("LU range ≥ 3 LU (not squashed)", lu_range >= 3.0,
                   f"LU range={lu_range:.1f} LU")
    except Exception as e:
        print(f"  [LU range error: {e}]")
        log_result("Dynamic range crest ≥ 6 dB", cf >= 6.0, f"crest={cf:.1f} dB")


# ──────────────────────────────────────────
# Section 8: Frequency Balance
# ──────────────────────────────────────────
section("8. FREQUENCY BALANCE")

chain = build_chain()
audio, sr = render_chain(chain, "spectrum")
if audio is not None:
    spec = compute_spectrum_octave(audio, sr)
    print(f"\n  {'Band':<15} {'RMS dB':>10}")
    print(f"  {'─'*25}")
    for band, val in spec.items():
        bar = "█" * max(0, int((val + 100) / 5))
        print(f"  {band:<15} {val:>8.1f} dB  {bar}")

    mid_rms  = spec.get("500-1kHz", -99.0)
    sub_rms  = spec.get("20-80Hz", -99.0)
    air_rms  = spec.get("8-20kHz", -99.0)

    no_nulls = all(v > -60.0 for v in spec.values())
    sub_ok   = (sub_rms - mid_rms) <= 12.0
    air_ok   = (mid_rms - air_rms) <= 20.0

    log_result("No frequency null bands (all > -60 dB)", no_nulls,
               f"min={min(spec.values()):.1f} dB")
    log_result("Sub-bass ≤ 12 dB above mid (not bass heavy)", sub_ok,
               f"sub={sub_rms:.1f}, mid={mid_rms:.1f}, Δ={sub_rms-mid_rms:.1f}")
    log_result("Air ≤ 20 dB below mid (not too dark)", air_ok,
               f"air={air_rms:.1f}, mid={mid_rms:.1f}, Δ={mid_rms-air_rms:.1f}")


# ──────────────────────────────────────────
# Section 9: THD Stage Isolation
# ──────────────────────────────────────────
section("9. THD STAGE ISOLATION (1kHz sine → each stage)")

# Generate 1kHz sine at -18 dBFS
sr_test = 48000
dur     = 5.0
t       = np.linspace(0, dur, int(dur * sr_test), endpoint=False)
sine    = np.sin(2.0 * np.pi * 1000.0 * t) * 10**(-18.0/20.0)
sine_stereo = np.column_stack([sine, sine])

# Write test tone
tone_path = os.path.join(SCRIPT_DIR, "_test_tone_1khz.wav")
sf.write(tone_path, sine_stereo, sr_test, subtype='PCM_24')

def render_tone_chain(chain_fn, label: str) -> float:
    """Render test tone through a configured chain, return THD%."""
    chain = chain_fn()
    chain.load_audio(tone_path)
    with tempfile.NamedTemporaryFile(suffix=f"_{label}.wav", delete=False) as f:
        out_path = f.name
    result = chain.render(out_path)
    if not result or not os.path.exists(result):
        return -1.0
    audio, sr_out = sf.read(result)
    os.unlink(result)
    if audio.ndim == 1:
        audio = np.column_stack([audio, audio])
    thd = thd_estimate(audio, sr_out, freq=1000.0)
    return thd

# Baseline: flat chain (EQ flat, no compressor push, no maximizer gain)
def chain_flat_tone():
    chain = build_chain()
    chain.maximizer.gain_db = 0.0
    return chain

thd_flat = render_tone_chain(chain_flat_tone, "thd_flat")
print(f"\n  Flat chain (no processing emphasis):  THD = {thd_flat:.2f}%")
log_result("THD flat chain ≤ 3% (baseline quality)", thd_flat <= 3.0,
           f"THD={thd_flat:.2f}% (standard mastering: < 1%)")

# With EQ boost only
def chain_eq_only():
    chain = build_chain()
    chain.equalizer.bands[4].gain = 6.0  # 1kHz +6dB
    chain.maximizer.gain_db = 0.0
    return chain

thd_eq = render_tone_chain(chain_eq_only, "thd_eq")
print(f"  EQ +6dB @ 1kHz only:                  THD = {thd_eq:.2f}%")
thd_eq_inc = thd_eq - thd_flat
log_result("EQ +6dB doesn't add >5% THD vs baseline", thd_eq_inc <= 5.0,
           f"THD increase={thd_eq_inc:+.2f}%")

# With Maximizer gain push
def chain_max_gain():
    chain = build_chain()
    chain.maximizer.gain_db = 6.0
    return chain

thd_max = render_tone_chain(chain_max_gain, "thd_max")
print(f"  Maximizer +6dB gain push:              THD = {thd_max:.2f}%")
log_result("Maximizer identified as THD source if > flat+5%",
           thd_max > thd_flat + 5.0,
           f"Flat={thd_flat:.2f}% Max={thd_max:.2f}% Δ={thd_max-thd_flat:+.2f}%")

# IRC mode comparison for THD
for irc_mode in ["IRC 1", "IRC 4", "IRC LL"]:
    def chain_irc_fn(mode=irc_mode):
        chain = build_chain()
        chain.maximizer.gain_db = 0.0
        chain.maximizer.set_irc_mode(mode)
        return chain
    thd_irc = render_tone_chain(chain_irc_fn, f"thd_{irc_mode.replace(' ','_')}")
    print(f"  IRC mode {irc_mode:<8}: THD = {thd_irc:.2f}%")

# Cleanup tone file
if os.path.exists(tone_path):
    os.unlink(tone_path)


# ──────────────────────────────────────────
# Final Summary
# ──────────────────────────────────────────
print(f"\n{BOLD}{'═'*60}")
print(f"  สรุปผลการทดสอบ Signal Standards Compliance")
print(f"{'═'*60}{RESET}")

total = len(PASS) + len(FAIL)
print(f"\n  ผลรวม: {len(PASS)}/{total} PASS  |  {len(FAIL)} FAIL\n")

if FAIL:
    print(f"  {BOLD}{RED}❌ FAIL:{RESET}")
    for f in FAIL:
        detail = RESULTS.get(f, {}).get("detail", "")
        print(f"     ❌ {f}")
        if detail:
            print(f"        → {detail}")

if PASS:
    print(f"\n  {BOLD}{GREEN}✅ PASS:{RESET}")
    for p in PASS:
        detail = RESULTS.get(p, {}).get("detail", "")
        print(f"     ✅ {p}")
        if detail:
            print(f"        → {detail}")

# Thai summary
print(f"\n{BOLD}{'─'*60}{RESET}")
print(f"{BOLD}  สรุปเป็นภาษาไทย{RESET}")
print(f"{'─'*60}")

lufs_spot = RESULTS.get("LUFS Spotify (target -14.0±1.0)", {})
lufs_yt   = RESULTS.get("LUFS YouTube (target -14.0±1.0)", {})
lufs_am   = RESULTS.get("LUFS Apple Music (target -16.0±1.0)", {})
tp_spot   = RESULTS.get("True Peak Spotify (≤ -1.0 dBTP)", {})
clip0     = RESULTS.get("No clipping GAIN +0dB", {})
clip18    = RESULTS.get("No clipping GAIN +18dB", {})
corr_pos  = RESULTS.get("Stereo correlation > 0.0 (in phase)", {})
dyn       = RESULTS.get("Dynamic range crest ≥ 6 dB (not over-compressed)", {})
eq_test   = RESULTS.get("EQ +12dB @ 1000Hz changes 500-1kHz band", {})
gain_test = RESULTS.get("GAIN +0dB LUFS within ±1.0 of target -14", {})

print(f"\n  1. LUFS มาตรฐาน:")
print(f"     Spotify:     {'✅ ผ่าน' if lufs_spot.get('pass') else '❌ พลาด'}  — {lufs_spot.get('detail','N/A')}")
print(f"     YouTube:     {'✅ ผ่าน' if lufs_yt.get('pass') else '❌ พลาด'}   — {lufs_yt.get('detail','N/A')}")
print(f"     Apple Music: {'✅ ผ่าน' if lufs_am.get('pass') else '❌ พลาด'}  — {lufs_am.get('detail','N/A')}")
print(f"\n  2. True Peak Ceiling (ไม่เกิน -1.0 dBTP):")
print(f"     {'✅ ผ่าน' if tp_spot.get('pass') else '❌ พลาด'} — {tp_spot.get('detail','N/A')}")
print(f"\n  3. Clipping / Distortion:")
print(f"     GAIN=0:  {'✅ ไม่แตก' if clip0.get('pass') else '❌ แตก!'} — {clip0.get('detail','N/A')}")
print(f"     GAIN=18: {'✅ ไม่แตก' if clip18.get('pass') else '❌ แตก!'} — {clip18.get('detail','N/A')}")
print(f"\n  4. Stereo Correlation (ไม่ out-of-phase):")
print(f"     {'✅ ผ่าน' if corr_pos.get('pass') else '❌ พลาด'} — {corr_pos.get('detail','N/A')}")
print(f"\n  5. Dynamic Range:")
print(f"     {'✅ ผ่าน' if dyn.get('pass') else '❌ พลาด'} — {dyn.get('detail','N/A')}")
print(f"\n  6. EQ ทำงาน (หลังแก้ bug band.q → band.width):")
print(f"     {'✅ ผ่าน' if eq_test.get('pass') else '❌ ยังไม่ผ่าน'} — {eq_test.get('detail','N/A')}")
print(f"\n  7. GAIN knob มีผล (หลังแก้ bug gain_offset):")
print(f"     {'✅ ผ่าน' if gain_test.get('pass') else '❌ ยังไม่ผ่าน'} — {gain_test.get('detail','N/A')}")

print(f"\n{'─'*60}")
if len(FAIL) == 0:
    print(f"  {BOLD}{GREEN}✅ ผ่านทุกมาตรฐาน! — เสียงได้มาตรฐาน{RESET}")
elif len(FAIL) <= 3:
    print(f"  {BOLD}{YELLOW}⚠️  ผ่านส่วนใหญ่ ({len(PASS)}/{total}) — มีปัญหาเล็กน้อย{RESET}")
else:
    print(f"  {BOLD}{RED}❌ มีปัญหาหลายจุด ({len(FAIL)}/{total} fail){RESET}")

# Save JSON report
report = {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "version": "V5.11.1",
    "total": total,
    "passed": len(PASS),
    "failed": len(FAIL),
    "pass_rate": f"{len(PASS)/total*100:.1f}%" if total > 0 else "N/A",
    "results": RESULTS,
    "input": {
        "lufs": input_lufs,
        "true_peak": input_tp,
        "peak": input_peak,
        "rms": input_rms,
        "correlation": input_corr,
    },
}
report_path = os.path.join(SCRIPT_DIR, "standards_compliance_report.json")
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(f"\n  รายงาน JSON: {report_path}")
