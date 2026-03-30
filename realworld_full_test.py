"""
LongPlay Studio V5.11.1 — Full Real-World Feature Test
=======================================================
Tests all major features with real WAV/MP4 files.
Simulates "opening the app and doing real work".

Tests:
  1. Platform mastering (Spotify/YouTube/Apple Music) + LUFS verification
  2. EQ real effect (bass boost/cut, high boost, flat preset)
  3. Compressor preset comparison (Gentle Glue / Punchy / Aggressive)
  4. Stereo Imager width test (0% / 100% / 200%)
  5. Maximizer IRC modes (IRC 1 / 3 / 4 / LL)
  6. Video import (audio extraction from MP4)
  7. Batch processing (3 songs simultaneously)
  8. Save/Load project settings
  9. Realtime meter callback during render
 10. Multi-file consistency (3 songs → same LUFS target)
 11. Output quality gate (no clips, correct duration/SR)
"""

import os, sys, json, time, shutil, tempfile, subprocess
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import soundfile as sf
import pyloudnorm as pyln

# ─── File paths ───────────────────────────────────────────
HOOK_DIR   = os.path.join(SCRIPT_DIR, "Vinyl Prophet Vol.1", "Hook")
VDO_DIR    = os.path.join(SCRIPT_DIR, "Vinyl Prophet Vol.1", "vdo")
OUT_DIR    = os.path.join(SCRIPT_DIR, "_realworld_test_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

HOOK1 = os.path.join(HOOK_DIR, "1.Higher Vibration_hook.wav")
HOOK2 = os.path.join(HOOK_DIR, "2.Mercy (A Brighter Day)_hook.wav")
HOOK3 = os.path.join(HOOK_DIR, "3.Golden Vision_hook.wav")
# Use test_audio.mp4 (has audio track). The vdo/1-3.mp4 files are video-only
# (muted promotional clips). test_audio.mp4 is created by the test setup for
# a proper video-import real-world test.
MP4_1 = os.path.join(VDO_DIR, "test_audio.mp4")
if not os.path.exists(MP4_1):
    MP4_1 = os.path.join(VDO_DIR, "1.mp4")  # fallback to original

# ─── Helpers ──────────────────────────────────────────────
R = "\033[0m"; BOLD="\033[1m"; G="\033[92m"; RD="\033[91m"; CY="\033[96m"; YL="\033[93m"

RESULTS = {}   # {test_key: {pass, measured, expected, detail}}
ALL_OUTPUTS = []  # track output files for quality gate

def sec(title):
    print(f"\n{BOLD}{CY}{'─'*62}{R}\n{BOLD}{CY}  {title}{R}\n{BOLD}{CY}{'─'*62}{R}")

def log(key, passed, measured="", expected="", detail=""):
    passed = bool(passed)
    sym = f"{G}✅{R}" if passed else f"{RD}❌{R}"
    msg = f"  {sym} {key}"
    if measured:
        msg += f"  →  {measured}"
    if expected and not passed:
        msg += f"  (ต้องการ: {expected})"
    print(msg)
    if detail:
        print(f"       {detail}")
    RESULTS[key] = {"pass": passed, "measured": str(measured), "expected": str(expected)}

def build_chain(meter_events=None):
    from modules.master.chain import MasterChain
    chain = MasterChain()
    events = [] if meter_events is None else meter_events
    chain.set_meter_callback(lambda d: events.append(d))
    return chain, events

def render(chain, src, name):
    out = os.path.join(OUT_DIR, name)
    chain.load_audio(src)
    result = chain.render(out)
    if result and os.path.exists(result):
        ALL_OUTPUTS.append(result)
    return result

def load(path):
    if not path or not os.path.exists(path): return None, None
    a, sr = sf.read(path)
    if a.ndim == 1: a = np.column_stack([a, a])
    return a, sr

def lufs(audio, sr):
    try:
        m = pyln.Meter(sr)
        return float(m.integrated_loudness(audio.astype(np.float64)))
    except: return -99.0

def true_peak(audio, sr):
    try:
        from scipy.signal import resample_poly
        a = audio[:, 0] if audio.ndim > 1 else audio
        n = len(a); chunk = min(sr, n); mx = 0.0
        for s in range(0, n, chunk):
            seg = a[s:min(s+chunk, n)]
            if len(seg) < 4: continue
            up = resample_poly(seg, 4, 1)
            mx = max(mx, float(np.max(np.abs(up))))
        return 20*np.log10(max(mx, 1e-10))
    except:
        return 20*np.log10(max(float(np.max(np.abs(audio))), 1e-10))

def peak(audio):      return float(np.max(np.abs(audio)))
def rms_db(audio):    return 20*np.log10(max(float(np.sqrt(np.mean(audio**2))), 1e-10))
def crest(audio):     return 20*np.log10(max(peak(audio)/max(float(np.sqrt(np.mean(audio**2))),1e-10),1e-10))
def corr(audio):
    if audio.ndim < 2: return 1.0
    L, R = audio[:,0], audio[:,1]
    n = np.sqrt(np.mean(L**2)*np.mean(R**2))
    return float(np.mean(L*R)/n) if n > 1e-10 else 0.0
def width(audio):
    if audio.ndim < 2: return 0.0
    M = (audio[:,0]+audio[:,1])/2; S = (audio[:,0]-audio[:,1])/2
    mr = float(np.sqrt(np.mean(M**2))); sr2 = float(np.sqrt(np.mean(S**2)))
    return sr2/mr if mr > 1e-10 else 0.0

def octave_rms(audio, sr, lo, hi):
    mono = audio[:,0] if audio.ndim>1 else audio
    n = len(mono); fft = np.fft.rfft(mono); freqs = np.fft.rfftfreq(n, 1.0/sr)
    mask = (freqs >= lo) & (freqs < hi); power = np.abs(fft)**2
    return 20*np.log10(max(float(np.sqrt(np.mean(power[mask]))),1e-10)) if mask.sum()>0 else -99.0

# ══════════════════════════════════════════════════════════
print(f"\n{BOLD}{'═'*62}")
print(f"  LongPlay Studio V5.11.1 — Full Real-World Feature Test")
print(f"  ไฟล์ทดสอบ: Vinyl Prophet Vol.1 ({len(os.listdir(HOOK_DIR))} WAV files)")
print(f"{'═'*62}{R}")

# Verify input files
for f in [HOOK1, HOOK2, HOOK3]:
    a, sr = sf.read(f)
    print(f"  {os.path.basename(f)}: {len(a)/sr:.1f}s  @{sr}Hz  {a.shape}")

# ──────────────────────────────────────────────────────────
# TEST 1: Platform Mastering + LUFS Verification
# ──────────────────────────────────────────────────────────
sec("TEST 1: Platform Mastering + LUFS Verification")

platform_cfg = {
    "Spotify":     (-14.0, -1.0),
    "YouTube":     (-14.0, -1.0),
    "Apple Music": (-16.0, -1.0),
}
meter_lufs_by_platform = {}

for platform, (tgt_lufs, tgt_tp) in platform_cfg.items():
    events = []
    chain, events = build_chain(events)
    chain.set_platform(platform)
    chain.target_lufs = tgt_lufs
    chain.target_tp   = tgt_tp

    out_path = render(chain, HOOK1, f"t1_{platform.replace(' ','_')}.wav")
    audio, sr = load(out_path)

    if audio is None:
        log(f"T1 {platform} render", False, "render failed")
        continue

    size_mb = os.path.getsize(out_path) / 1e6
    measured_lufs = lufs(audio, sr)
    measured_tp   = true_peak(audio, sr)
    delta = measured_lufs - tgt_lufs

    # Internal meter LUFS (last callback event)
    if events:
        last_ev = events[-1] if isinstance(events[-1], dict) else {}
        int_lufs = last_ev.get("lufs_integrated", last_ev.get("lufs", -99.0))
        meter_lufs_by_platform[platform] = float(int_lufs)
    else:
        meter_lufs_by_platform[platform] = -99.0

    print(f"\n  {BOLD}{platform}{R}  target={tgt_lufs} LUFS")

    log(f"T1 {platform} WAV exported",   size_mb > 0.5, f"{size_mb:.1f} MB", "> 0.5MB")
    log(f"T1 {platform} LUFS ±0.5",      abs(delta) <= 0.5,
        f"{measured_lufs:.2f} LUFS (Δ{delta:+.2f})", f"{tgt_lufs}±0.5")
    log(f"T1 {platform} TruePeak ≤ {tgt_tp}", measured_tp <= tgt_tp + 0.1,
        f"{measured_tp:.2f} dBTP", f"≤{tgt_tp}")

    # Compare internal vs pyloudnorm
    if meter_lufs_by_platform[platform] != -99.0:
        diff = abs(meter_lufs_by_platform[platform] - measured_lufs)
        log(f"T1 {platform} meter vs pyloudnorm ±1.0 LU", diff <= 1.0,
            f"meter={meter_lufs_by_platform[platform]:.2f} vs pyln={measured_lufs:.2f} (Δ{diff:.2f})", "Δ≤1.0")

# ──────────────────────────────────────────────────────────
# TEST 2: EQ Real Effect
# ──────────────────────────────────────────────────────────
sec("TEST 2: EQ Real Effect")

# Flat reference
chain_flat, _ = build_chain()
out_flat = render(chain_flat, HOOK1, "t2_eq_flat.wav")
audio_flat, sr_flat = load(out_flat)

tests_eq = [
    (0, +6.0, (20, 80),      "20-80Hz",   "Bass +6 → sub band up"),
    (0, -6.0, (20, 80),      "20-80Hz",   "Bass −6 → sub band down"),
    # Band 6 = 8kHz highshelf (DEFAULT_FREQS[6]=8000). Measure 6-12kHz to match the shelf.
    # Previously used band 7 (16kHz lowpass) but measured 4-8kHz — wrong window → Δ≈0.
    (6, +6.0, (6000, 12000), "6-12kHz",  "High +6 → air band up (band6=8kHz highshelf)"),
]

for band_idx, gain_val, (lo, hi), band_label, desc in tests_eq:
    chain_eq, _ = build_chain()
    chain_eq.equalizer.bands[band_idx].gain = gain_val
    out_eq = render(chain_eq, HOOK1, f"t2_eq_b{band_idx}_g{int(gain_val)}.wav")
    audio_eq, sr_eq = load(out_eq)

    if audio_eq is None:
        log(f"T2 EQ {desc}", False, "render failed"); continue

    rms_flat = octave_rms(audio_flat, sr_flat, lo, hi)
    rms_eq   = octave_rms(audio_eq,   sr_eq,   lo, hi)
    delta    = rms_eq - rms_flat
    direction = "up" if gain_val > 0 else "down"
    passed = (delta > 1.5) if gain_val > 0 else (delta < -1.5)
    log(f"T2 EQ {desc}", passed,
        f"{rms_flat:.1f}→{rms_eq:.1f} dB (Δ{delta:+.1f})", f"Δ{'>+1.5' if gain_val>0 else '<-1.5'}")

# EQ Flat preset
chain_flat_preset, _ = build_chain()
chain_flat_preset.equalizer.load_tone_preset("Flat")
out_fp = render(chain_flat_preset, HOOK1, "t2_eq_flat_preset.wav")
audio_fp, sr_fp = load(out_fp)
if audio_fp is not None and audio_flat is not None:
    # Compare octave bands 250-8kHz
    max_diff = 0.0
    for lo2, hi2 in [(250,500),(500,1000),(1000,2000),(2000,4000),(4000,8000)]:
        d = abs(octave_rms(audio_fp, sr_fp, lo2, hi2) - octave_rms(audio_flat, sr_flat, lo2, hi2))
        max_diff = max(max_diff, d)
    log("T2 EQ Flat preset matches no-preset ±0.5dB", max_diff <= 0.5,
        f"max_diff={max_diff:.2f}dB", "≤0.5dB")

# ──────────────────────────────────────────────────────────
# TEST 3: Compressor Preset Comparison
# ──────────────────────────────────────────────────────────
sec("TEST 3: Compressor Preset Comparison")

preset_crest = {}
for preset_name in ["Gentle Glue", "Punchy", "Aggressive"]:
    chain_c, _ = build_chain()
    try:
        chain_c.dynamics.load_preset(preset_name)
    except Exception as e:
        log(f"T3 {preset_name} load", False, str(e)); continue

    out_c = render(chain_c, HOOK1, f"t3_comp_{preset_name.replace(' ','_')}.wav")
    audio_c, sr_c = load(out_c)
    if audio_c is None:
        log(f"T3 {preset_name} render", False, "failed"); continue

    cf = crest(audio_c)
    preset_crest[preset_name] = cf
    log(f"T3 {preset_name} renders OK", True, f"crest={cf:.1f}dB")

# Aggressive should have less crest (more squashed) than Gentle Glue
if "Gentle Glue" in preset_crest and "Aggressive" in preset_crest:
    gg = preset_crest["Gentle Glue"]
    ag = preset_crest["Aggressive"]
    log("T3 Aggressive ≤ Gentle Glue crest (more compressed)", ag <= gg + 1.0,
        f"Aggressive={ag:.1f}  GentleGlue={gg:.1f}", "Aggressive≤GentleGlue+1")

# ──────────────────────────────────────────────────────────
# TEST 4: Stereo Imager Real Effect
# ──────────────────────────────────────────────────────────
sec("TEST 4: Stereo Imager Real Effect")

input_audio, input_sr = sf.read(HOOK1)
if input_audio.ndim == 1: input_audio = np.column_stack([input_audio, input_audio])
input_corr  = corr(input_audio)
input_width = width(input_audio)

width_results = {}
for w_pct in [0, 100, 200]:
    chain_i, _ = build_chain()
    chain_i.imager.width = w_pct
    out_i = render(chain_i, HOOK1, f"t4_imager_{w_pct}pct.wav")
    audio_i, sr_i = load(out_i)
    if audio_i is None:
        log(f"T4 Width={w_pct}% render", False, "failed"); continue
    c = corr(audio_i)
    w = width(audio_i)
    width_results[w_pct] = {"corr": c, "width": w}
    print(f"  Width={w_pct}%:  corr={c:.3f}  stereo_width={w:.3f}")

if 0 in width_results:
    log("T4 Width=0% → mono (corr ≥ 0.99)", width_results[0]["corr"] >= 0.99,
        f"corr={width_results[0]['corr']:.3f}", "≥0.99")

if 100 in width_results:
    c100 = width_results[100]["corr"]
    log("T4 Width=100% ≈ input correlation (±0.05)", abs(c100 - input_corr) <= 0.05,
        f"corr={c100:.3f}  input={input_corr:.3f}", "Δ≤0.05")

if 100 in width_results and 200 in width_results:
    log("T4 Width=200% wider than 100%", width_results[200]["width"] > width_results[100]["width"],
        f"200%={width_results[200]['width']:.3f}  100%={width_results[100]['width']:.3f}", "200%>100%")

if 0 in width_results and 100 in width_results and 200 in width_results:
    w0  = width_results[0]["width"]
    w100= width_results[100]["width"]
    w200= width_results[200]["width"]
    log("T4 Width gradient 0%<100%<200%", w0 < w100 < w200,
        f"0%={w0:.3f}  100%={w100:.3f}  200%={w200:.3f}", "ascending")

# ──────────────────────────────────────────────────────────
# TEST 5: Maximizer IRC Modes
# ──────────────────────────────────────────────────────────
sec("TEST 5: Maximizer IRC Modes")

irc_results = {}
for irc_mode in ["IRC 1", "IRC 3", "IRC 4", "IRC LL"]:
    chain_m, _ = build_chain()
    chain_m.maximizer.set_irc_mode(irc_mode)
    out_m = render(chain_m, HOOK1, f"t5_irc_{irc_mode.replace(' ','_')}.wav")
    audio_m, sr_m = load(out_m)
    if audio_m is None:
        log(f"T5 {irc_mode} render", False, "failed"); continue

    pk   = peak(audio_m)
    clips= int(np.sum(np.abs(audio_m) >= 1.0))
    tp   = true_peak(audio_m, sr_m)
    irc_results[irc_mode] = {"peak": pk, "clips": clips, "tp": tp}
    print(f"  {irc_mode:<10}: peak={20*np.log10(max(pk,1e-10)):.2f}dBFS  TP={tp:.2f}dBTP  clips={clips}")

    log(f"T5 {irc_mode} no clips",       clips == 0,    f"clips={clips}", "0")
    log(f"T5 {irc_mode} TP ≤ −0.9 dBTP", tp <= -0.9,   f"{tp:.2f}dBTP",  "≤−0.9")

# ──────────────────────────────────────────────────────────
# TEST 6: Video Import (Audio Extract from MP4)
# ──────────────────────────────────────────────────────────
sec("TEST 6: Video Import — MP4 Audio Extraction")

mp4_exists = os.path.exists(MP4_1)
mp4_size   = os.path.getsize(MP4_1) / 1e6 if mp4_exists else 0

log("T6 MP4 file exists", mp4_exists, f"{mp4_size:.1f}MB", ">0.1MB")

if mp4_exists:
    # Extract audio using ffmpeg
    out_mp4_audio = os.path.join(OUT_DIR, "t6_mp4_audio.wav")
    # Try PATH first, then known macOS homebrew/system locations
    ffmpeg = (shutil.which("ffmpeg") or
              ("/opt/homebrew/bin/ffmpeg" if os.path.exists("/opt/homebrew/bin/ffmpeg") else None) or
              ("/usr/local/bin/ffmpeg"    if os.path.exists("/usr/local/bin/ffmpeg")    else None) or
              "ffmpeg")
    try:
        result = subprocess.run(
            [ffmpeg, "-y", "-i", MP4_1, "-vn", "-acodec", "pcm_s16le",
             "-ar", "48000", out_mp4_audio],
            capture_output=True, timeout=30
        )
        audio_mp4, sr_mp4 = load(out_mp4_audio) if result.returncode == 0 else (None, None)
    except Exception as e:
        print(f"  [ffmpeg error: {e}]")
        audio_mp4, sr_mp4 = None, None

    if audio_mp4 is not None:
        dur = len(audio_mp4) / sr_mp4
        log("T6 MP4 audio extracted", True, f"{dur:.1f}s @ {sr_mp4}Hz")
        log("T6 MP4 audio duration > 1s", dur > 1.0, f"{dur:.1f}s", ">1s")
        ALL_OUTPUTS.append(out_mp4_audio)

        # Master the extracted MP4 audio
        chain_mp4, _ = build_chain()
        chain_mp4.target_lufs = -14.0
        out_mp4_master = render(chain_mp4, out_mp4_audio, "t6_mp4_mastered.wav")
        audio_mp4m, sr_mp4m = load(out_mp4_master)
        if audio_mp4m is not None:
            mp4_lufs = lufs(audio_mp4m, sr_mp4m)
            log("T6 MP4 mastered LUFS ±1.0 of −14", abs(mp4_lufs - (-14.0)) <= 1.0,
                f"{mp4_lufs:.2f} LUFS", "−14±1.0")
    else:
        log("T6 MP4 audio extracted", False, "ffmpeg failed or not available")

# ──────────────────────────────────────────────────────────
# TEST 7: Batch Processing (3 songs)
# ──────────────────────────────────────────────────────────
sec("TEST 7: Batch Processing — 3 Songs")

batch_songs = [HOOK1, HOOK2, HOOK3]
batch_outputs = []
batch_lufs_vals = []
t_batch_start = time.perf_counter()

for i, song in enumerate(batch_songs):
    chain_b, _ = build_chain()
    chain_b.target_lufs = -14.0
    out_b = render(chain_b, song, f"t7_batch_{i+1}.wav")
    batch_outputs.append(out_b)
    audio_b, sr_b = load(out_b)
    if audio_b is not None:
        batch_lufs_vals.append(lufs(audio_b, sr_b))
    else:
        batch_lufs_vals.append(None)
    print(f"  Song {i+1}: {os.path.basename(song)} → {('OK' if out_b else 'FAIL')}")

t_batch = time.perf_counter() - t_batch_start
print(f"  Total batch time: {t_batch:.1f}s")

all_ok = all(p and os.path.exists(p) and os.path.getsize(p) > 1e6 for p in batch_outputs)
log("T7 All 3 batch songs rendered > 1MB",      all_ok,
    f"{sum(1 for p in batch_outputs if p and os.path.exists(p))}/3 OK")

valid_lufs = [v for v in batch_lufs_vals if v is not None]
if valid_lufs:
    max_dev = max(abs(v - (-14.0)) for v in valid_lufs)
    log("T7 All batch LUFS within ±1.0 of −14", max_dev <= 1.0,
        f"vals={[f'{v:.2f}' for v in valid_lufs]}  max_dev={max_dev:.2f}", "max_dev≤1.0")

# ──────────────────────────────────────────────────────────
# TEST 8: Save / Load Project Settings
# ──────────────────────────────────────────────────────────
sec("TEST 8: Save / Load Project Settings")

settings_path = os.path.join(OUT_DIR, "t8_test_settings.json")

chain_save, _ = build_chain()
chain_save.target_lufs = -16.0
chain_save.target_tp   = -0.3
chain_save.equalizer.bands[3].gain = 4.5   # custom EQ
chain_save.dynamics.single_band.threshold = -20.0
chain_save.intensity = 75

saved_ok = chain_save.save_settings(settings_path)
log("T8 save_settings writes JSON", os.path.exists(settings_path) and os.path.getsize(settings_path) > 10,
    f"{os.path.getsize(settings_path)} bytes" if os.path.exists(settings_path) else "file missing")

# Load into new chain
chain_load, _ = build_chain()
load_ok = chain_load.load_settings(settings_path)
log("T8 load_settings returns True", bool(load_ok), str(load_ok))

# Verify target_lufs was restored
loaded_lufs = chain_load.target_lufs
log("T8 Loaded target_lufs matches saved −16.0", abs(loaded_lufs - (-16.0)) < 0.1,
    f"loaded={loaded_lufs}", "−16.0")

# Verify EQ band was restored
try:
    loaded_band_gain = chain_load.equalizer.bands[3].gain
    log("T8 Loaded EQ band[3] gain matches saved 4.5",
        abs(loaded_band_gain - 4.5) < 0.5,
        f"loaded={loaded_band_gain:.2f}", "4.5")
except Exception as e:
    log("T8 Loaded EQ band[3] gain", False, str(e))

# ──────────────────────────────────────────────────────────
# TEST 9: Realtime Meter Callback During Render
# ──────────────────────────────────────────────────────────
sec("TEST 9: Realtime Meter Callback During Render")

meter_events = []
chain_rt, meter_events = build_chain(meter_events)
chain_rt.target_lufs = -14.0
out_rt = render(chain_rt, HOOK1, "t9_meter_test.wav")

log("T9 Meter callback fires ≥1 event", len(meter_events) >= 1,
    f"{len(meter_events)} events")

if meter_events:
    # Get the richest event (final stage)
    best_ev = max(meter_events, key=lambda e: len(e) if isinstance(e, dict) else 0)
    if isinstance(best_ev, dict):
        print(f"  Best event keys: {list(best_ev.keys())}")

        has_lufs = "lufs_integrated" in best_ev or "lufs" in best_ev
        log("T9 Meter has lufs_integrated or lufs key", has_lufs,
            str(list(best_ev.keys())[:6]))

        rms_val = best_ev.get("left_rms_db", best_ev.get("rms_db", 0.0))
        log("T9 left_rms_db is negative (real signal)", float(rms_val) < 0,
            f"{float(rms_val):.2f} dB", "< 0")

        corr_val = best_ev.get("correlation", best_ev.get("corr", 1.0))
        log("T9 correlation between −1 and +1", -1.0 <= float(corr_val) <= 1.0,
            f"{float(corr_val):.3f}", "−1 to +1")

# ──────────────────────────────────────────────────────────
# TEST 10: Multi-file Consistency
# ──────────────────────────────────────────────────────────
sec("TEST 10: Multi-file Consistency (3 songs → same LUFS target)")

lufs_vals = {}
tp_vals   = {}
corr_vals = {}

for song, label in [(HOOK1,"Song1"), (HOOK2,"Song2"), (HOOK3,"Song3")]:
    chain_mc, _ = build_chain()
    chain_mc.target_lufs = -14.0
    chain_mc.target_tp   = -1.0
    out_mc = render(chain_mc, song, f"t10_{label}.wav")
    audio_mc, sr_mc = load(out_mc)
    if audio_mc is None: continue
    lufs_vals[label]  = lufs(audio_mc, sr_mc)
    tp_vals[label]    = true_peak(audio_mc, sr_mc)
    corr_vals[label]  = corr(audio_mc)
    print(f"  {label}: LUFS={lufs_vals[label]:.2f}  TP={tp_vals[label]:.2f}  corr={corr_vals[label]:.3f}")

if len(lufs_vals) == 3:
    all_lufs_ok  = all(abs(v - (-14.0)) <= 0.5 for v in lufs_vals.values())
    all_tp_ok    = all(v <= -0.9 for v in tp_vals.values())
    all_corr_ok  = all(v > 0.5 for v in corr_vals.values())
    log("T10 All 3 songs LUFS within ±0.5 of −14", all_lufs_ok,
        f"vals={[f'{v:.2f}' for v in lufs_vals.values()]}")
    log("T10 All 3 True Peak ≤ −0.9 dBTP", all_tp_ok,
        f"vals={[f'{v:.2f}' for v in tp_vals.values()]}")
    log("T10 All 3 stereo correlation > 0.5", all_corr_ok,
        f"vals={[f'{v:.3f}' for v in corr_vals.values()]}")

# ──────────────────────────────────────────────────────────
# TEST 11: Output Quality Gate
# ──────────────────────────────────────────────────────────
sec("TEST 11: Output Quality Gate — All Output Files")

print(f"  Checking {len(ALL_OUTPUTS)} output files...")
clip_failures = []
dur_failures  = []
sr_failures   = []

input_dur = len(sf.read(HOOK1)[0]) / sf.read(HOOK1)[1]
input_sr  = sf.read(HOOK1)[1]

for path in ALL_OUTPUTS:
    if not os.path.exists(path): continue
    audio_out, sr_out = sf.read(path)
    dur_out = len(audio_out) / sr_out

    if np.any(np.abs(audio_out) > 1.0):
        clip_failures.append(os.path.basename(path))
    if abs(dur_out - input_dur) > 0.5 and "mp4" not in path:
        dur_failures.append(f"{os.path.basename(path)}({dur_out:.1f}s)")
    if sr_out != input_sr and "mp4" not in path:
        sr_failures.append(f"{os.path.basename(path)}({sr_out}Hz)")

log("T11 No output files have hard clips (>1.0)",
    len(clip_failures) == 0,
    f"{len(clip_failures)} failures" if clip_failures else "all clean",
    "0 failures")
if clip_failures:
    print(f"       Clipping in: {clip_failures[:5]}")

log("T11 All output durations within ±0.5s of input",
    len(dur_failures) == 0,
    f"{len(dur_failures)} failures" if dur_failures else "all OK")

log("T11 All output sample rates match input ({input_sr}Hz)",
    len(sr_failures) == 0,
    f"{len(sr_failures)} failures" if sr_failures else "all OK")

# ══════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════
print(f"\n{BOLD}{'═'*62}")
print(f"  สรุปผลการทดสอบ LongPlay Studio — Real-World Full Test")
print(f"{'═'*62}{R}")

passed = [k for k,v in RESULTS.items() if v["pass"]]
failed = [k for k,v in RESULTS.items() if not v["pass"]]
total  = len(RESULTS)
print(f"\n  ผลรวม: {len(passed)}/{total} PASS  |  {len(failed)} FAIL\n")

# Thai table by test section
sections = {
    "Test 1: Platform Mastering + LUFS":   "T1",
    "Test 2: EQ Real Effect":              "T2",
    "Test 3: Compressor Presets":          "T3",
    "Test 4: Stereo Imager":               "T4",
    "Test 5: Maximizer IRC Modes":         "T5",
    "Test 6: Video Import":                "T6",
    "Test 7: Batch Processing":            "T7",
    "Test 8: Save/Load Settings":          "T8",
    "Test 9: Realtime Meter":              "T9",
    "Test 10: Multi-file Consistency":     "T10",
    "Test 11: Output Quality Gate":        "T11",
}

print(f"  {'หัวข้อ':<35} {'ผล':<8} {'PASS':<5} {'FAIL'}")
print(f"  {'─'*62}")
for section_name, prefix in sections.items():
    sec_keys = [k for k in RESULTS if k.startswith(prefix + " ")]
    sec_pass = sum(1 for k in sec_keys if RESULTS[k]["pass"])
    sec_fail = len(sec_keys) - sec_pass
    status = f"{G}✅ ผ่าน{R}" if sec_fail == 0 else (f"{RD}❌ พลาด{R}" if sec_pass == 0 else f"{YL}⚠️ บางส่วน{R}")
    print(f"  {section_name:<35} {status:<15} {sec_pass:<5} {sec_fail}")

print(f"\n  {'─'*62}")
if failed:
    print(f"\n  {BOLD}{RD}❌ รายการที่ไม่ผ่าน:{R}")
    for k in failed:
        m = RESULTS[k].get("measured","")
        e = RESULTS[k].get("expected","")
        print(f"    • {k}")
        if m: print(f"      วัดได้: {m}  |  ต้องการ: {e}")

if not failed:
    print(f"\n  {BOLD}{G}✅ ผ่านทุกการทดสอบ! LongPlay Studio ทำงานได้สมบูรณ์{R}")
elif len(failed) <= 5:
    print(f"\n  {BOLD}{YL}⚠️  ผ่าน {len(passed)}/{total} — มีปัญหาเล็กน้อย{R}")
else:
    print(f"\n  {BOLD}{RD}❌ มีปัญหาหลายจุด ({len(failed)}/{total} fail){R}")

# Save JSON report
report = {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "version": "V5.11.1",
    "total": total, "passed": len(passed), "failed": len(failed),
    "pass_rate": f"{len(passed)/total*100:.1f}%",
    "results": {k: {"pass": bool(v["pass"]), "measured": v["measured"]} for k,v in RESULTS.items()},
}
report_path = os.path.join(OUT_DIR, "realworld_full_report.json")
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(f"\n  Output files: {OUT_DIR}")
print(f"  JSON report:  {report_path}")
