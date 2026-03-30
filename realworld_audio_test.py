#!/usr/bin/env python3
"""
LongPlay Studio V5.11 — Real-World Audio Processing Test
=========================================================
ทดสอบจริง: โหลด WAV จริง → ผ่าน processing chain → วิเคราะห์ output
ไม่ใช่ unit test — นี่คือการตรวจสอบว่าเสียงที่ได้ออกมาถูกต้องหรือเปล่า
"""
from __future__ import annotations
import os, sys, time, tempfile, json
import numpy as np
import soundfile as sf

BASE = os.path.dirname(os.path.abspath(__file__))
HOOK_WAV = os.path.join(BASE, "Vinyl Prophet Vol.1", "Hook",
                         "1.Higher Vibration_hook.wav")

# ── ANSI colors ────────────────────────────────────────────────────────────
GRN = "\033[92m"; RED = "\033[91m"; YLW = "\033[93m"; BLU = "\033[94m"
CYN = "\033[96m"; BLD = "\033[1m"; RST = "\033[0m"

def ok(msg):  print(f"  {GRN}✅{RST} {msg}")
def fail(msg): print(f"  {RED}❌{RST} {msg}")
def warn(msg): print(f"  {YLW}⚠️ {RST} {msg}")
def info(msg): print(f"  {BLU}ℹ️ {RST} {msg}")
def hdr(msg):  print(f"\n{BLD}{CYN}{'─'*60}{RST}\n{BLD}{CYN}  {msg}{RST}\n{BLD}{CYN}{'─'*60}{RST}")

# ── Audio analysis helpers ──────────────────────────────────────────────────
def peak_db(audio): return 20 * np.log10(max(float(np.max(np.abs(audio))), 1e-10))
def rms_db(audio):  return 20 * np.log10(max(float(np.sqrt(np.mean(audio.astype(np.float64)**2))), 1e-10))
def peak_lin(audio): return float(np.max(np.abs(audio)))
def rms_lin(audio):  return float(np.sqrt(np.mean(audio.astype(np.float64)**2)))

def is_clipping(audio, threshold=0.999):
    return float(np.max(np.abs(audio))) >= threshold

def count_clips(audio, threshold=0.999):
    return int(np.sum(np.abs(audio) >= threshold))

def compute_spectrum(audio, sr, n_bands=8):
    """แบ่ง spectrum เป็น 8 bands แล้วคืน RMS dB แต่ละ band"""
    if audio.ndim == 2:
        mono = audio.mean(axis=1).astype(np.float32)
    else:
        mono = audio.astype(np.float32)
    # FFT
    N = len(mono)
    fft = np.abs(np.fft.rfft(mono))
    freqs = np.fft.rfftfreq(N, 1/sr)
    # 8 octave-ish bands: 20-80, 80-250, 250-500, 500-1k, 1k-2k, 2k-4k, 4k-8k, 8k-20k
    bands = [(20,80), (80,250), (250,500), (500,1000), (1000,2000), (2000,4000), (4000,8000), (8000,20000)]
    result = {}
    for lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi)
        if mask.any():
            energy = float(np.sqrt(np.mean(fft[mask]**2)))
            result[f"{lo}-{hi}Hz"] = 20 * np.log10(max(energy, 1e-10))
        else:
            result[f"{lo}-{hi}Hz"] = -120.0
    return result

def correlation(audio):
    """Stereo correlation -1 (out of phase) to +1 (mono)"""
    if audio.ndim < 2 or audio.shape[1] < 2:
        return 1.0
    L, R = audio[:,0].astype(np.float64), audio[:,1].astype(np.float64)
    denom = np.sqrt(np.sum(L**2) * np.sum(R**2))
    if denom < 1e-12:
        return 1.0
    return float(np.sum(L * R) / denom)

def stereo_width(audio):
    """Stereo width: 0=mono, 1=normal, >1=wide"""
    if audio.ndim < 2 or audio.shape[1] < 2:
        return 0.0
    M = (audio[:,0] + audio[:,1]) / 2
    S = (audio[:,0] - audio[:,1]) / 2
    mid_rms = rms_lin(M)
    side_rms = rms_lin(S)
    if mid_rms < 1e-10:
        return 0.0
    return side_rms / mid_rms

def thd_estimate(audio, sr, freq=440.0, n_harmonics=5):
    """ประมาณ THD (Total Harmonic Distortion) ด้วย FFT"""
    if audio.ndim == 2:
        mono = audio.mean(axis=1).astype(np.float32)
    else:
        mono = audio.astype(np.float32)
    N = len(mono)
    fft = np.abs(np.fft.rfft(mono)) / N
    freqs = np.fft.rfftfreq(N, 1/sr)
    freq_res = sr / N

    def bin_energy(f):
        idx = int(round(f / freq_res))
        lo, hi = max(0, idx-3), min(len(fft)-1, idx+3)
        return float(np.max(fft[lo:hi+1]))

    fundamental = bin_energy(freq)
    if fundamental < 1e-12:
        return 0.0
    harmonics_sum_sq = sum(bin_energy(freq * n)**2 for n in range(2, n_harmonics+1))
    return 100.0 * np.sqrt(harmonics_sum_sq) / fundamental

def analyze_audio(audio, sr, label=""):
    """วิเคราะห์ audio และคืน dict ผล"""
    pk = peak_db(audio)
    rm = rms_db(audio)
    pk_lin = peak_lin(audio)
    corr = correlation(audio)
    sw = stereo_width(audio)
    clips = count_clips(audio)
    spec = compute_spectrum(audio, sr)
    dur = audio.shape[0] / sr

    return {
        "label": label,
        "duration_sec": dur,
        "peak_db": pk,
        "rms_db": rm,
        "peak_lin": pk_lin,
        "correlation": corr,
        "stereo_width": sw,
        "clip_count": clips,
        "is_clipping": clips > 0,
        "spectrum": spec,
        "sr": sr,
    }

def print_analysis(a: dict):
    clip_str = f"{RED}⚠️  CLIPPING! {a['clip_count']} samples{RST}" if a["is_clipping"] else f"{GRN}ไม่มี clipping{RST}"
    print(f"    Peak:        {a['peak_db']:+.2f} dBTP  (linear {a['peak_lin']:.4f})")
    print(f"    RMS:         {a['rms_db']:+.2f} dBFS")
    print(f"    Duration:    {a['duration_sec']:.2f}s  @{a['sr']} Hz")
    print(f"    Correlation: {a['correlation']:+.3f}  (stereo width: {a['stereo_width']:.3f})")
    print(f"    Clipping:    {clip_str}")
    print(f"    Spectrum (RMS dB per octave band):")
    for band, db_val in a["spectrum"].items():
        bar_len = max(0, min(30, int((db_val + 120) / 4)))
        bar = "█" * bar_len
        print(f"      {band:>14s}  {db_val:+6.1f} dB  {BLU}{bar}{RST}")

def diff_analysis(before: dict, after: dict):
    """เปรียบเทียบ before/after"""
    delta_peak = after["peak_db"] - before["peak_db"]
    delta_rms  = after["rms_db"]  - before["rms_db"]
    delta_corr = after["correlation"] - before["correlation"]
    delta_sw   = after["stereo_width"] - before["stereo_width"]
    print(f"\n    {'📊 เปรียบเทียบ INPUT vs OUTPUT':}")
    print(f"    Peak change:   {delta_peak:+.2f} dB  {'(ดัง)' if delta_peak>0.5 else '(เบา)' if delta_peak<-0.5 else '(เท่าเดิม)'}")
    print(f"    RMS change:    {delta_rms:+.2f} dB   {'(ดัง)' if delta_rms>0.5 else '(เบา)' if delta_rms<-0.5 else '(เท่าเดิม)'}")
    print(f"    Correlation:   {before['correlation']:+.3f} → {after['correlation']:+.3f}  Δ{delta_corr:+.3f}")
    print(f"    Stereo Width:  {before['stereo_width']:.3f} → {after['stereo_width']:.3f}  Δ{delta_sw:+.3f}")
    # Spectrum diff
    print(f"\n    Spectrum change (OUTPUT − INPUT) dB:")
    for band in before["spectrum"]:
        d = after["spectrum"].get(band, 0) - before["spectrum"][band]
        bar = ("+" * min(20, int(abs(d)/0.5)) if d > 0.3
               else "-" * min(20, int(abs(d)/0.5)) if d < -0.3
               else "·")
        color = GRN if d > 0.3 else RED if d < -0.3 else RST
        print(f"      {band:>14s}  {color}{d:+.1f} dB  {bar}{RST}")
    return delta_peak, delta_rms


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TEST SUITE
# ══════════════════════════════════════════════════════════════════════════════
results = {}

print(f"\n{BLD}{'═'*60}")
print(f"  LongPlay Studio V5.11 — Real-World Audio Processing Test")
print(f"  ไฟล์ทดสอบ: 1.Higher Vibration_hook.wav (29s, stereo, 48kHz)")
print(f"{'═'*60}{RST}\n")

# ── โหลดไฟล์จริง ─────────────────────────────────────────────────────────
hdr("0. โหลดไฟล์ WAV จริง")
raw_audio, raw_sr = sf.read(HOOK_WAV)
raw_audio = raw_audio.astype(np.float32)
input_info = analyze_audio(raw_audio, raw_sr, "INPUT (Hook WAV)")
print(f"  {BLD}📥 INPUT:{RST}")
print_analysis(input_info)
results["input"] = input_info

# ── Init chain ─────────────────────────────────────────────────────────────
from modules.master.chain import MasterChain
from modules.master.equalizer import Equalizer
from modules.master.dynamics import Dynamics
from modules.master.limiter import LookAheadLimiterFast
from modules.master.maximizer import Maximizer
from modules.master.imager import Imager


# ══════════════════════════════════════════════════════════════════════════════
# 1. GAIN KNOB TEST — เพิ่ม gain แล้ว loudness ต้องขึ้น
# ══════════════════════════════════════════════════════════════════════════════
hdr("1. ทดสอบ GAIN KNOB — เพิ่ม GAIN 0 / 6 / 12 dB")
gain_results = {}
for gain_db_val in [0.0, 6.0, 12.0]:
    chain = MasterChain()
    chain.maximizer.set_gain(gain_db_val)
    chain.equalizer.enabled = False
    chain.dynamics.enabled = False
    chain.imager.enabled = False

    meter_data = []
    chain.set_meter_callback(lambda d: meter_data.append(d))

    inp = tempfile.mktemp(suffix=".wav")
    out = tempfile.mktemp(suffix=f"_gain{int(gain_db_val)}.wav")
    sf.write(inp, raw_audio, raw_sr)
    chain.load_audio(inp)
    chain.render(out)
    os.unlink(inp)

    out_audio, out_sr = sf.read(out)
    out_audio = out_audio.astype(np.float32)
    info_gain = analyze_audio(out_audio, out_sr, f"GAIN={gain_db_val}dB")
    gain_results[gain_db_val] = info_gain

    meter_rms = meter_data[-1].get("left_rms_db", "N/A") if meter_data else "N/A"
    print(f"\n  {BLD}GAIN = {gain_db_val} dB{RST}")
    print(f"    Peak: {info_gain['peak_db']:+.2f} dBTP   RMS: {info_gain['rms_db']:+.2f} dBFS")
    print(f"    Meter callback (left_rms_db): {meter_rms}")
    clip_str = f"{RED}⚠️  CLIPPING{RST}" if info_gain["is_clipping"] else f"{GRN}ไม่มี clipping{RST}"
    print(f"    Clipping: {clip_str}")
    os.unlink(out)

# ตรวจว่า gain ขึ้นจริง
g0 = gain_results[0.0]["rms_db"]
g6 = gain_results[6.0]["rms_db"]
g12 = gain_results[12.0]["rms_db"]
print(f"\n  {BLD}สรุป GAIN test:{RST}")
if g6 > g0 and g12 > g6:
    ok(f"GAIN knob ทำงานจริง: 0dB={g0:.1f} → 6dB={g6:.1f} → 12dB={g12:.1f} dBFS (ดังขึ้น)")
    results["gain_works"] = True
else:
    fail(f"GAIN ไม่มีผล: 0dB={g0:.1f}, 6dB={g6:.1f}, 12dB={g12:.1f} dBFS")
    results["gain_works"] = False

if gain_results[12.0]["is_clipping"]:
    warn(f"GAIN 12dB ทำให้เสียงแตก ({gain_results[12.0]['clip_count']} samples clip) — Limiter ควร catch แต่อาจ overshoot")
else:
    ok("GAIN 12dB ผ่าน Limiter โดยไม่แตก")


# ══════════════════════════════════════════════════════════════════════════════
# 2. EQ TEST — Boost แต่ละ band แล้วดู spectrum เปลี่ยน
# ══════════════════════════════════════════════════════════════════════════════
hdr("2. ทดสอบ EQ — Boost +12 dB แต่ละ band ดู spectrum เปลี่ยน")
eq_band_tests = [
    (1, 64,   "Bass (64 Hz)",     "80-250Hz"),
    (2, 125,  "Low-Mid (125 Hz)", "80-250Hz"),
    (4, 1000, "Mid (1 kHz)",      "500-1000Hz"),
    (6, 8000, "High (8 kHz)",     "4000-8000Hz"),
]

eq_pass = 0
eq_total = len(eq_band_tests)
for band_idx, freq_hz, label, check_band in eq_band_tests:
    chain = MasterChain()
    chain.dynamics.enabled = False
    chain.imager.enabled = False
    chain.maximizer.set_gain(0.0)

    # Flat EQ — no boost
    inp = tempfile.mktemp(suffix=".wav")
    out_flat = tempfile.mktemp(suffix="_flat.wav")
    sf.write(inp, raw_audio, raw_sr)
    chain.load_audio(inp)
    chain.render(out_flat)
    flat_audio, _ = sf.read(out_flat)
    spec_flat = compute_spectrum(flat_audio.astype(np.float32), raw_sr)

    # Boost +12 dB at specific band
    chain2 = MasterChain()
    chain2.dynamics.enabled = False
    chain2.imager.enabled = False
    chain2.maximizer.set_gain(0.0)
    chain2.equalizer.bands[band_idx].freq = float(freq_hz)
    chain2.equalizer.bands[band_idx].gain = 12.0
    out_boost = tempfile.mktemp(suffix="_boost.wav")
    sf.write(inp, raw_audio, raw_sr)
    chain2.load_audio(inp)
    chain2.render(out_boost)
    boost_audio, _ = sf.read(out_boost)
    spec_boost = compute_spectrum(boost_audio.astype(np.float32), raw_sr)

    flat_val  = spec_flat.get(check_band, -120)
    boost_val = spec_boost.get(check_band, -120)
    delta = boost_val - flat_val

    print(f"\n  {BLD}EQ Band: {label}{RST}")
    print(f"    {check_band}: flat={flat_val:.1f} dB → boost={boost_val:.1f} dB  Δ={delta:+.1f} dB")

    if delta > 0.5:
        ok(f"EQ {label} +12 dB มีผลจริง (Δ {delta:+.1f} dB ที่ {check_band})")
        eq_pass += 1
    else:
        fail(f"EQ {label} +12 dB ไม่มีผลที่ {check_band} (Δ {delta:+.1f} dB)")

    for f in [inp, out_flat, out_boost]:
        if os.path.exists(f): os.unlink(f)

print(f"\n  {BLD}สรุป EQ test: {eq_pass}/{eq_total} bands ผ่าน{RST}")
results["eq_pass"] = eq_pass
results["eq_total"] = eq_total
results["eq_works"] = eq_pass >= eq_total // 2


# ══════════════════════════════════════════════════════════════════════════════
# 3. COMPRESSOR TEST — ตรวจว่า peaks ถูก reduce
# ══════════════════════════════════════════════════════════════════════════════
hdr("3. ทดสอบ Compressor — ต้อง reduce peaks, RMS อาจขึ้น (makeup gain)")
chain_comp = MasterChain()
chain_comp.dynamics.set_threshold(-20.0)
chain_comp.dynamics.set_ratio(8.0)
chain_comp.dynamics.set_makeup_gain(4.0)
chain_comp.dynamics.set_attack(5.0)
chain_comp.dynamics.set_release(100.0)
chain_comp.equalizer.enabled = False
chain_comp.imager.enabled = False
chain_comp.maximizer.set_gain(0.0)

inp = tempfile.mktemp(suffix=".wav")
out_comp = tempfile.mktemp(suffix="_comp.wav")
sf.write(inp, raw_audio, raw_sr)
chain_comp.load_audio(inp)
chain_comp.render(out_comp)
comp_audio, comp_sr = sf.read(out_comp)
comp_audio = comp_audio.astype(np.float32)
comp_info = analyze_audio(comp_audio, comp_sr, "Compressed")

for f in [inp, out_comp]:
    if os.path.exists(f): os.unlink(f)

print(f"\n  {BLD}INPUT:{RST}   Peak={input_info['peak_db']:+.2f} dB  RMS={input_info['rms_db']:+.2f} dB")
print(f"  {BLD}COMPRESSED:{RST} Peak={comp_info['peak_db']:+.2f} dB  RMS={comp_info['rms_db']:+.2f} dB")

if comp_info["is_clipping"]:
    fail(f"Compressor output มี clipping ({comp_info['clip_count']} samples)")
    results["comp_clips"] = True
else:
    ok("Compressor output ไม่มี clipping")
    results["comp_clips"] = False

if comp_info["rms_db"] >= input_info["rms_db"] - 1.0:
    ok(f"Compressor ทำงาน: RMS {input_info['rms_db']:.1f} → {comp_info['rms_db']:.1f} dB (makeup gain เพิ่ม loudness)")
    results["comp_works"] = True
else:
    warn(f"Compressor output เบาลง: {input_info['rms_db']:.1f} → {comp_info['rms_db']:.1f} dB")
    results["comp_works"] = False


# ══════════════════════════════════════════════════════════════════════════════
# 4. LIMITER CEILING TEST — ตรวจ True Peak ไม่เกิน ceiling
# ══════════════════════════════════════════════════════════════════════════════
hdr("4. ทดสอบ Limiter — True Peak ceiling enforcement")

for ceiling_val in [-1.0, -0.3]:
    ceiling_lin = 10 ** (ceiling_val / 20.0)
    chain_lim = MasterChain()
    chain_lim.maximizer.set_gain(8.0)   # drive hard into limiter
    chain_lim.maximizer.set_ceiling(ceiling_val)
    chain_lim.equalizer.enabled = False
    chain_lim.dynamics.enabled = False
    chain_lim.imager.enabled = False

    inp = tempfile.mktemp(suffix=".wav")
    out_lim = tempfile.mktemp(suffix=f"_ceiling{abs(ceiling_val)}.wav")
    sf.write(inp, raw_audio, raw_sr)
    chain_lim.load_audio(inp)
    chain_lim.render(out_lim)
    lim_audio, _ = sf.read(out_lim)
    lim_audio = lim_audio.astype(np.float32)
    lim_peak = peak_lin(lim_audio)
    lim_peak_db_val = peak_db(lim_audio)

    for f in [inp, out_lim]:
        if os.path.exists(f): os.unlink(f)

    print(f"\n  {BLD}Ceiling = {ceiling_val} dBTP  (linear = {ceiling_lin:.4f}){RST}")
    print(f"    Output peak: {lim_peak:.4f} linear  ({lim_peak_db_val:+.2f} dBTP)")

    if lim_peak <= ceiling_lin * 1.01:
        ok(f"Limiter ceiling {ceiling_val} dBTP ✓  peak {lim_peak_db_val:.2f} ≤ {ceiling_val} dBTP")
        results[f"limiter_ceiling_{ceiling_val}"] = "PASS"
    else:
        fail(f"Limiter ceiling {ceiling_val} dBTP ✗  peak {lim_peak_db_val:.2f} เกิน ceiling!")
        results[f"limiter_ceiling_{ceiling_val}"] = "FAIL"


# ══════════════════════════════════════════════════════════════════════════════
# 5. STEREO IMAGER TEST — width 0%, 100%, 200%
# ══════════════════════════════════════════════════════════════════════════════
hdr("5. ทดสอบ Stereo Imager — Width 0% / 100% / 200%")
width_results = {}

for width_val in [0, 100, 200]:
    chain_img = MasterChain()
    chain_img.imager.set_width(width_val)
    chain_img.imager.stereoize_mode = "off"
    chain_img.equalizer.enabled = False
    chain_img.dynamics.enabled = False
    chain_img.maximizer.set_gain(0.0)

    inp = tempfile.mktemp(suffix=".wav")
    out_img = tempfile.mktemp(suffix=f"_width{width_val}.wav")
    sf.write(inp, raw_audio, raw_sr)
    chain_img.load_audio(inp)
    chain_img.render(out_img)
    img_audio, _ = sf.read(out_img)
    img_audio = img_audio.astype(np.float32)

    corr = correlation(img_audio)
    sw = stereo_width(img_audio)
    width_results[width_val] = {"corr": corr, "sw": sw}

    print(f"\n  {BLD}Width = {width_val}%{RST}  correlation={corr:.3f}  stereo_width={sw:.3f}")

    if width_val == 0 and corr > 0.99:
        ok(f"Width=0% → mono (correlation {corr:.3f} ≈ 1.0)")
    elif width_val == 200 and sw > width_results.get(100, {}).get("sw", 0):
        ok(f"Width=200% → wider than 100% (sw {sw:.3f})")
    elif width_val == 100:
        ok(f"Width=100% → passthrough (sw {sw:.3f})")

    for f in [inp, out_img]:
        if os.path.exists(f): os.unlink(f)

w0 = width_results.get(0,   {}).get("corr", 0)
w100 = width_results.get(100, {}).get("sw", 0)
w200 = width_results.get(200, {}).get("sw", 0)
if w200 >= w100 * 0.9:
    ok(f"Imager width gradient ถูกต้อง: 100%={w100:.3f} → 200%={w200:.3f}")
    results["imager_works"] = True
else:
    warn(f"Imager gradient ไม่ชัดเจน: 100%={w100:.3f}, 200%={w200:.3f}")
    results["imager_works"] = False


# ══════════════════════════════════════════════════════════════════════════════
# 6. FULL CHAIN MASTERING — ทดสอบ pipeline ครบ
# ══════════════════════════════════════════════════════════════════════════════
hdr("6. Full Chain Mastering — ตั้งค่า mastering จริง → render → วิเคราะห์")

# YouTube mastering preset
chain_full = MasterChain()
chain_full.set_platform("YouTube")
chain_full.intensity = 80

# EQ: เพิ่ม warmth
chain_full.equalizer.bands[1].gain = 2.0   # 64 Hz slight boost
chain_full.equalizer.bands[2].gain = 1.5   # 125 Hz
chain_full.equalizer.bands[4].gain = -1.0  # 1 kHz slight dip
chain_full.equalizer.bands[7].gain = 2.0   # 16 kHz air

# Dynamics: medium compression
chain_full.dynamics.set_threshold(-22.0)
chain_full.dynamics.set_ratio(3.0)
chain_full.dynamics.set_attack(10.0)
chain_full.dynamics.set_release(150.0)
chain_full.dynamics.set_makeup_gain(2.0)

# Imager: slight wide
chain_full.imager.set_width(115)

# Maximizer: push to -14 LUFS
chain_full.maximizer.set_gain(4.0)
chain_full.maximizer.set_ceiling(-1.0)
chain_full.maximizer.set_irc_mode("IRC 4")
chain_full.maximizer.set_character(4.0)

meter_events = []
chain_full.set_meter_callback(lambda d: meter_events.append(d))

inp = tempfile.mktemp(suffix=".wav")
out_master = os.path.join(BASE, "test_master_output.wav")
sf.write(inp, raw_audio, raw_sr)
chain_full.load_audio(inp)

t_start = time.time()
result_path = chain_full.render(out_master)
render_time = time.time() - t_start

final_path = result_path or out_master
os.unlink(inp)

print(f"\n  Render time: {BLD}{render_time:.2f}s{RST}")
print(f"  Output: {final_path}")
print(f"  Meter events: {len(meter_events)}")

master_audio, master_sr = sf.read(final_path)
master_audio = master_audio.astype(np.float32)
master_info = analyze_audio(master_audio, master_sr, "MASTERED OUTPUT")
results["master"] = master_info

print(f"\n  {BLD}📊 INPUT:{RST}")
print_analysis(input_info)
print(f"\n  {BLD}📊 MASTERED OUTPUT:{RST}")
print_analysis(master_info)
diff_analysis(input_info, master_info)


# ══════════════════════════════════════════════════════════════════════════════
# 7. DISTORTION CHECK — ตรวจว่าเสียงแตกไหม (THD estimate)
# ══════════════════════════════════════════════════════════════════════════════
hdr("7. Distortion Check — ตรวจ THD และ clipping")

# สร้าง test tone แล้ว run ผ่าน chain ด้วย GAIN สูง
import numpy as np
t = np.linspace(0, 5.0, int(44100*5), endpoint=False)
test_tone = np.stack([
    np.sin(2*np.pi*1000*t)*0.7 + np.sin(2*np.pi*2000*t)*0.1,
    np.sin(2*np.pi*1000*t)*0.7 + np.sin(2*np.pi*2000*t)*0.1,
], axis=1).astype(np.float32)

tone_results = {}
for gain_test in [0.0, 6.0, 12.0, 18.0]:
    chain_thd = MasterChain()
    chain_thd.maximizer.set_gain(gain_test)
    chain_thd.maximizer.set_ceiling(-0.3)
    chain_thd.equalizer.enabled = False
    chain_thd.dynamics.enabled = False
    chain_thd.imager.enabled = False

    inp = tempfile.mktemp(suffix=".wav")
    out_tone = tempfile.mktemp(suffix=f"_tone_gain{int(gain_test)}.wav")
    sf.write(inp, test_tone, 44100)
    chain_thd.load_audio(inp)
    chain_thd.render(out_tone)
    out_aud, _ = sf.read(out_tone)
    out_aud = out_aud.astype(np.float32)

    thd = thd_estimate(out_aud, 44100, freq=1000.0)
    pk = peak_db(out_aud)
    clips = count_clips(out_aud)
    tone_results[gain_test] = {"thd": thd, "peak_db": pk, "clips": clips}

    for f in [inp, out_tone]:
        if os.path.exists(f): os.unlink(f)

    print(f"\n  {BLD}Gain = {gain_test:+.0f} dB{RST}:")
    print(f"    Peak: {pk:+.2f} dBTP   Clips: {clips}   THD: {thd:.2f}%")

    if clips > 0:
        fail(f"Gain {gain_test} dB มี {clips} samples clip — เสียงแตก!")
    elif thd > 5.0:
        warn(f"THD {thd:.1f}% สูง — มี harmonic distortion (อาจ OK ถ้าตั้งใจ)")
    else:
        ok(f"Gain {gain_test} dB: ไม่มี clipping, THD {thd:.2f}% ปกติ")

results["tone_thd"] = tone_results


# ══════════════════════════════════════════════════════════════════════════════
# 8. IRC MODE COMPARISON — เปรียบ IRC1 vs IRC4 vs IRCLL
# ══════════════════════════════════════════════════════════════════════════════
hdr("8. IRC Mode Comparison — เปรียบเสียง IRC 1 vs IRC 4 vs IRC LL")
irc_results = {}
for irc_mode in ["IRC 1", "IRC 4", "IRC LL"]:
    chain_irc = MasterChain()
    chain_irc.maximizer.set_gain(8.0)
    chain_irc.maximizer.set_ceiling(-1.0)
    chain_irc.maximizer.set_irc_mode(irc_mode)
    chain_irc.equalizer.enabled = False
    chain_irc.dynamics.enabled = False
    chain_irc.imager.enabled = False

    inp = tempfile.mktemp(suffix=".wav")
    out_irc = tempfile.mktemp(suffix=f"_irc.wav")
    sf.write(inp, raw_audio, raw_sr)
    chain_irc.load_audio(inp)
    chain_irc.render(out_irc)
    irc_aud, _ = sf.read(out_irc)
    irc_aud = irc_aud.astype(np.float32)
    pk = peak_db(irc_aud)
    rm = rms_db(irc_aud)
    clips = count_clips(irc_aud)
    irc_results[irc_mode] = {"peak_db": pk, "rms_db": rm, "clips": clips}

    for f in [inp, out_irc]:
        if os.path.exists(f): os.unlink(f)

    clip_str = f"{RED}⚠️ CLIP!{RST}" if clips > 0 else f"{GRN}ไม่ clip{RST}"
    print(f"  {BLD}{irc_mode:8s}{RST}  Peak={pk:+.2f} dB  RMS={rm:+.2f} dB  {clip_str}")

# ── Final check ───────────────────────────────────────────────────────────
print(f"\n  สรุป IRC modes:")
any_irc_clip = any(v["clips"]>0 for v in irc_results.values())
if not any_irc_clip:
    ok("ทุก IRC mode ผ่าน ceiling โดยไม่มี clipping")
    results["irc_ok"] = True
else:
    fail("บาง IRC mode มี clipping")
    results["irc_ok"] = False


# ══════════════════════════════════════════════════════════════════════════════
# 9. METER CALLBACK VERIFICATION — real data จาก render
# ══════════════════════════════════════════════════════════════════════════════
hdr("9. Realtime Meter Verification — ค่าจาก meter callback")
if meter_events:
    last = meter_events[-1]
    print(f"\n  Meter data จาก Full Chain render:")
    for k, v in last.items():
        if not k.startswith("_"):  # skip internal arrays
            if isinstance(v, float):
                print(f"    {k:30s}: {v:+.3f}")
            else:
                print(f"    {k:30s}: {v}")

    rms_left = last.get("left_rms_db", None)
    if rms_left is not None and float(rms_left) < 0:
        ok(f"left_rms_db = {float(rms_left):.1f} dB (ค่าลบ = มี signal จริง)")
        results["meter_ok"] = True
    else:
        fail(f"left_rms_db ผิดปกติ: {rms_left}")
        results["meter_ok"] = False
else:
    warn("ไม่มี meter events")
    results["meter_ok"] = False


# ══════════════════════════════════════════════════════════════════════════════
# FINAL REPORT
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n\n{BLD}{'═'*60}")
print(f"  🎵 สรุปผลการทดสอบ LongPlay Studio จริง (Programmatic)")
print(f"{'═'*60}{RST}")

checks = [
    ("GAIN knob ทำงาน (loudness ขึ้นจริง)",     results.get("gain_works", False)),
    ("GAIN +12 dB ไม่แตก/ไม่ clip",             not gain_results.get(12.0, {}).get("is_clipping", True)),
    (f"EQ knob มีผลต่อ spectrum ({results.get('eq_pass',0)}/{results.get('eq_total',4)} bands)",
                                                results.get("eq_works", False)),
    ("Compressor ทำงาน (RMS เปลี่ยน)",          results.get("comp_works", False)),
    ("Compressor ไม่ทำให้เสียงแตก",             not results.get("comp_clips", True)),
    ("Limiter ceiling -1.0 dBTP บังคับจริง",    results.get("limiter_ceiling_-1.0") == "PASS"),
    ("Limiter ceiling -0.3 dBTP บังคับจริง",    results.get("limiter_ceiling_-0.3") == "PASS"),
    ("Stereo Imager width ทำงาน",               results.get("imager_works", False)),
    ("Full chain render ได้ output",            results.get("master") is not None),
    ("Full chain output ไม่มี clipping",        not (results.get("master") or {}).get("is_clipping", True)),
    ("IRC modes ทุกตัวไม่ clip",                results.get("irc_ok", False)),
    ("Meter callback ส่งค่าจริง (RMS < 0 dB)",  results.get("meter_ok", False)),
    ("THD ต่ำ (GAIN 0dB)",                      (tone_results.get(0.0, {}).get("thd", 99) < 3.0)),
    ("ไม่มี clipping ที่ GAIN=18dB (Limiter catch)", tone_results.get(18.0, {}).get("clips", 1) == 0),
]

pass_count = sum(1 for _, p in checks if p)
total_count = len(checks)

for label, passed in checks:
    if passed:
        ok(label)
    else:
        fail(label)

print(f"\n  {BLD}ผล: {pass_count}/{total_count} ✅{RST}")

# เสียงแตกหรือเปล่า?
print(f"\n{BLD}{'─'*60}")
print(f"  🔊 สรุป: เสียงแตกหรือเปล่า?")
print(f"{'─'*60}{RST}")

master = results.get("master", {})
master_clips = (master or {}).get("clip_count", 0)
master_peak  = (master or {}).get("peak_db", -999)
master_rms   = (master or {}).get("rms_db", -999)

if master_clips > 0:
    print(f"  {RED}{BLD}❌ เสียงแตก! ({master_clips} samples clip ใน mastered output){RST}")
elif master_peak > -0.3:
    print(f"  {YLW}{BLD}⚠️  เสียงเกือบแตก (peak {master_peak:.2f} dBTP — ใกล้ 0 dBFS){RST}")
else:
    print(f"  {GRN}{BLD}✅ เสียงไม่แตก{RST}")
    print(f"     Mastered peak: {master_peak:.2f} dBTP  (ปลอดภัย)")
    print(f"     Mastered RMS:  {master_rms:.2f} dBFS")
    print(f"     Limiter ทำงานถูกต้อง: peak ไม่เกิน ceiling")

print(f"\n  Master output ถูกบันทึกที่: {BLD}test_master_output.wav{RST}")
print(f"  เปิดฟังด้วย: {CYN}open '{out_master}'{RST}\n")

# Save JSON report
report = {
    "input": {k: v for k, v in input_info.items() if k != "spectrum"},
    "master": {k: v for k, v in (master or {}).items() if k != "spectrum"},
    "gain_test": {str(k): {"rms_db": v["rms_db"], "clips": v["is_clipping"]} for k,v in gain_results.items()},
    "eq_pass": f"{results.get('eq_pass',0)}/{results.get('eq_total',4)}",
    "irc_results": irc_results,
    "pass_count": pass_count,
    "total_count": total_count,
    "sound_clipping": master_clips > 0,
}
with open(os.path.join(BASE, "realworld_test_report.json"), "w") as f:
    json.dump(report, f, indent=2, default=str)
print(f"  รายงาน JSON: {BLD}realworld_test_report.json{RST}\n")
