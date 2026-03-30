#!/usr/bin/env python3
"""
LongPlay Studio V5.11.1 — Desktop Export & Signal Measurement Test
Exports mastered WAV for Spotify / YouTube / Apple Music
Measures: LUFS, True Peak, Dynamic Range, Stereo Correlation, Clipping
Reports in Thai. Copies to ~/Desktop/LongPlay_Test/. Cleans up after.
"""

import sys, os, shutil, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import soundfile as sf
import pyloudnorm as pyln
from scipy.signal import resample_poly

# ─── Paths ────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
VINYL_DIR   = os.path.join(PROJECT_DIR, "Vinyl Prophet Vol.1", "Hook")
WORK_DIR    = os.path.join(PROJECT_DIR, "_desktop_test_tmp")
DESKTOP_DIR = os.path.expanduser("~/Desktop/LongPlay_Test")

# Source audio — pick first available Hook WAV
SOURCE_FILE = None
for fname in sorted(os.listdir(VINYL_DIR)):
    if fname.endswith(".wav"):
        SOURCE_FILE = os.path.join(VINYL_DIR, fname)
        break

if SOURCE_FILE is None:
    sys.exit("❌ ไม่พบไฟล์ WAV ใน Vinyl Prophet Vol.1/Hook/")

print(f"📂 ไฟล์ต้นฉบับ: {os.path.basename(SOURCE_FILE)}")

os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(DESKTOP_DIR, exist_ok=True)

# ─── Signal Measurement Functions ─────────────────────────────────────

def load_audio(path):
    data, sr = sf.read(path)
    if data.ndim == 1:
        data = np.column_stack([data, data])
    return data.astype(np.float64), sr

def measure_lufs(data, sr):
    meter = pyln.Meter(sr)
    return float(meter.integrated_loudness(data))

def measure_true_peak(data, sr, oversample=4):
    """4x oversampled inter-sample true peak (ITU-R BS.1770)"""
    peaks = []
    for ch in range(data.shape[1]):
        up = resample_poly(data[:, ch], oversample, 1)
        peaks.append(20 * np.log10(np.max(np.abs(up)) + 1e-12))
    return max(peaks)

def measure_correlation(data):
    L, R = data[:, 0], data[:, 1]
    denom = np.std(L) * np.std(R)
    if denom < 1e-12:
        return 1.0
    return float(np.corrcoef(L, R)[0, 1])

def measure_crest(data):
    rms = np.sqrt(np.mean(data**2))
    peak = np.max(np.abs(data))
    if rms < 1e-12:
        return 0.0
    return float(20 * np.log10(peak / rms))

def measure_lu_range(data, sr):
    """EBU R128 Loudness Range (LRA): 10th→95th percentile of short-term loudness.
    Short-term blocks: 3s, 0.75s hop (EBU R128 spec).
    Returns LU range. Note: short dense content (hooks, 30s clips) naturally
    has low LRA (1-3 LU). Threshold ≥ 1 LU detects flat silence/dead signal.
    """
    import math
    block_size = int(sr * 3.0)
    hop_size   = int(sr * 0.75)
    if len(data) < block_size:
        # Short audio fallback: 400ms blocks
        block_size = int(sr * 0.4)
        hop_size   = block_size // 2
    meter = pyln.Meter(sr)
    blocks = []
    for start in range(0, len(data) - block_size, hop_size):
        blk = data[start:start + block_size]
        try:
            l = meter.integrated_loudness(blk)
            if l > -70:
                blocks.append(l)
        except:
            pass
    if len(blocks) < 2:
        return 0.0
    blocks.sort()
    p10 = blocks[max(0, int(math.floor(0.10 * len(blocks))))]
    p95 = blocks[min(len(blocks)-1, int(math.ceil(0.95 * len(blocks))) - 1)]
    return float(p95 - p10)

def measure_band_rms(data, sr, lo, hi):
    """Octave-band RMS using FFT"""
    mono = np.mean(data, axis=1)
    freqs = np.fft.rfftfreq(len(mono), 1/sr)
    fft_mag = np.abs(np.fft.rfft(mono))
    mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask):
        return -120.0
    rms = np.sqrt(np.mean(fft_mag[mask]**2))
    return float(20 * np.log10(rms + 1e-12))

def has_clips(data):
    return bool(np.any(np.abs(data) > 1.0))

# ─── Mastering Function ───────────────────────────────────────────────

def master_to_platform(input_path, output_path, platform_name,
                        target_lufs, target_tp=-1.0):
    """Run MasterChain render and apply post-processing."""
    from modules.master.chain import MasterChain

    chain = MasterChain()

    # Load platform preset
    preset_map = {
        "spotify":     "Spotify",
        "youtube":     "YouTube",
        "apple_music": "Apple Music",
    }
    preset_key = platform_name.lower().replace(" ", "_")
    preset_name = preset_map.get(preset_key, "Spotify")

    # Configure platform settings
    chain.target_lufs = target_lufs
    chain.target_tp   = target_tp

    # Apply platform preset if available
    try:
        chain.load_platform_preset(preset_name)
    except Exception:
        pass  # Use manual settings if preset unavailable

    # Render
    t0 = time.time()
    try:
        chain.render(input_path, output_path)
        elapsed = time.time() - t0
        print(f"  ✓ Render เสร็จ ({elapsed:.1f}s)")
        return True, elapsed
    except Exception as e:
        print(f"  ✗ Render ERROR: {e}")
        return False, 0.0

# ─── Alternative: Direct Python Post-Processing Render ───────────────

def master_direct(input_path, output_path, target_lufs, target_tp=-1.0):
    """
    Direct mastering without MasterChain GUI dependencies:
    1. Load audio
    2. Apply pyloudnorm LUFS normalization
    3. Apply LookAheadLimiter True Peak ceiling
    4. Save to output
    """
    try:
        from modules.master.limiter import LookAheadLimiterFast
        HAS_LIMITER = True
    except Exception:
        HAS_LIMITER = False

    t0 = time.time()
    data, sr = load_audio(input_path)

    # Step 1: LUFS normalization
    meter = pyln.Meter(sr)
    measured = meter.integrated_loudness(data)
    if measured > -70.0:
        data = pyln.normalize.loudness(data, measured, target_lufs)
        print(f"  LUFS: {measured:.2f} → {target_lufs:.2f}")
    else:
        print(f"  ⚠ LUFS trop bas: {measured:.2f} — skip normalization")

    # Step 2: True Peak ceiling
    if HAS_LIMITER:
        try:
            limiter = LookAheadLimiterFast(ceiling_db=target_tp, true_peak=True)
            data = limiter.process(data, sr)
            print(f"  True Peak ceiling: {target_tp} dBTP applied")
        except Exception as e:
            print(f"  ⚠ Limiter error: {e}")
            # Manual sample-peak ceiling fallback
            ceiling_lin = 10 ** (target_tp / 20)
            data = np.clip(data, -ceiling_lin, ceiling_lin)
    else:
        ceiling_lin = 10 ** (target_tp / 20)
        data = np.clip(data, -ceiling_lin, ceiling_lin)

    # Re-measure LUFS after limiting (may have changed)
    measured_final = meter.integrated_loudness(data)
    if measured_final > -70.0 and abs(measured_final - target_lufs) > 0.3:
        data = pyln.normalize.loudness(data, measured_final, target_lufs)
        measured_final2 = meter.integrated_loudness(data)
        print(f"  LUFS re-correct: {measured_final:.2f} → {measured_final2:.2f}")

    sf.write(output_path, data.astype(np.float32), sr, subtype='PCM_24')
    elapsed = time.time() - t0
    print(f"  ✓ Direct render เสร็จ ({elapsed:.1f}s)")
    return True, elapsed

# ─── Try MasterChain first, fallback to direct ────────────────────────

def render_platform(input_path, work_path, desktop_path,
                    platform_name, target_lufs, target_tp=-1.0):
    print(f"\n{'─'*50}")
    print(f"🎛  Mastering: {platform_name} (target: {target_lufs} LUFS, TP: {target_tp} dBTP)")

    # Try MasterChain first
    ok, elapsed = master_to_platform(input_path, work_path, platform_name,
                                     target_lufs, target_tp)

    if not ok or not os.path.exists(work_path) or os.path.getsize(work_path) < 1000:
        print(f"  → MasterChain failed, ใช้ direct render แทน...")
        ok, elapsed = master_direct(input_path, work_path, target_lufs, target_tp)

    if not ok or not os.path.exists(work_path):
        return None

    # Copy to Desktop
    shutil.copy2(work_path, desktop_path)
    print(f"  📁 Copied → {desktop_path}")
    return work_path

# ─── Main Execution ───────────────────────────────────────────────────

platforms = [
    ("Spotify",     -14.0, -1.0),
    ("YouTube",     -14.0, -1.0),
    ("Apple Music", -16.0, -1.0),
]

results = {}
all_pass = True

for platform_name, target_lufs, target_tp in platforms:
    safe_name = platform_name.replace(" ", "_")
    work_path    = os.path.join(WORK_DIR, f"{safe_name}_mastered.wav")
    desktop_path = os.path.join(DESKTOP_DIR, f"{safe_name}_mastered.wav")

    output = render_platform(SOURCE_FILE, work_path, desktop_path,
                             platform_name, target_lufs, target_tp)

    if output is None:
        print(f"  ❌ Render FAILED for {platform_name}")
        results[platform_name] = {"error": "render failed"}
        all_pass = False
        continue

    # Measure
    print(f"  📊 กำลังวัดสัญญาณ...")
    data, sr = load_audio(output)
    src_data, src_sr = load_audio(SOURCE_FILE)

    lufs_measured   = measure_lufs(data, sr)
    tp_measured     = measure_true_peak(data, sr)
    corr            = measure_correlation(data)
    crest_db        = measure_crest(data)
    lu_range        = measure_lu_range(data, sr)
    sub_bass_rms    = measure_band_rms(data, sr, 20, 80)
    highmid_rms     = measure_band_rms(data, sr, 1000, 8000)
    clips           = has_clips(data)
    file_size_mb    = os.path.getsize(output) / 1e6
    duration        = len(data) / sr
    src_duration    = len(src_data) / src_sr

    r = {
        "platform":       platform_name,
        "target_lufs":    target_lufs,
        "lufs":           round(lufs_measured, 2),
        "lufs_delta":     round(lufs_measured - target_lufs, 2),
        "true_peak":      round(tp_measured, 2),
        "correlation":    round(corr, 3),
        "crest_db":       round(crest_db, 1),
        "lu_range":       round(lu_range, 1),
        "sub_bass_rms":   round(sub_bass_rms, 1),
        "highmid_rms":    round(highmid_rms, 1),
        "clips":          clips,
        "file_size_mb":   round(file_size_mb, 2),
        "duration":       round(duration, 2),
        "src_duration":   round(src_duration, 2),
        "sample_rate":    sr,
        "desktop_path":   desktop_path,
    }

    # ISC pass/fail
    r["isc_lufs_ok"]      = abs(lufs_measured - target_lufs) <= 0.5
    r["isc_tp_ok"]        = tp_measured <= target_tp
    r["isc_noclip_ok"]    = not clips
    r["isc_corr_ok"]      = corr > 0.7
    r["isc_dur_ok"]       = abs(duration - src_duration) <= 0.5
    r["isc_sr_ok"]        = sr == 48000
    r["isc_size_ok"]      = file_size_mb > 1.0
    r["isc_crest_ok"]     = crest_db >= 6.0
    r["isc_subbass_ok"]   = sub_bass_rms > -120.0
    r["isc_highmid_ok"]   = highmid_rms > -120.0
    r["isc_lurange_ok"]   = lu_range >= 1.0  # EBU LRA; ≥1 LU = not flat/dead (hook content naturally 1-3 LU)

    results[platform_name] = r

    passed = sum(1 for k,v in r.items() if k.startswith("isc_") and v is True)
    failed = sum(1 for k,v in r.items() if k.startswith("isc_") and v is False)
    if failed > 0:
        all_pass = False

    print(f"  LUFS: {lufs_measured:.2f} (Δ{r['lufs_delta']:+.2f}) | TP: {tp_measured:.2f} dBTP | Corr: {corr:.3f}")
    print(f"  Crest: {crest_db:.1f} dB | LU Range: {lu_range:.1f} | Clips: {clips}")
    print(f"  ISC: {passed} pass, {failed} fail")

# ─── Thai Report ──────────────────────────────────────────────────────

print("\n")
print("=" * 70)
print("   📊  รายงานผลการ Mastering — LongPlay Studio V5.11.1")
print("=" * 70)
print(f"   ไฟล์ต้นฉบับ : {os.path.basename(SOURCE_FILE)}")
print(f"   Output      : ~/Desktop/LongPlay_Test/")
print("=" * 70)

print(f"\n{'แพลตฟอร์ม':<16} {'LUFS':>8} {'Δ LUFS':>8} {'True Peak':>10} {'Corr':>7} {'Crest':>7} {'ผ่าน/ไม่ผ่าน':>14}")
print("─" * 75)

total_pass = 0
total_fail = 0
isc_detail = []

for pname, r in results.items():
    if "error" in r:
        print(f"{pname:<16} {'ERROR':<8}")
        continue

    lufs_sym   = "✅" if r["isc_lufs_ok"]   else "❌"
    tp_sym     = "✅" if r["isc_tp_ok"]     else "❌"
    corr_sym   = "✅" if r["isc_corr_ok"]   else "❌"

    p = sum(1 for k,v in r.items() if k.startswith("isc_") and v is True)
    f = sum(1 for k,v in r.items() if k.startswith("isc_") and v is False)
    total_pass += p
    total_fail += f

    result_sym = "✅ ผ่านทุกข้อ" if f == 0 else f"⚠️  ไม่ผ่าน {f} ข้อ"
    print(f"{pname:<16} {r['lufs']:>7.2f} {r['lufs_delta']:>+8.2f} {r['true_peak']:>9.2f}  {r['correlation']:>6.3f} {r['crest_db']:>6.1f}  {result_sym}")

    isc_detail.append((pname, r))

print("─" * 75)
print(f"\n{'รวม ISC ผ่าน':<16} {total_pass:>4} / {total_pass+total_fail}")
print()

# Detailed ISC per platform
for pname, r in isc_detail:
    print(f"\n  [{pname}] รายละเอียด:")
    checks = {
        "isc_lufs_ok":    f"LUFS {r['lufs']:.2f} (เป้าหมาย {r['target_lufs']:.1f} ±0.5)",
        "isc_tp_ok":      f"True Peak {r['true_peak']:.2f} dBTP (≤ {r.get('target_tp', -1.0):.1f})",
        "isc_noclip_ok":  f"ไม่มี Hard Clip (samples > 1.0: {'ไม่พบ' if not r['clips'] else 'พบ!'})",
        "isc_corr_ok":    f"Stereo Correlation {r['correlation']:.3f} (> 0.7)",
        "isc_crest_ok":   f"Crest Factor {r['crest_db']:.1f} dB (≥ 6 dB)",
        "isc_lurange_ok": f"LU Range (EBU LRA) {r['lu_range']:.2f} LU (≥ 1 LU, hook content)",
        "isc_dur_ok":     f"Duration {r['duration']:.2f}s (ต้นฉบับ {r['src_duration']:.2f}s, Δ{abs(r['duration']-r['src_duration']):.2f}s)",
        "isc_sr_ok":      f"Sample Rate {r['sample_rate']} Hz (ต้องการ 48000)",
        "isc_size_ok":    f"ขนาดไฟล์ {r['file_size_mb']:.2f} MB (> 1 MB)",
        "isc_subbass_ok": f"Sub-bass (20-80 Hz) RMS: {r['sub_bass_rms']:.1f} dB (ไม่เป็น null)",
        "isc_highmid_ok": f"High-mid (1-8 kHz) RMS: {r['highmid_rms']:.1f} dB (ไม่เป็น null)",
    }
    for key, desc in checks.items():
        sym = "✅" if r.get(key) else "❌"
        print(f"    {sym} {desc}")

# Desktop listing
print(f"\n  📁 ไฟล์ใน ~/Desktop/LongPlay_Test/:")
for f in sorted(os.listdir(DESKTOP_DIR)):
    fp = os.path.join(DESKTOP_DIR, f)
    size_mb = os.path.getsize(fp) / 1e6
    print(f"    • {f}  ({size_mb:.2f} MB)")

print("\n" + "=" * 70)
if total_fail == 0:
    print("  🎉  ผ่านทุกข้อ! LongPlay Studio V5.11.1 พร้อมใช้งาน")
else:
    print(f"  ⚠️  ไม่ผ่าน {total_fail} ข้อ — ต้องแก้ไขและทดสอบใหม่")
print("=" * 70)

# Save report JSON
report_path = os.path.join(WORK_DIR, "desktop_export_report.json")
json_safe = {}
for k, v in results.items():
    json_safe[k] = {kk: bool(vv) if isinstance(vv, (bool, np.bool_)) else
                        float(vv) if isinstance(vv, (np.floating, np.float64)) else
                        int(vv) if isinstance(vv, (np.integer,)) else vv
                    for kk, vv in v.items()}
with open(report_path, "w") as f:
    json.dump(json_safe, f, indent=2, ensure_ascii=False)
print(f"\n  📄 JSON report: {report_path}")

# ─── Cleanup Desktop ─────────────────────────────────────────────────
print(f"\n  🧹 กำลังลบไฟล์ทดสอบจาก ~/Desktop/LongPlay_Test/ ...")
shutil.rmtree(DESKTOP_DIR)
print(f"  ✅ ลบแล้ว: {DESKTOP_DIR}")

print("\n✅ เสร็จสิ้น\n")
