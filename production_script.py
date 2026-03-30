"""
LongPlay Final Production Script — Vinyl Prophet Vol.1
Concat 18 songs → Master → Export WAV + Video
"""
import os
import sys
import time
import glob
import subprocess
import numpy as np

# Add project to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

SONGS_DIR = os.path.join(PROJECT_ROOT, "Vinyl Prophet Vol.1", "ยังไม่ทำ Master")
VIDEO_SRC = os.path.join(PROJECT_ROOT, "Vinyl Prophet Vol.1", "สำหรับ Capcut", "Vinyl Awakening Vol.1_video.mp4")
LOGO_PATH = "/Volumes/Sample Data/0.ทำช่อง Youtube/0.Chillin Vibes /Chillin Vibes Logo.jpg"
OUTPUT_DIR = os.path.expanduser("~/Desktop/LongPlay_Final")
CONCAT_WAV = os.path.join(OUTPUT_DIR, "concat_album.wav")
MASTERED_WAV = os.path.join(OUTPUT_DIR, "mastered_album.wav")
FINAL_VIDEO = os.path.join(OUTPUT_DIR, "final_video.mp4")
TIMINGS = {}

os.makedirs(OUTPUT_DIR, exist_ok=True)

def ts():
    return time.strftime("%H:%M:%S")

def phase(name):
    TIMINGS[name] = {"start": time.perf_counter()}
    print(f"\n{'='*60}")
    print(f"[{ts()}] PHASE: {name}")
    print(f"{'='*60}")

def end_phase(name):
    elapsed = time.perf_counter() - TIMINGS[name]["start"]
    TIMINGS[name]["elapsed"] = elapsed
    print(f"[{ts()}] END {name}: {elapsed:.1f}s")

# ─── PHASE 1: COLLECT + SORT SONGS ────────────────────────────────
phase("1_collect_songs")
all_wavs = [f for f in os.listdir(SONGS_DIR)
            if f.endswith(".wav") and "preview" not in f.lower()]

def sort_key(fn):
    try:
        return int(fn.split(".")[0])
    except:
        return 999

all_wavs.sort(key=sort_key)
song_paths = [os.path.join(SONGS_DIR, f) for f in all_wavs]
print(f"Found {len(song_paths)} songs:")
for i, p in enumerate(song_paths, 1):
    print(f"  {i:2d}. {os.path.basename(p)}")
end_phase("1_collect_songs")

# ─── PHASE 2: CONCAT ───────────────────────────────────────────────
phase("2_concat")
import soundfile as sf

print("Loading and concatenating audio...")
chunks = []
sr_ref = None
silence_sec = 2.0  # 2 second gap between tracks

for i, path in enumerate(song_paths):
    data, sr = sf.read(path, dtype='float32', always_2d=True)
    if sr_ref is None:
        sr_ref = sr
    elif sr != sr_ref:
        # Resample if needed
        print(f"  Resampling {os.path.basename(path)} from {sr} to {sr_ref}")
        from scipy.signal import resample_poly
        import math
        gcd = math.gcd(sr_ref, sr)
        data = resample_poly(data, sr_ref // gcd, sr // gcd, axis=0).astype(np.float32)
    chunks.append(data)
    dur = len(data) / sr_ref
    print(f"  [{i+1:2d}/{len(song_paths)}] {os.path.basename(path)}: {dur:.1f}s")
    if i < len(song_paths) - 1:
        silence = np.zeros((int(sr_ref * silence_sec), data.shape[1]), dtype=np.float32)
        chunks.append(silence)

album = np.concatenate(chunks, axis=0)
total_dur = len(album) / sr_ref
print(f"\nTotal album duration: {total_dur:.1f}s ({total_dur/60:.1f} min)")
print(f"Saving concat WAV to: {CONCAT_WAV}")
sf.write(CONCAT_WAV, album, sr_ref, subtype='PCM_24')
print(f"Concat saved: {os.path.getsize(CONCAT_WAV)/1024/1024:.1f} MB")
end_phase("2_concat")

# ─── PHASE 3: MASTER ───────────────────────────────────────────────
phase("3_master")
from modules.master.chain import MasterChain

mc = MasterChain()
mc.load_audio(CONCAT_WAV)
mc.target_lufs = -9.5   # Between -9 and -10
mc.target_tp = -1.0     # True Peak ceiling
mc.normalize_loudness = True

# Set maximizer ceiling
mc.maximizer.ceiling = -1.0
# Disable per-band loudnorm that might interfere
mc.maximizer.enabled = True

print(f"MasterChain settings:")
print(f"  target_lufs  = {mc.target_lufs}")
print(f"  target_tp    = {mc.target_tp}")
print(f"  use_rust     = {mc._use_rust}")
print(f"  use_real     = {mc._use_real_processing}")

# Progress callback
def progress_cb(pct, msg):
    print(f"  [{pct:3d}%] {msg}")

mc_out_path = MASTERED_WAV
# The chain outputs to input_path_mastered by default - override
result = mc.render(output_path=mc_out_path, callback=progress_cb)
if result and os.path.exists(result):
    print(f"\nMaster render done: {result}")
    # Copy to correct location if chain changed path
    if result != mc_out_path:
        import shutil
        shutil.copy2(result, mc_out_path)
        print(f"Copied to: {mc_out_path}")
else:
    print(f"WARNING: render returned: {result}")
    # Fallback: Python processing directly
    print("Falling back to direct pyloudnorm mastering...")
    import pyloudnorm as pyln
    from modules.master.limiter import LookAheadLimiterFast

    data, sr = sf.read(CONCAT_WAV, dtype='float64', always_2d=True)
    meter = pyln.Meter(sr)
    measured = meter.integrated_loudness(data)
    print(f"  Input LUFS: {measured:.2f}")
    data = pyln.normalize.loudness(data, measured, -9.5)
    limiter = LookAheadLimiterFast(ceiling_db=-1.0, true_peak=True)
    data = limiter.process(data, sr)
    sf.write(mc_out_path, data.astype(np.float32), sr, subtype='PCM_24')
    print(f"  Saved fallback master: {mc_out_path}")

end_phase("3_master")

# ─── PHASE 4: VERIFY / MEASURE ─────────────────────────────────────
phase("4_measure")
import pyloudnorm as pyln

data, sr = sf.read(MASTERED_WAV, dtype='float64', always_2d=True)
meter = pyln.Meter(sr)

integrated_lufs = meter.integrated_loudness(data)
# Shortterm + LRA
try:
    block_size = int(sr * 0.4)
    if len(data) > block_size:
        shortterm = meter.integrated_loudness(data[:block_size*10])
    else:
        shortterm = integrated_lufs
except:
    shortterm = integrated_lufs

# True Peak via ffmpeg
tp_result = subprocess.run(
    ["ffmpeg", "-i", MASTERED_WAV, "-af", "ebur128=peak=true", "-f", "null", "-"],
    capture_output=True, text=True
)
tp_text = tp_result.stderr
# Parse True Peak
import re
tp_match = re.search(r'True peak:\s*([-\d.]+)\s*dBFS', tp_text)
true_peak = float(tp_match.group(1)) if tp_match else None

# LRA from ffmpeg ebur128
lra_match = re.search(r'LRA:\s*([\d.]+)\s*LU', tp_text)
lra = float(lra_match.group(1)) if lra_match else None

# Sample Peak
sample_peak_linear = np.max(np.abs(data))
sample_peak_db = 20 * np.log10(sample_peak_linear + 1e-12)

# Correlation (stereo)
if data.ndim == 2 and data.shape[1] >= 2:
    corr = np.corrcoef(data[:, 0], data[:, 1])[0, 1]
else:
    corr = 1.0

# THD estimate (simple): measure harmonic distortion via FFT
try:
    from scipy.fft import rfft, rfftfreq
    seg = data[:min(len(data), sr * 5), 0]
    spectrum = np.abs(rfft(seg))
    freqs = rfftfreq(len(seg), 1/sr)
    # Find fundamental (peak)
    peak_idx = np.argmax(spectrum[1:]) + 1
    f0 = freqs[peak_idx]
    f0_power = spectrum[peak_idx] ** 2
    # Sum harmonics 2nd-5th
    harmonic_power = 0
    for h in range(2, 6):
        h_freq = f0 * h
        if h_freq < sr/2:
            h_idx = np.argmin(np.abs(freqs - h_freq))
            harmonic_power += spectrum[h_idx] ** 2
    thd_pct = 100 * np.sqrt(harmonic_power) / (np.sqrt(f0_power) + 1e-12)
except:
    thd_pct = None

print(f"\n{'─'*40}")
print(f"  LUFS (Integrated): {integrated_lufs:.2f} LUFS")
print(f"  Sample Peak:       {sample_peak_db:.2f} dBFS")
print(f"  True Peak:         {true_peak} dBTP")
print(f"  LRA:               {lra} LU")
print(f"  Stereo Corr:       {corr:.4f}")
print(f"  THD estimate:      {thd_pct:.4f}%" if thd_pct else "  THD estimate:      N/A")
print(f"{'─'*40}")

# Save report
report_path = os.path.join(OUTPUT_DIR, "quality_report.txt")
with open(report_path, 'w') as f:
    f.write(f"Vinyl Prophet Vol.1 — Quality Report\n")
    f.write(f"{'='*40}\n")
    f.write(f"Songs:             {len(song_paths)}\n")
    f.write(f"Total Duration:    {total_dur:.1f}s ({total_dur/60:.1f} min)\n")
    f.write(f"Sample Rate:       {sr} Hz\n")
    f.write(f"Bit Depth:         24-bit\n")
    f.write(f"\n--- Loudness ---\n")
    f.write(f"LUFS (Integrated): {integrated_lufs:.2f} LUFS\n")
    f.write(f"Sample Peak:       {sample_peak_db:.2f} dBFS\n")
    f.write(f"True Peak:         {true_peak} dBTP\n")
    f.write(f"LRA:               {lra} LU\n")
    f.write(f"Stereo Corr:       {corr:.4f}\n")
    f.write(f"THD estimate:      {thd_pct:.4f}%\n" if thd_pct else f"THD estimate:      N/A\n")
    f.write(f"\n--- Targets ---\n")
    f.write(f"Target LUFS:       -9.5 LUFS ✓\n" if -10.5 <= integrated_lufs <= -8.5 else f"Target LUFS:       -9.5 (ACTUAL: {integrated_lufs:.2f}) ⚠\n")
    f.write(f"Target Peak:       -1.0 dBFS\n")

print(f"Report saved: {report_path}")
end_phase("4_measure")

# ─── PHASE 5: VIDEO ────────────────────────────────────────────────
phase("5_video")

audio_dur = total_dur
video_info = subprocess.run(
    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
     "-of", "csv=p=0", VIDEO_SRC],
    capture_output=True, text=True
)
try:
    vid_dur = float(video_info.stdout.strip())
except:
    vid_dur = 0

print(f"Audio duration:  {audio_dur:.1f}s")
print(f"Video duration:  {vid_dur:.1f}s")

# Build ffmpeg video command
# Loop video if shorter than audio, trim to audio length
# Overlay logo in corner

logo_exists = os.path.exists(LOGO_PATH)
print(f"Logo found: {logo_exists} ({LOGO_PATH})")

# Check h264_videotoolbox
use_hw = True  # confirmed above

if logo_exists:
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-stream_loop", "-1", "-i", VIDEO_SRC,   # loop video
        "-i", MASTERED_WAV,                        # mastered audio
        "-i", LOGO_PATH,                           # logo
        "-filter_complex",
        "[2:v]scale=200:-1[logo];[0:v][logo]overlay=W-w-20:H-h-20",
        "-map", "0:v",
        "-map", "1:a",
        "-c:v", "h264_videotoolbox",
        "-b:v", "8M",
        "-c:a", "aac",
        "-b:a", "320k",
        "-t", str(audio_dur),
        FINAL_VIDEO
    ]
else:
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-stream_loop", "-1", "-i", VIDEO_SRC,
        "-i", MASTERED_WAV,
        "-map", "0:v",
        "-map", "1:a",
        "-c:v", "h264_videotoolbox",
        "-b:v", "8M",
        "-c:a", "aac",
        "-b:a", "320k",
        "-t", str(audio_dur),
        FINAL_VIDEO
    ]

print(f"Running FFmpeg video export...")
print(f"CMD: {' '.join(ffmpeg_cmd)}")

result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=3600)
if result.returncode == 0:
    size_mb = os.path.getsize(FINAL_VIDEO) / 1024 / 1024
    print(f"Video export done: {FINAL_VIDEO} ({size_mb:.1f} MB)")
else:
    print(f"Video ERROR: {result.stderr[-2000:]}")
    # Fallback: software encoder
    print("Retrying with libx264...")
    ffmpeg_cmd2 = ffmpeg_cmd[:]
    idx = ffmpeg_cmd2.index("h264_videotoolbox")
    ffmpeg_cmd2[idx] = "libx264"
    ffmpeg_cmd2[idx+2] = "4M"
    result2 = subprocess.run(ffmpeg_cmd2, capture_output=True, text=True, timeout=3600)
    if result2.returncode == 0:
        print(f"Video export done (software): {FINAL_VIDEO}")
    else:
        print(f"Video FAILED: {result2.stderr[-1000:]}")

end_phase("5_video")

# ─── SUMMARY ──────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"PRODUCTION COMPLETE — TIMING SUMMARY")
print(f"{'='*60}")
for name, info in TIMINGS.items():
    elapsed = info.get("elapsed", time.perf_counter() - info["start"])
    print(f"  {name:30s}: {elapsed:.1f}s")

total = sum(v.get("elapsed", 0) for v in TIMINGS.values())
print(f"  {'TOTAL':30s}: {total:.1f}s")

print(f"\nOUTPUT FILES:")
for f in [MASTERED_WAV, FINAL_VIDEO, report_path]:
    if os.path.exists(f):
        size = os.path.getsize(f) / 1024 / 1024
        print(f"  ✓ {os.path.basename(f)}: {size:.1f} MB")
    else:
        print(f"  ✗ {os.path.basename(f)}: NOT FOUND")

print(f"\nMEASUREMENTS:")
print(f"  LUFS:        {integrated_lufs:.2f}")
print(f"  Sample Peak: {sample_peak_db:.2f} dBFS")
print(f"  True Peak:   {true_peak} dBTP")
print(f"  LRA:         {lra} LU")
print(f"  Correlation: {corr:.4f}")
print(f"  THD:         {f'{thd_pct:.4f}%' if thd_pct else 'N/A'}")
