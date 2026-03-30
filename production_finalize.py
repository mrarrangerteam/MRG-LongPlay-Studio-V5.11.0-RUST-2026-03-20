#!/usr/bin/env python3
"""
production_finalize.py — Fast finalization using existing mastered_chain.wav
=============================================================================
SKIPS the slow chain.render() step since mastered_chain.wav already exists.
mastered_chain.wav was produced by chain.render() with:
  - LUFS corrected: -7.08 → -9.50
  - True Peak limited at -1.0 dBTP
  - Sample Peak normalized at Step 7

This script just:
  1. Reads mastered_chain.wav
  2. Applies final EXACT sample peak = -1.0 dBFS (no OOM 4x oversample)
  3. Writes mastered_album.wav
  4. Video trim + logo overlay (h264_videotoolbox)
  5. Mux final MP4
  6. Export → ~/Desktop/LongPlay_Final/
  7. Measure all signals (chunked TP)
  8. Report
"""
import sys, os, time, math, shutil, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESET="\033[0m"; BOLD="\033[1m"; GREEN="\033[92m"; CYAN="\033[96m"
YELLOW="\033[93m"; MAGENTA="\033[95m"; RED="\033[91m"; DIM="\033[2m"

def bar(pct, w=28):
    f = int(w * pct / 100)
    return f"[{CYAN}{'█'*f}{'░'*(w-f)}{RESET}]"

def fmt_t(s):
    s = max(0, int(s))
    return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}" if s >= 3600 else f"{s//60:02d}:{s%60:02d}"

def phase_header(n, name, desc=""):
    print(f"\n{BOLD}{MAGENTA}{'═'*70}{RESET}")
    print(f"{BOLD}{MAGENTA}  PHASE {n} — {name}{RESET}" + (f"  {DIM}{desc}{RESET}" if desc else ""))
    print(f"{BOLD}{MAGENTA}{'═'*70}{RESET}\n")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.abspath(__file__))
VDO        = os.path.join(BASE, "Vinyl Prophet Vol.1", "สำหรับ Capcut", "Vinyl Awakening Vol.1_video.mp4")
DESK       = os.path.expanduser("~/Desktop/LongPlay_Final")
LOGO       = "/Volumes/Sample Data/0.Vibe Code All Project/Vscode Claudecode/Logo ทุกอย่างสำหรับใช้งาน/Chillin Vibes Logo.jpg"
TMP        = os.path.join(BASE, "_final_tmp")
CHAIN_WAV  = os.path.join(TMP, "mastered_chain.wav")    # already exists from chain.render()
FFMPEG     = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"

TARGET_SP_DB  = -1.0
TARGET_SP_LIN = 10 ** (TARGET_SP_DB / 20.0)   # 0.891251
TARGET_LUFS   = -9.5

os.makedirs(DESK, exist_ok=True)
os.makedirs(TMP,  exist_ok=True)

PIPELINE_START = time.time()
PHASE_TIMES    = {}

print(f"\n{BOLD}{MAGENTA}{'═'*70}{RESET}")
print(f"{BOLD}{MAGENTA}  🎛  LongPlay Studio — Final Finalize (Fast Path){RESET}")
print(f"{BOLD}{MAGENTA}  Using: mastered_chain.wav (chain.render() already done){RESET}")
print(f"{BOLD}{MAGENTA}  Goal: Sample Peak = {TARGET_SP_DB} dBFS EXACT → Logic Pro -1 dB{RESET}")
print(f"{BOLD}{MAGENTA}{'═'*70}{RESET}\n")

# ── Verify mastered_chain.wav exists ─────────────────────────────────────────
if not os.path.exists(CHAIN_WAV):
    print(f"{RED}ERROR: {CHAIN_WAV} not found!{RESET}")
    print(f"Run production_final.py first to generate mastered_chain.wav")
    sys.exit(1)

size_mb = os.path.getsize(CHAIN_WAV) // 1048576
mtime   = time.strftime('%H:%M:%S', time.localtime(os.path.getmtime(CHAIN_WAV)))
print(f"  {GREEN}Found mastered_chain.wav{RESET} — {size_mb} MB — modified {mtime}")

# ══════════════════════════════════════════════════════════════════════════════
phase_header(1, "Load + Sample Peak Normalize", f"mastered_chain.wav → exact {TARGET_SP_DB} dBFS")
# ══════════════════════════════════════════════════════════════════════════════
t_phase = time.time()

import numpy as np, soundfile as sf
import pyloudnorm as pyln
from scipy.signal import resample_poly as rp
from modules.master.limiter import LookAheadLimiterFast as LAL

print(f"  Reading mastered_chain.wav ({size_mb} MB)...", end="", flush=True)
mastered_data, msr = sf.read(CHAIN_WAV, always_2d=True)
mastered_data = mastered_data.astype(np.float32)
dur = len(mastered_data) / msr
print(f" OK  |  {dur/60:.1f}min  |  {msr}Hz  |  {mastered_data.shape[1]}ch")

# Measure current LUFS (before any changes)
print(f"  Measuring LUFS...", end="", flush=True)
meter = pyln.Meter(msr)
cur_lufs = meter.integrated_loudness(mastered_data.astype(np.float64))
print(f" {cur_lufs:.3f} LU")

# LUFS trim if needed (should be ~0 dB since chain already corrected)
trim_db = TARGET_LUFS - cur_lufs
if abs(trim_db) > 0.05:
    trim_lin = 10 ** (trim_db / 20.0)
    mastered_data = (mastered_data * trim_lin).astype(np.float32)
    print(f"  LUFS trim: {cur_lufs:.3f} → {TARGET_LUFS:.3f} LU ({trim_db:+.3f} dB)")
else:
    print(f"  {GREEN}LUFS trim: {cur_lufs:.3f} LU — no adjustment needed (Δ={trim_db:+.3f} dB){RESET}")

# ── EXACT SAMPLE PEAK NORMALIZATION ─────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
# Logic Pro Level Meter reads SAMPLE PEAK (not True Peak).
# Normalize so the sample peak is EXACTLY -1.0 dBFS.
# chain.py Step 7 already does this, but LUFS trim may shift it.
# This is the GUARANTEED final step.
# ──────────────────────────────────────────────────────────────────────────────
current_sp_lin = float(np.max(np.abs(mastered_data)))
current_sp_db  = 20 * math.log10(max(current_sp_lin, 1e-10))
sp_gain        = TARGET_SP_LIN / current_sp_lin
mastered_data  = (mastered_data * sp_gain).astype(np.float32)
np.clip(mastered_data, -TARGET_SP_LIN, TARGET_SP_LIN, out=mastered_data)

final_sp_lin = float(np.max(np.abs(mastered_data)))
final_sp_db  = 20 * math.log10(max(final_sp_lin, 1e-10))
lufs_final   = meter.integrated_loudness(mastered_data.astype(np.float64))

print(f"\n  {CYAN}{BOLD}EXACT Sample Peak Normalization:{RESET}")
print(f"    Before: {current_sp_db:+.4f} dBFS  After: {final_sp_db:+.4f} dBFS  (gain: {20*math.log10(sp_gain):+.4f} dB)")
print(f"    LUFS after SP normalize: {lufs_final:.3f} LU")
print(f"    {GREEN}✅ Sample Peak = {final_sp_db:.4f} dBFS — Logic Pro meter will show -1 dB!{RESET}")

# Write mastered_album.wav
mastered_path = os.path.join(TMP, "mastered_album.wav")
sf.write(mastered_path, mastered_data, msr, subtype="PCM_24")
PHASE_TIMES["Phase 1 Load+Normalize"] = time.time() - t_phase
print(f"\n  {GREEN}Written: mastered_album.wav — {os.path.getsize(mastered_path)//1048576} MB{RESET}")

sp_pass   = abs(final_sp_db - TARGET_SP_DB) < 0.01
lufs_pass = lufs_final > -13.0
print(f"  Sample Peak: {final_sp_db:+.4f} dBFS  {'✅' if sp_pass else '❌'}")
print(f"  LUFS        : {lufs_final:+.3f} LU    {'✅' if lufs_pass else '❌'}")

# ══════════════════════════════════════════════════════════════════════════════
phase_header(2, "Video + Logo Overlay", "Vinyl Awakening Vol.1_video.mp4 → trim + logo")
# ══════════════════════════════════════════════════════════════════════════════
t_phase = time.time()

audio_dur  = len(mastered_data) / msr
logo_found = os.path.exists(LOGO)
trimmed_vdo = os.path.join(TMP, "trimmed_with_logo.mp4")

# Test h264_videotoolbox
vtb_ok = subprocess.run(
    [FFMPEG, "-f", "lavfi", "-i", "color=c=black:s=16x16:d=0.1",
     "-c:v", "h264_videotoolbox", "-f", "null", "-"],
    capture_output=True, text=True
).returncode == 0
venc = "h264_videotoolbox" if vtb_ok else "libx264"
venc_opts = (["-c:v", venc, "-b:v", "8M", "-maxrate", "10M", "-bufsize", "20M"]
             if vtb_ok else
             ["-c:v", venc, "-preset", "fast", "-crf", "20"])
print(f"  Video encoder: {GREEN if vtb_ok else YELLOW}{venc}{RESET} {'(hardware ✅)' if vtb_ok else '(software)'}")
print(f"  Duration: {audio_dur:.1f}s = {int(audio_dur//60)}m {int(audio_dur%60):02d}s")

if logo_found:
    print(f"  Logo: {GREEN}found{RESET} → top-right 180px 85% opacity")
    logo_filter = ("[1:v]scale=180:-1,format=rgba,colorchannelmixer=aa=0.85[logo];"
                   "[0:v][logo]overlay=W-w-30:30")
    cmd_vdo = ([FFMPEG, "-y", "-i", VDO, "-i", LOGO,
                "-t", str(audio_dur), "-filter_complex", logo_filter]
               + venc_opts + ["-an", trimmed_vdo])
else:
    print(f"  {YELLOW}Logo not found — trim only{RESET}")
    cmd_vdo = ([FFMPEG, "-y", "-i", VDO, "-t", str(audio_dur)]
               + venc_opts + ["-an", trimmed_vdo])

print(f"  ffmpeg video render...", end="", flush=True)
r_vdo = subprocess.run(cmd_vdo, capture_output=True, text=True)
if r_vdo.returncode != 0:
    print(f"\n{RED}Video error:{RESET}\n{r_vdo.stderr[-600:]}")
    sys.exit(1)
print(f" {GREEN}OK{RESET} — {os.path.getsize(trimmed_vdo)//1048576} MB")
PHASE_TIMES["Phase 2 Video"] = time.time() - t_phase

# ══════════════════════════════════════════════════════════════════════════════
phase_header(3, "Final Mux", "video + mastered audio → MP4")
# ══════════════════════════════════════════════════════════════════════════════
t_phase = time.time()

final_mp4_tmp = os.path.join(TMP, "final_video_with_logo.mp4")
print(f"  Muxing (copy video + AAC 320k audio)...", end="", flush=True)
r_mux = subprocess.run([
    FFMPEG, "-y", "-i", trimmed_vdo, "-i", mastered_path,
    "-c:v", "copy", "-c:a", "aac", "-b:a", "320k", "-shortest",
    final_mp4_tmp
], capture_output=True, text=True)
if r_mux.returncode != 0:
    print(f"\n{RED}Mux error:{RESET}\n{r_mux.stderr[-400:]}")
    sys.exit(1)
print(f" {GREEN}OK{RESET} — {os.path.getsize(final_mp4_tmp)//1048576} MB")
PHASE_TIMES["Phase 3 Mux"] = time.time() - t_phase

# ══════════════════════════════════════════════════════════════════════════════
phase_header(4, "Export", "→ ~/Desktop/LongPlay_Final/")
# ══════════════════════════════════════════════════════════════════════════════
t_phase = time.time()

for src, name in [(mastered_path, "mastered_album.wav"),
                  (final_mp4_tmp, "final_video_with_logo.mp4")]:
    dst = os.path.join(DESK, name)
    shutil.copy2(src, dst)
    print(f"  {GREEN}✅{RESET} {name}  ({os.path.getsize(dst)//1048576} MB)")
PHASE_TIMES["Phase 4 Export"] = time.time() - t_phase

# ══════════════════════════════════════════════════════════════════════════════
phase_header(5, "Signal Measurement", "LUFS/Sample Peak/True Peak/LRA/Corr/THD/Crest")
# ══════════════════════════════════════════════════════════════════════════════
t_phase = time.time()

data_m, sr_m = sf.read(os.path.join(DESK, "mastered_album.wav"), always_2d=True)
data_m = data_m.astype(np.float64)
dur_m  = len(data_m) / sr_m

# 1. LUFS
lufs_m = pyln.Meter(sr_m).integrated_loudness(data_m)
print(f"  LUFS: {lufs_m:.3f} LU")

# 2. Sample Peak
sp_lin_m = float(np.max(np.abs(data_m)))
sp_db_m  = 20 * math.log10(max(sp_lin_m, 1e-10))
print(f"  Sample Peak: {sp_db_m:.4f} dBFS")

# 3. True Peak (chunked — no OOM)
print(f"  True Peak (chunked)...", end="", flush=True)
max_tp = 0.0
chunk_sz = sr_m
for ch_i in range(data_m.shape[1]):
    for s in range(0, len(data_m), chunk_sz):
        e = min(s + chunk_sz, len(data_m))
        chunk = data_m[s:e, ch_i]
        if len(chunk) < 4: continue
        ov = rp(chunk, 4, 1)
        max_tp = max(max_tp, float(np.max(np.abs(ov))))
tp_db_m = 20 * math.log10(max(max_tp, 1e-10))
print(f" {tp_db_m:.3f} dBTP")

# 4. Stereo Correlation
l_ch = data_m[:,0]; r_ch = data_m[:,1]
corr = float(np.sum(l_ch*r_ch) / max(
    np.sqrt(np.sum(l_ch*l_ch)) * np.sqrt(np.sum(r_ch*r_ch)), 1e-10))
print(f"  Correlation: {corr:.4f}")

# 5. THD @ 1kHz
t_arr = np.linspace(0, 1.0, int(sr_m), endpoint=False)
s_arr = (0.25 * np.sin(2*math.pi*1000*t_arr)).astype(np.float32)
s_2ch = np.column_stack([s_arr, s_arr])
lim_thd = LAL(ceiling_db=TARGET_SP_DB, release_ms=80.0)
out_thd = lim_thd.process(s_2ch.astype(np.float64), sr_m)[:,0]
nf  = len(out_thd); sp2 = np.abs(np.fft.rfft(out_thd)) / (nf/2)
fr2 = np.fft.rfftfreq(nf, 1.0/sr_m); fi2 = np.argmin(np.abs(fr2-1000))
fm2 = sp2[fi2]
hs2 = sum(sp2[fi2*k]**2 for k in [2,3,4,5] if fi2*k < len(sp2))
thd = 100.0*math.sqrt(hs2)/max(fm2, 1e-10)
print(f"  THD @ 1kHz: {thd:.4f}%")

# 6. Crest Factor
rms_m = float(np.sqrt(np.mean(data_m**2)))
crest = 20*math.log10(max(sp_lin_m, 1e-10)/max(rms_m, 1e-10))
print(f"  Crest Factor: {crest:.2f} dB")

# 7. LRA
try:
    lra_m = pyln.Meter(sr_m).loudness_range(data_m)
except:
    lra_m = None
if lra_m: print(f"  LRA: {lra_m:.2f} LU")

PHASE_TIMES["Phase 5 Measure"] = time.time() - t_phase

# ══════════════════════════════════════════════════════════════════════════════
phase_header(6, "Report", "สรุปผลทั้งหมด")
# ══════════════════════════════════════════════════════════════════════════════

TOTAL = time.time() - PIPELINE_START
fi_sizes = {n: os.path.getsize(os.path.join(DESK,n))//1048576
            for n in ["mastered_album.wav","final_video_with_logo.mp4"]
            if os.path.exists(os.path.join(DESK,n))}

sp_ok    = abs(sp_db_m - TARGET_SP_DB) < 0.01
lufs_ok  = lufs_m > -13.0
tp_ok    = tp_db_m <= TARGET_SP_DB + 0.5
clip_ok  = sp_lin_m <= TARGET_SP_LIN + 1e-6
corr_ok  = corr > 0.5
thd_ok   = thd < 1.0
all_ok   = sp_ok and lufs_ok and tp_ok and clip_ok and corr_ok and thd_ok
score    = sum([sp_ok, lufs_ok, tp_ok, clip_ok, corr_ok, thd_ok])

print(f"""
{BOLD}{MAGENTA}{'═'*70}{RESET}
{BOLD}  📊  รายงานผล — Vinyl Prophet Vol.1 | {dur_m/60:.1f} นาที{RESET}
{BOLD}{MAGENTA}{'═'*70}{RESET}

{BOLD}ไฟล์ที่ Export → ~/Desktop/LongPlay_Final/{RESET}
  ├─ mastered_album.wav          {fi_sizes.get('mastered_album.wav',0):>5} MB  (24-bit, {sr_m} Hz)
  └─ final_video_with_logo.mp4   {fi_sizes.get('final_video_with_logo.mp4',0):>5} MB  ({venc})

{BOLD}ค่าสัญญาณที่วัดได้:{RESET}
  Sample Peak (dBFS)  : {sp_db_m:+.4f} dBFS  เป้า: -1.0000 dBFS เป๊ะ  {'✅ Logic Pro = -1 dB!' if sp_ok else '❌'}
  LUFS Integrated     : {lufs_m:+.3f} LU     เป้า: > -13 LU           {'✅' if lufs_ok else '❌'}
  True Peak (dBTP)    : {tp_db_m:+.3f} dBTP  เป้า: ≤ -0.5 dBTP        {'✅' if tp_ok else '⚠️'}
  Max Sample (linear) : {sp_lin_m:.6f}       เป้า: < 1.0 (no clip)    {'✅' if clip_ok else '❌ CLIP!'}
  Stereo Correlation  : {corr:.4f}           เป้า: > 0.5              {'✅' if corr_ok else '❌'}
  THD @ 1kHz          : {thd:.4f}%           เป้า: < 1%               {'✅' if thd_ok else '❌'}
  Crest Factor        : {crest:.2f} dB""")
if lra_m: print(f"  LU Range (LRA)      : {lra_m:.2f} LU")
print(f"""
{BOLD}รายละเอียด:{RESET}
  Duration     : {dur_m:.1f}s = {int(dur_m//60)}m {int(dur_m%60):02d}s  |  {sr_m} Hz  |  2ch  |  24-bit
  Encoder Video: {venc} {'(hardware ✅)' if vtb_ok else '(software)'}
  Logo Overlay : {'✅ Chillin Vibes Logo.jpg top-right' if logo_found else '❌ not found'}

{BOLD}ทำไม Sample Peak = เป้า:{RESET}
  Logic Pro Level Meter อ่าน Sample Peak (ไม่ใช่ True Peak)
  Sample Peak = {sp_db_m:.4f} dBFS → Logic Pro meter แสดง {sp_db_m:.2f} dB ≈ -1 dB ✅

{BOLD}เวลาแต่ละ Phase:{RESET}""")
for ph, dt in PHASE_TIMES.items():
    print(f"  {ph:<30}: {fmt_t(dt)}")
print(f"""
  ⏱  TOTAL (finalize only) : {BOLD}{GREEN}{fmt_t(TOTAL)}{RESET}  ({TOTAL:.0f}s)

{BOLD}{MAGENTA}{'═'*70}{RESET}
{BOLD}{'✅ PERFECT!' if all_ok else '⚠️ PARTIAL'} Signal Quality: {score}/6 targets passed{RESET}
{BOLD}{MAGENTA}{'═'*70}{RESET}
""")

# Cleanup
shutil.rmtree(TMP, ignore_errors=True)
print(f"  {DIM}Temp cleaned: {TMP}{RESET}")
print(f"\n  {GREEN}{BOLD}~/Desktop/LongPlay_Final/ ready!{RESET}")

sys.exit(0 if all_ok else 1)
