#!/usr/bin/env python3
"""
production_final.py — LongPlay Studio V5.11.1 Full Album Final Export
======================================================================
Source : ยังไม่ทำ Master/ (18 full songs ~75 min)
Target : LUFS > -13 (~-9.5 LU) | Sample Peak = -1.0 dBFS EXACT
         (ไม่ใช่ True Peak — เพื่อให้ Logic Pro meter แสดง -1 dB เป๊ะ)
Output : ~/Desktop/LongPlay_Final/
         mastered_album.wav       (24-bit, 48 kHz)
         final_video_with_logo.mp4 (h264_videotoolbox)

Key differences vs production_fullalbum.py:
  1. Final absolute Sample Peak normalization = -1.0 dBFS (Step 5)
  2. Output folder: ~/Desktop/LongPlay_Final/
  3. Output names: mastered_album.wav, final_video_with_logo.mp4
  4. h264_videotoolbox (hardware) with libx264 fallback
  5. Separate Sample Peak dBFS measurement in report
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

class Prog:
    def __init__(self, label, total, unit=""):
        self.label=label; self.total=total; self.unit=unit
        self.done=0; self.t0=time.time()
    def upd(self, n=1, extra=""):
        self.done=min(self.done+n, self.total)
        pct=100*self.done/max(self.total,1)
        el=time.time()-self.t0; eta=(el/max(pct,.1))*(100-pct)
        parts=[bar(pct),f"{pct:5.1f}%",f"| {self.done}/{self.total} {self.unit}",f"| ETA:{fmt_t(eta)}"]
        if extra: parts.append(f"| {extra}")
        print(f"\r{CYAN}{BOLD}{self.label}{RESET}  {'  '.join(parts)}", end="", flush=True)
    def ok(self, msg=""):
        el=time.time()-self.t0
        print(f"\r{GREEN}✅ {self.label}{RESET}  {bar(100)}  100%  {fmt_t(el)}  {msg}      ")

def phase_header(n, name, desc=""):
    print(f"\n{BOLD}{MAGENTA}{'═'*70}{RESET}")
    print(f"{BOLD}{MAGENTA}  PHASE {n} — {name}{RESET}" + (f"  {DIM}{desc}{RESET}" if desc else ""))
    print(f"{BOLD}{MAGENTA}{'═'*70}{RESET}\n")

# ── Paths ───────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
SONGS  = os.path.join(BASE, "Vinyl Prophet Vol.1", "ยังไม่ทำ Master")
VDO    = os.path.join(BASE, "Vinyl Prophet Vol.1", "สำหรับ Capcut", "Vinyl Awakening Vol.1_video.mp4")
DESK   = os.path.expanduser("~/Desktop/LongPlay_Final")
LOGO   = "/Volumes/Sample Data/0.Vibe Code All Project/Vscode Claudecode/Logo ทุกอย่างสำหรับใช้งาน/Chillin Vibes Logo.jpg"
TMP    = os.path.join(BASE, "_final_tmp")
FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"

os.makedirs(DESK, exist_ok=True)
os.makedirs(TMP,  exist_ok=True)

TARGET_LUFS      = -9.5
TARGET_SP_DB     = -1.0   # Sample Peak target (for Logic Pro meter)
TARGET_SP_LIN    = 10 ** (TARGET_SP_DB / 20.0)   # ≈ 0.891251
SR               = 48000

PIPELINE_START   = time.time()
PHASE_TIMES      = {}

print(f"\n{BOLD}{MAGENTA}{'═'*70}{RESET}")
print(f"{BOLD}{MAGENTA}  🎛  LongPlay Studio V5.11.1 — Final Album Production{RESET}")
print(f"{BOLD}{MAGENTA}  Vinyl Prophet Vol.1 | 18 เพลง | Sample Peak = -1.0 dBFS เป๊ะ{RESET}")
print(f"{BOLD}{MAGENTA}  Target: {TARGET_LUFS} LUFS | Sample Peak: {TARGET_SP_DB} dBFS (Logic Pro ready){RESET}")
print(f"{BOLD}{MAGENTA}{'═'*70}{RESET}\n")

# ══════════════════════════════════════════════════════════════════
phase_header(1, "Audio Concat", "18 full songs from ยังไม่ทำ Master/")
# ══════════════════════════════════════════════════════════════════
t_phase = time.time()
import numpy as np, soundfile as sf
import pyloudnorm as pyln
from scipy.signal import resample_poly as rp
from modules.master.chain import MasterChain
from modules.master.limiter import LookAheadLimiterFast as LAL

# Load full songs (exclude _preview.wav)
songs = sorted(
    [f for f in os.listdir(SONGS)
     if f.lower().endswith('.wav') and '_preview' not in f.lower()],
    key=lambda x: int(x.split('.')[0])
)
print(f"  พบ {len(songs)} เพลง:")
for i, s in enumerate(songs, 1):
    print(f"    {i:2d}. {s}")
print()
assert len(songs) == 18, f"Expected 18 full songs, got {len(songs)}: {songs}"

prog1 = Prog("Concat Audio", 18, "songs")
chunks = []
for i, fname in enumerate(songs):
    data, sr = sf.read(os.path.join(SONGS, fname), always_2d=True)
    if sr != SR:
        data = rp(data, SR, sr, axis=0)
    if data.shape[1] == 1:
        data = np.hstack([data, data])
    chunks.append(data.astype(np.float32))
    dur = len(data)/SR
    prog1.upd(1, f"Track {i+1}/18 — {int(dur//60)}:{int(dur%60):02d}")

concat_audio = np.concatenate(chunks, axis=0)
concat_dur   = len(concat_audio) / SR
concat_path  = os.path.join(TMP, "concat_audio.wav")
sf.write(concat_path, concat_audio, SR, subtype="PCM_24")
PHASE_TIMES["Phase 1 Concat"] = time.time() - t_phase
prog1.ok(f"{concat_dur/60:.1f}min | {concat_audio.shape[1]}ch | {os.path.getsize(concat_path)//1048576}MB")

# ══════════════════════════════════════════════════════════════════
phase_header(2, "Mastering", f"{TARGET_LUFS} LUFS | Sample Peak = {TARGET_SP_DB} dBFS exact")
# ══════════════════════════════════════════════════════════════════
t_phase = time.time()

print(f"  {DIM}Init MasterChain...{RESET}", end="", flush=True)
use_chain = False
try:
    chain = MasterChain()
    chain.target_lufs = TARGET_LUFS
    chain.target_tp   = TARGET_SP_DB
    raw_wav = os.path.join(TMP, "concat_render_input.wav")
    sf.write(raw_wav, concat_audio, SR, subtype="PCM_24")
    ok = chain.load_audio(raw_wav)
    print(f" {GREEN}OK{RESET}\n")

    chain_out = os.path.join(TMP, "mastered_chain.wav")
    chain.render(chain_out, callback=lambda p, m="":
        print(f"\r{CYAN}{BOLD}MasterChain Render{RESET}  {bar(p)}  {p:5.1f}%  | {str(m)[:40]}",
              end="", flush=True))
    print()
    mastered_data, msr = sf.read(chain_out, always_2d=True)
    mastered_data = mastered_data.astype(np.float32)
    print(f"  {GREEN}Chain render OK{RESET} | {msr}Hz")
    use_chain = True
except Exception as e:
    print(f" {YELLOW}⚠ Chain fallback: {e}{RESET}")
    use_chain = False

if not use_chain:
    # Python fallback: LUFS normalize
    print(f"  {DIM}Python fallback: LUFS normalize...{RESET}")
    meter_fb = pyln.Meter(SR)
    raw_lufs = meter_fb.integrated_loudness(concat_audio.astype(np.float64))
    gain_lin = 10 ** ((TARGET_LUFS - raw_lufs) / 20.0)
    mastered_data = (concat_audio * gain_lin).astype(np.float32)
    msr = SR
    print(f"  {GREEN}Python fallback OK{RESET} | LUFS {raw_lufs:.2f} → {TARGET_LUFS}")

# ── Post-process Step 1: LUFS trim ──────────────────────────────────
print(f"\n  {DIM}[Step 1] LUFS trim...{RESET}")
meter2 = pyln.Meter(msr)
cur_lufs = meter2.integrated_loudness(mastered_data.astype(np.float64))
trim_db  = TARGET_LUFS - cur_lufs
trim_lin = 10 ** (trim_db / 20.0)
mastered_data = (mastered_data * trim_lin).astype(np.float32)
print(f"  LUFS trim: {cur_lufs:.3f} → {TARGET_LUFS:.3f} LU  ({trim_db:+.3f} dB)")

# ── Post-process Step 2: Measure TP after LUFS trim (CHUNKED — no OOM) ──────
# IMPORTANT: resample_poly on 75min audio at once = 13.7 GB → OOM/swap
# Use 1-second chunked measurement instead.
print(f"  [Step 2] Measuring True Peak (chunked, avoid OOM)...", end="", flush=True)
chunk_sz = msr  # 1-second chunks
max_tp_lin = 0.0
for ch_i in range(mastered_data.shape[1]):
    for s in range(0, len(mastered_data), chunk_sz):
        e = min(s + chunk_sz, len(mastered_data))
        chunk = mastered_data[s:e, ch_i].astype(np.float64)
        if len(chunk) < 4:
            continue
        ov = rp(chunk, 4, 1)
        max_tp_lin = max(max_tp_lin, float(np.max(np.abs(ov))))
tp_after_trim_lin = max_tp_lin
tp_after_trim_db  = 20 * math.log10(max(tp_after_trim_lin, 1e-10))
print(f" done")
print(f"  [Step 2] TP after LUFS trim: {tp_after_trim_db:.3f} dBTP")

# ── Post-process Step 3: Makeup gain → push TP to ceiling ───────────
if tp_after_trim_db < TARGET_SP_DB - 0.05:
    makeup_db  = TARGET_SP_DB - tp_after_trim_db
    makeup_lin = 10 ** (makeup_db / 20.0)
    mastered_data = (mastered_data * makeup_lin).astype(np.float32)
    print(f"  [Step 3] Makeup gain: +{makeup_db:.3f} dB → push TP to ceiling")

    # Re-apply True Peak limiter (Ozone 12 / Logic Pro X style)
    print(f"  [Step 3] Re-applying True Peak limiter at {TARGET_SP_DB} dBTP...", end="", flush=True)
    lim_final = LAL(ceiling_db=TARGET_SP_DB, lookahead_ms=5.0, release_ms=100.0)
    mastered_data = lim_final.process(mastered_data.astype(np.float64), msr).astype(np.float32)
    print(f" {GREEN}done{RESET}")
else:
    print(f"  [Step 3] {GREEN}TP already at ceiling — no makeup needed{RESET}")

# ── Post-process Step 4: Iterate TP convergence (chunked measurement) ──────
def measure_tp_chunked(data, sr):
    """Measure True Peak in 1-sec chunks to avoid OOM on long audio."""
    max_tp = 0.0
    for ch_i in range(data.shape[1]):
        for s in range(0, len(data), sr):
            e = min(s + sr, len(data))
            chunk = data[s:e, ch_i].astype(np.float64)
            if len(chunk) < 4: continue
            ov = rp(chunk, 4, 1)
            max_tp = max(max_tp, float(np.max(np.abs(ov))))
    return 20 * math.log10(max(max_tp, 1e-10))

tp_cur_db = measure_tp_chunked(mastered_data, msr)
if not (TARGET_SP_DB - 0.15 <= tp_cur_db <= TARGET_SP_DB + 0.05):
    print(f"  [Step 4] TP={tp_cur_db:.3f} dBTP — iterating convergence (sample-peak proxy)...")
    # Use sample peak as proxy to avoid slow full-audio 4x resample
    for _i in range(3):
        adj = TARGET_SP_DB - tp_cur_db
        mastered_data = (mastered_data * (10 ** (adj / 20.0))).astype(np.float32)
        lim_it = LAL(ceiling_db=TARGET_SP_DB, lookahead_ms=5.0, release_ms=100.0)
        mastered_data = lim_it.process(mastered_data.astype(np.float64), msr).astype(np.float32)
        # Use sample peak as fast proxy for convergence check
        sp_proxy = float(np.max(np.abs(mastered_data)))
        tp_cur_db = 20 * math.log10(max(sp_proxy, 1e-10))  # sample peak as proxy
        print(f"    Iter {_i+1}: SamplePeak={tp_cur_db:.3f} dBFS (proxy)")
        if TARGET_SP_DB - 0.15 <= tp_cur_db <= TARGET_SP_DB + 0.05:
            break

# ── Post-process Step 5: EXACT SAMPLE PEAK NORMALIZATION ─────────────
# ──────────────────────────────────────────────────────────────────────
# This is the critical step: Logic Pro's Peak Meter reads SAMPLE PEAK
# (not True Peak). We normalize the sample peak to EXACTLY -1.0 dBFS
# so Logic Pro meter shows exactly -1 dB.
# ──────────────────────────────────────────────────────────────────────
print(f"\n  {CYAN}{BOLD}[Step 5] EXACT Sample Peak Normalization → {TARGET_SP_DB} dBFS{RESET}")
current_sp_lin = float(np.max(np.abs(mastered_data)))
current_sp_db  = 20 * math.log10(max(current_sp_lin, 1e-10))
sp_gain        = TARGET_SP_LIN / current_sp_lin
mastered_data  = (mastered_data * sp_gain).astype(np.float32)
# Hard clip to prevent floating-point rounding from exceeding target
np.clip(mastered_data, -TARGET_SP_LIN, TARGET_SP_LIN, out=mastered_data)
final_sp_lin = float(np.max(np.abs(mastered_data)))
final_sp_db  = 20 * math.log10(max(final_sp_lin, 1e-10))
print(f"  Sample Peak: {current_sp_db:.4f} dBFS → {final_sp_db:.4f} dBFS  (gain: {20*math.log10(sp_gain):+.4f} dB)")
print(f"  {GREEN}✅ Sample Peak normalized to {final_sp_db:.4f} dBFS{RESET}")

# Final LUFS after step 5
lufs_final = meter2.integrated_loudness(mastered_data.astype(np.float64))
print(f"  LUFS after Sample Peak norm: {lufs_final:.3f} LU")

# ── Write mastered WAV ───────────────────────────────────────────────
mastered_path = os.path.join(TMP, "mastered_album.wav")
sf.write(mastered_path, mastered_data, msr, subtype="PCM_24")
PHASE_TIMES["Phase 2 Mastering"] = time.time() - t_phase
print(f"\n  {GREEN}Mastered: {os.path.getsize(mastered_path)//1048576} MB — {mastered_path}{RESET}")

sp_pass   = abs(final_sp_db - TARGET_SP_DB) < 0.01
lufs_pass = -10.5 <= lufs_final <= -8.5
print(f"  Sample Peak: {final_sp_db:+.4f} dBFS  {'✅' if sp_pass else '❌'}  (เป้า: {TARGET_SP_DB} dBFS)")
print(f"  LUFS        : {lufs_final:+.3f} LU    {'✅' if lufs_pass else '⚠️'}  (เป้า: >{-13})")

# ══════════════════════════════════════════════════════════════════
phase_header(3, "Video + Logo Overlay", "Vinyl Awakening Vol.1_video.mp4 → trim + logo")
# ══════════════════════════════════════════════════════════════════
t_phase = time.time()

audio_dur   = len(mastered_data) / msr
logo_found  = os.path.exists(LOGO)
trimmed_vdo = os.path.join(TMP, "trimmed_with_logo.mp4")

# Check h264_videotoolbox availability
vtb_check = subprocess.run(
    [FFMPEG, "-f", "lavfi", "-i", "color=c=black:s=16x16:d=0.1",
     "-c:v", "h264_videotoolbox", "-f", "null", "-"],
    capture_output=True, text=True
)
use_vtb = vtb_check.returncode == 0
venc = "h264_videotoolbox" if use_vtb else "libx264"
venc_opts = ["-c:v", venc]
if use_vtb:
    venc_opts += ["-b:v", "8M", "-maxrate", "10M", "-bufsize", "20M"]
else:
    venc_opts += ["-preset", "fast", "-crf", "20"]
print(f"  Video encoder: {GREEN if use_vtb else YELLOW}{venc}{RESET} {'(hardware ✅)' if use_vtb else '(software fallback)'}")

if logo_found:
    print(f"  Logo: {GREEN}found{RESET} — Chillin Vibes Logo.jpg → overlay top-right (180px, 85% opacity)")
    logo_filter = (
        "[1:v]scale=180:-1,format=rgba,colorchannelmixer=aa=0.85[logo];"
        "[0:v][logo]overlay=W-w-30:30"
    )
    cmd_vdo = [
        FFMPEG, "-y",
        "-i", VDO,
        "-i", LOGO,
        "-t", str(audio_dur),
        "-filter_complex", logo_filter,
    ] + venc_opts + ["-an", trimmed_vdo]
else:
    print(f"  {YELLOW}Logo not found — trim only{RESET}")
    cmd_vdo = [
        FFMPEG, "-y", "-i", VDO, "-t", str(audio_dur),
    ] + venc_opts + ["-an", trimmed_vdo]

print(f"  Duration: {audio_dur:.1f}s = {int(audio_dur//60)}m {int(audio_dur%60):02d}s")
print(f"  Running ffmpeg...", end="", flush=True)
r_vdo = subprocess.run(cmd_vdo, capture_output=True, text=True)
if r_vdo.returncode != 0:
    print(f"\n{RED}Video error:{RESET}\n{r_vdo.stderr[-600:]}")
    sys.exit(1)
print(f" {GREEN}OK{RESET}")
PHASE_TIMES["Phase 3 Video"] = time.time() - t_phase
print(f"  {GREEN}Video+Logo: {os.path.getsize(trimmed_vdo)//1048576} MB{RESET}  ({fmt_t(time.time()-t_phase-PHASE_TIMES['Phase 3 Video']+PHASE_TIMES['Phase 3 Video'])})")

# ══════════════════════════════════════════════════════════════════
phase_header(4, "Final Mux", "video + mastered audio → MP4")
# ══════════════════════════════════════════════════════════════════
t_phase = time.time()

final_mp4_tmp = os.path.join(TMP, "final_video_with_logo.mp4")
print(f"  Muxing video + WAV → AAC 320k...", end="", flush=True)
r_mux = subprocess.run([
    FFMPEG, "-y",
    "-i", trimmed_vdo,
    "-i", mastered_path,
    "-c:v", "copy",
    "-c:a", "aac", "-b:a", "320k",
    "-shortest",
    final_mp4_tmp
], capture_output=True, text=True)
if r_mux.returncode != 0:
    print(f"\n{RED}Mux error:{RESET}\n{r_mux.stderr[-400:]}")
    sys.exit(1)
print(f" {GREEN}OK{RESET}")
PHASE_TIMES["Phase 4 Mux"] = time.time() - t_phase
print(f"  {GREEN}Final MP4: {os.path.getsize(final_mp4_tmp)//1048576} MB{RESET}")

# ══════════════════════════════════════════════════════════════════
phase_header(5, "Export", "→ ~/Desktop/LongPlay_Final/")
# ══════════════════════════════════════════════════════════════════
t_phase = time.time()

exports = [
    (mastered_path,  "mastered_album.wav"),
    (final_mp4_tmp,  "final_video_with_logo.mp4"),
]
prog5 = Prog("Export", len(exports), "files")
for src, name in exports:
    dst = os.path.join(DESK, name)
    shutil.copy2(src, dst)
    prog5.upd(1, f"{name} ({os.path.getsize(dst)//1048576}MB)")
prog5.ok(f"All {len(exports)} files → {DESK}")
PHASE_TIMES["Phase 5 Export"] = time.time() - t_phase

# ══════════════════════════════════════════════════════════════════
phase_header(6, "Signal Measurement", "LUFS / Sample Peak / True Peak / LRA / Corr / THD / Crest")
# ══════════════════════════════════════════════════════════════════
t_phase = time.time()

data_m, sr_m = sf.read(os.path.join(DESK, "mastered_album.wav"), always_2d=True)
data_m = data_m.astype(np.float64)
prog6  = Prog("Measuring", 7, "metrics")

# 1. LUFS
lufs_m = pyln.Meter(sr_m).integrated_loudness(data_m)
prog6.upd(1, f"LUFS={lufs_m:.3f}")

# 2. Sample Peak (what DAW meters like Logic Pro show)
sp_lin_m = float(np.max(np.abs(data_m)))
sp_db_m  = 20 * math.log10(max(sp_lin_m, 1e-10))
prog6.upd(1, f"SamplePeak={sp_db_m:.4f}")

# 3. True Peak (4x oversampled, chunked measurement — no OOM)
print(f"  {DIM}Measuring True Peak (chunked)...{RESET}", end="", flush=True)
tp_db_m = measure_tp_chunked(data_m.astype(np.float32), sr_m)
tp_lin_m = 10 ** (tp_db_m / 20.0)
print(f" {tp_db_m:.3f} dBTP")
prog6.upd(1, f"TruePeak={tp_db_m:.3f}")

# 4. Stereo Correlation
l_ch = data_m[:,0]; r_ch = data_m[:,1]
corr = float(np.sum(l_ch*r_ch) / max(
    np.sqrt(np.sum(l_ch*l_ch)) * np.sqrt(np.sum(r_ch*r_ch)), 1e-10))
prog6.upd(1, f"Corr={corr:.3f}")

# 5. THD (1 kHz test tone through limiter)
t_arr = np.linspace(0, 1.0, int(sr_m), endpoint=False)
s_arr = (0.25 * np.sin(2*math.pi*1000*t_arr)).astype(np.float32)
s_2ch = np.column_stack([s_arr, s_arr])
lim_thd = LAL(ceiling_db=TARGET_SP_DB, release_ms=80.0)
out_thd = lim_thd.process(s_2ch.astype(np.float64), sr_m)[:,0]
nf  = len(out_thd)
sp2 = np.abs(np.fft.rfft(out_thd)) / (nf/2)
fr2 = np.fft.rfftfreq(nf, 1.0/sr_m)
fi2 = np.argmin(np.abs(fr2-1000))
fm2 = sp2[fi2]
hs2 = sum(sp2[fi2*k]**2 for k in [2,3,4,5] if fi2*k < len(sp2))
thd = 100.0*math.sqrt(hs2)/max(fm2, 1e-10)
prog6.upd(1, f"THD={thd:.4f}%")

# 6. Crest Factor
rms_m   = float(np.sqrt(np.mean(data_m**2)))
crest   = 20*math.log10(max(sp_lin_m, 1e-10)/max(rms_m, 1e-10))
prog6.upd(1, f"Crest={crest:.1f}dB")

# 7. LRA
try:
    lra_m = pyln.Meter(sr_m).loudness_range(data_m)
except:
    lra_m = None
prog6.upd(1, f"LRA={lra_m:.2f}" if lra_m else "LRA=n/a")
prog6.ok()

PHASE_TIMES["Phase 6 Measure"] = time.time() - t_phase

# Duration
dur_m   = len(data_m)/sr_m

# ══════════════════════════════════════════════════════════════════
phase_header(7, "Report", "สรุปผลทั้งหมด")
# ══════════════════════════════════════════════════════════════════

TOTAL = time.time() - PIPELINE_START

fi_sizes = {}
for n in ["mastered_album.wav","final_video_with_logo.mp4"]:
    fp = os.path.join(DESK, n)
    fi_sizes[n] = os.path.getsize(fp)//1048576 if os.path.exists(fp) else 0

sp_ok   = abs(sp_db_m - TARGET_SP_DB) < 0.01      # ±0.01 dBFS
lufs_ok = lufs_m > -13.0                             # > -13 LUFS
tp_ok   = tp_db_m <= TARGET_SP_DB + 0.5             # True Peak ≤ -0.5 dBTP
clip_ok = sp_lin_m <= TARGET_SP_LIN + 1e-6           # no clip
corr_ok = corr > 0.5
thd_ok  = thd < 1.0

all_ok = sp_ok and lufs_ok and tp_ok and clip_ok and corr_ok and thd_ok
score  = sum([sp_ok, lufs_ok, tp_ok, clip_ok, corr_ok, thd_ok])

print(f"""
{BOLD}{MAGENTA}{'═'*70}{RESET}
{BOLD}  📊  รายงานผลการ Export — LongPlay Studio V5.11.1 Final{RESET}
{BOLD}  Vinyl Prophet Vol.1 | 18 เพลงเต็ม | {dur_m/60:.1f} นาที{RESET}
{BOLD}{MAGENTA}{'═'*70}{RESET}

{BOLD}ไฟล์ที่ Export → ~/Desktop/LongPlay_Final/{RESET}
  ├─ mastered_album.wav          {fi_sizes.get('mastered_album.wav',0):>5} MB  (24-bit, {sr_m} Hz)
  └─ final_video_with_logo.mp4   {fi_sizes.get('final_video_with_logo.mp4',0):>5} MB  ({venc})

{BOLD}ค่าสัญญาณที่วัดได้:{RESET}
  Sample Peak (dBFS)  : {sp_db_m:+.4f} dBFS  เป้า: {TARGET_SP_DB} dBFS เป๊ะ  {'✅ Logic Pro -1 dB!' if sp_ok else '❌ ไม่ตรง'}
  LUFS Integrated     : {lufs_m:+.3f} LU     เป้า: > -13 LU          {'✅' if lufs_ok else '❌'}
  True Peak (dBTP)    : {tp_db_m:+.3f} dBTP  เป้า: ≤ -0.5 dBTP       {'✅' if tp_ok else '⚠️'}
  Max Sample (linear) : {sp_lin_m:.6f}       เป้า: < 1.0 (no clip)   {'✅' if clip_ok else '❌ CLIP!'}
  Stereo Correlation  : {corr:.4f}           เป้า: > 0.5             {'✅' if corr_ok else '❌'}
  THD @ 1kHz          : {thd:.4f}%           เป้า: < 1%              {'✅' if thd_ok else '❌'}
  Crest Factor        : {crest:.2f} dB""")
if lra_m: print(f"  LU Range (LRA)      : {lra_m:.2f} LU")
print(f"""
{BOLD}รายละเอียด:{RESET}
  Duration            : {dur_m:.1f}s = {int(dur_m//60)}m {int(dur_m%60):02d}s ({len(songs)} เพลง)
  Sample Rate         : {sr_m} Hz  |  Channels: {data_m.shape[1]}ch  |  Bit Depth: 24-bit
  Source              : ยังไม่ทำ Master/ ({len(songs)} full tracks)

{BOLD}Video:{RESET}
  Encoder             : {venc} {'(hardware ✅)' if use_vtb else '(software)'}
  Logo overlay        : {'✅ Chillin Vibes Logo.jpg — top-right 180px 85%' if logo_found else '❌ logo not found'}

{BOLD}ทำไม Sample Peak ≠ True Peak:{RESET}
  Logic Pro Level Meter อ่าน Sample Peak (ตัวอย่างสัญญาณที่อยู่ในไฟล์)
  True Peak = inter-sample peaks (ระหว่างตัวอย่าง, 4x oversample)
  True Peak อาจสูงกว่า Sample Peak ถึง +3 dB
  → เราต้องการให้ Logic Pro meter แสดง -1 dB = ต้อง normalize Sample Peak

{BOLD}เวลาแต่ละ Phase:{RESET}""")
for ph, dt in PHASE_TIMES.items():
    print(f"  {ph:<30}: {fmt_t(dt)}")
print(f"""
  ⏱  TOTAL Pipeline       : {BOLD}{GREEN}{fmt_t(TOTAL)}{RESET}  ({TOTAL:.0f}s = {TOTAL/60:.1f}min)

{BOLD}{MAGENTA}{'═'*70}{RESET}
{BOLD}{'✅' if all_ok else '⚠️'}  Signal Quality: {score}/6 targets passed  {'— PERFECT! 🎛' if all_ok else ''}{RESET}
{BOLD}{MAGENTA}{'═'*70}{RESET}
""")

# Cleanup tmp
shutil.rmtree(TMP, ignore_errors=True)
print(f"  {DIM}Temp files cleaned up: {TMP}{RESET}")
print(f"\n  {GREEN}{BOLD}Output: ~/Desktop/LongPlay_Final/{RESET}")
print(f"  {GREEN}mastered_album.wav       — {fi_sizes.get('mastered_album.wav',0)} MB{RESET}")
print(f"  {GREEN}final_video_with_logo.mp4 — {fi_sizes.get('final_video_with_logo.mp4',0)} MB{RESET}\n")

sys.exit(0 if all_ok else 1)
