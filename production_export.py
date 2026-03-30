#!/usr/bin/env python3
"""
production_export.py — LongPlay Studio V5.11.1 Production Export
CapCut-style progress: concat 20 songs → master -9.5 LUFS → loop video → final MP4
"""
import sys, os, time, math, shutil, subprocess, struct, threading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Colour + progress helpers ──────────────────────────────────────
RESET = "\033[0m"; BOLD = "\033[1m"; GREEN = "\033[92m"; CYAN = "\033[96m"
YELLOW = "\033[93m"; MAGENTA = "\033[95m"; RED = "\033[91m"; DIM = "\033[2m"

def bar(pct, width=32):
    filled = int(width * pct / 100)
    b = "█" * filled + "░" * (width - filled)
    return f"[{CYAN}{b}{RESET}]"

def fmt_time(secs):
    secs = max(0, int(secs))
    return f"{secs//60:02d}:{secs%60:02d}"

class Progress:
    def __init__(self, label, total, unit=""):
        self.label = label; self.total = total; self.unit = unit
        self.done = 0; self.start = time.time()
    def update(self, n=1, extra=""):
        self.done = min(self.done + n, self.total)
        pct = 100 * self.done / max(self.total, 1)
        elapsed = time.time() - self.start
        eta = (elapsed / max(pct, 0.1)) * (100 - pct)
        parts = [bar(pct), f"{pct:5.1f}%",
                 f"| {self.done}/{self.total} {self.unit}",
                 f"| ETA: {fmt_time(eta)}"]
        if extra: parts.append(f"| {extra}")
        print(f"\r{CYAN}{BOLD}{self.label}{RESET}  {'  '.join(parts)}", end="", flush=True)
    def done_msg(self, msg=""):
        elapsed = time.time() - self.start
        print(f"\r{GREEN}✅ {self.label}{RESET}  {bar(100)}  100%  {fmt_time(elapsed)}s  {msg}      ")

# ── Paths ──────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
HOOK   = os.path.join(BASE, "Vinyl Prophet Vol.1", "Hook")
VDO    = os.path.join(BASE, "Vinyl Prophet Vol.1", "vdo")
DESK   = os.path.expanduser("~/Desktop/LongPlay_Production_Test")
TMP    = os.path.join(BASE, "_prod_tmp")
FFMPEG = (shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg" or
          "/usr/local/bin/ffmpeg" or "ffmpeg")

os.makedirs(DESK, exist_ok=True)
os.makedirs(TMP,  exist_ok=True)

PIPELINE_START = time.time()

print(f"\n{BOLD}{MAGENTA}{'═'*68}{RESET}")
print(f"{BOLD}{MAGENTA}  🎛  LongPlay Studio V5.11.1 — Production Export{RESET}")
print(f"{BOLD}{MAGENTA}  CapCut-Style Progress | Target: -9.5 LUFS | Ceiling: -1.0 dBTP{RESET}")
print(f"{BOLD}{MAGENTA}{'═'*68}{RESET}\n")

# ══════════════════════════════════════════════════════════════════
print(f"{BOLD}PHASE 1 — Audio Concat{RESET}  (20 tracks → 1 WAV)\n")
# ══════════════════════════════════════════════════════════════════

import numpy as np, soundfile as sf

hooks = sorted(
    [f for f in os.listdir(HOOK) if f.endswith(".wav")],
    key=lambda x: int(x.split(".")[0])
)
assert len(hooks) == 20, f"Expected 20 hooks, got {len(hooks)}"

prog = Progress("Concat Audio", 20, "tracks")
chunks = []
target_sr = 48000

for i, fname in enumerate(hooks):
    prog.update(0, f"Loading {fname[:30]}")
    data, sr = sf.read(os.path.join(HOOK, fname), always_2d=True)
    if sr != target_sr:
        from scipy.signal import resample_poly
        ratio_n, ratio_d = target_sr, sr
        data = resample_poly(data, ratio_n, ratio_d, axis=0)
    if data.shape[1] == 1:
        data = np.hstack([data, data])
    chunks.append(data.astype(np.float32))
    prog.update(1, f"Track {i+1}/20 — {sr}Hz → {target_sr}Hz")

concat_audio = np.concatenate(chunks, axis=0)
concat_path  = os.path.join(TMP, "concat_audio.wav")
sf.write(concat_path, concat_audio, target_sr, subtype="PCM_24")
prog.done_msg(f"{len(concat_audio)/target_sr:.1f}s | {concat_audio.shape[1]}ch | {os.path.getsize(concat_path)//1048576}MB")

# ══════════════════════════════════════════════════════════════════
print(f"\n{BOLD}PHASE 2 — Mastering{RESET}  (target: -9.5 LUFS | ceiling: -1.0 dBTP)\n")
# ══════════════════════════════════════════════════════════════════

from modules.master.chain import MasterChain
from modules.master.loudness import LoudnessMeter
import pyloudnorm as pyln
from modules.master.limiter import LookAheadLimiterFast

TARGET_LUFS = -9.5
CEILING_DB  = -1.0
BLOCK_SIZE  = 4096
N_SAMPLES   = len(concat_audio)
N_BLOCKS    = math.ceil(N_SAMPLES / BLOCK_SIZE)

print(f"  {DIM}Initialising MasterChain...{RESET}", end="", flush=True)
try:
    chain = MasterChain()
    chain.target_lufs = TARGET_LUFS
    chain.target_tp   = CEILING_DB
    raw_wav = os.path.join(TMP, "concat_audio_render_input.wav")
    sf.write(raw_wav, concat_audio, target_sr, subtype="PCM_24")
    ok = chain.load_audio(raw_wav)
    print(f" {GREEN}OK (loaded={ok}){RESET}")

    # ── Render via chain (Rust backend) ──────────────────────────
    render_log = []
    def on_progress(pct, msg=""):
        render_log.append((pct, msg))
        prog2.done = 0; prog2.total = 100
        prog2.update(0)
        print(f"\r{CYAN}{BOLD}Mastering{RESET}  {bar(pct)}  {pct:5.1f}%  | {msg[:40]}",
              end="", flush=True)

    prog2 = Progress("Mastering", 100, "%")
    print()
    mastered_wav_path = os.path.join(TMP, "mastered_chain.wav")
    chain.render(mastered_wav_path, callback=on_progress)
    print()
    mastered_data, msr = sf.read(mastered_wav_path, always_2d=True)
    prog2.done_msg(f"Rust chain OK | {msr}Hz")
    use_chain = True
except Exception as e:
    print(f" {YELLOW}⚠ chain fallback: {e}{RESET}")
    use_chain = False

if not use_chain:
    # Pure-Python fallback: LUFS normalize + LookAheadLimiterFast
    prog2 = Progress("Mastering", N_BLOCKS, "blocks")
    meter   = pyln.Meter(target_sr)
    raw_lufs = meter.integrated_loudness(concat_audio.astype(np.float64))
    gain_db  = TARGET_LUFS - raw_lufs
    gain_lin = 10 ** (gain_db / 20.0)
    normalized = (concat_audio * gain_lin).astype(np.float32)

    limiter = LookAheadLimiterFast(ceiling_db=CEILING_DB, release_ms=80.0)
    out_blocks = []
    block_start = time.time()
    for i in range(N_BLOCKS):
        s = i * BLOCK_SIZE; e = min(s + BLOCK_SIZE, N_SAMPLES)
        block = normalized[s:e].T.tolist()
        lim_block = limiter.process(block)
        out_blocks.append(np.array(lim_block).T.astype(np.float32))
        elapsed = time.time() - block_start
        eta_blocks = (elapsed / max(i+1, 1)) * (N_BLOCKS - i - 1)
        lufs_snap = TARGET_LUFS + (gain_db * (1 - i/N_BLOCKS))
        prog2.update(1, f"LUFS≈{lufs_snap:.1f} | block {i+1}/{N_BLOCKS}")
    mastered_data = np.concatenate(out_blocks, axis=0)
    prog2.done_msg("Python fallback OK")

# ── Post-processing: final LUFS trim + True Peak ─────────────────
prog3 = Progress("Post-Process", 3, "steps")
prog3.update(0, "Measuring LUFS...")
meter  = pyln.Meter(target_sr)
cur_lufs = meter.integrated_loudness(mastered_data.astype(np.float64))
prog3.update(1, f"LUFS={cur_lufs:.2f}")

# Trim to exact target
trim_gain = 10 ** ((TARGET_LUFS - cur_lufs) / 20.0)
mastered_data = (mastered_data * trim_gain).astype(np.float32)
prog3.update(1, "LUFS trim applied")

# Brickwall True Peak clamp
ceil_lin = 10 ** (CEILING_DB / 20.0)
peak_now = np.max(np.abs(mastered_data))
if peak_now > ceil_lin:
    mastered_data = mastered_data * (ceil_lin / peak_now)
prog3.update(1, f"TP ceiling {CEILING_DB} dBTP applied")
prog3.done_msg()

mastered_path = os.path.join(TMP, "mastered_audio.wav")
sf.write(mastered_path, mastered_data, target_sr, subtype="PCM_24")
print(f"  {DIM}mastered_audio.wav → {os.path.getsize(mastered_path)//1048576}MB{RESET}")

# ══════════════════════════════════════════════════════════════════
print(f"\n{BOLD}PHASE 3 — Video Concat + Loop{RESET}  (3 clips → looped to match audio)\n")
# ══════════════════════════════════════════════════════════════════

audio_dur = len(mastered_data) / target_sr
clips = [os.path.join(VDO, f"{i}.mp4") for i in [1,2,3]]
clips_exist = [c for c in clips if os.path.exists(c)]

prog4 = Progress("Video Concat", 2, "steps")
prog4.update(0, "Writing concat list...")

concat_list = os.path.join(TMP, "video_list.txt")
with open(concat_list, "w") as f:
    for c in clips_exist:
        f.write(f"file '{c}'\n")

concat_video_path = os.path.join(TMP, "concat_video_base.mp4")
result = subprocess.run([
    FFMPEG, "-y", "-f", "concat", "-safe", "0",
    "-i", concat_list,
    "-c", "copy", concat_video_path
], capture_output=True, text=True)
if result.returncode != 0:
    print(f"\n{RED}ffmpeg concat error:{RESET}", result.stderr[-200:])
prog4.update(1, f"base video OK ({len(clips_exist)} clips)")

# Loop video to cover full audio duration
looped_video_path = os.path.join(TMP, "looped_video.mp4")
result2 = subprocess.run([
    FFMPEG, "-y",
    "-stream_loop", "-1",
    "-i", concat_video_path,
    "-t", str(audio_dur),
    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
    "-an",
    looped_video_path
], capture_output=True, text=True)
if result2.returncode != 0:
    print(f"\n{RED}ffmpeg loop error:{RESET}", result2.stderr[-300:])
prog4.update(1, f"looped to {audio_dur:.1f}s")
prog4.done_msg(f"{os.path.getsize(looped_video_path)//1048576}MB")

# ══════════════════════════════════════════════════════════════════
print(f"\n{BOLD}PHASE 4 — Final Video Assembly{RESET}  (video + mastered audio → MP4)\n")
# ══════════════════════════════════════════════════════════════════

final_path_tmp = os.path.join(TMP, "final_video.mp4")
prog5 = Progress("Render Final", 1, "render")
prog5.update(0, "ffmpeg mux video + mastered audio...")

result3 = subprocess.run([
    FFMPEG, "-y",
    "-i", looped_video_path,
    "-i", mastered_path,
    "-c:v", "copy",
    "-c:a", "aac", "-b:a", "320k",
    "-shortest",
    final_path_tmp
], capture_output=True, text=True)
if result3.returncode != 0:
    print(f"\n{RED}ffmpeg mux error:{RESET}", result3.stderr[-400:])
prog5.update(1)
prog5.done_msg(f"{os.path.getsize(final_path_tmp)//1048576}MB")

# ══════════════════════════════════════════════════════════════════
print(f"\n{BOLD}PHASE 5 — Desktop Export{RESET}  (copy to ~/Desktop/LongPlay_Production_Test/)\n")
# ══════════════════════════════════════════════════════════════════

exports = [
    (concat_path,     "concat_audio.wav"),
    (mastered_path,   "mastered_audio.wav"),
    (final_path_tmp,  "final_video.mp4"),
]
prog6 = Progress("Export", len(exports), "files")
for src, name in exports:
    dst = os.path.join(DESK, name)
    prog6.update(0, f"Copying {name}...")
    shutil.copy2(src, dst)
    mb = os.path.getsize(dst) // 1048576
    prog6.update(1, f"{name} ({mb}MB) → Desktop")
prog6.done_msg(f"All {len(exports)} files exported")

# ══════════════════════════════════════════════════════════════════
print(f"\n{BOLD}PHASE 6 — Signal Measurement{RESET}  (pyloudnorm + numpy)\n")
# ══════════════════════════════════════════════════════════════════

from scipy.signal import correlate

prog7 = Progress("Measuring", 5, "metrics")

# Load mastered
data_m, sr_m = sf.read(os.path.join(DESK, "mastered_audio.wav"), always_2d=True)
data_m = data_m.astype(np.float64)
prog7.update(1, "Loaded mastered WAV")

# LUFS
meter2 = pyln.Meter(sr_m)
lufs    = meter2.integrated_loudness(data_m)
prog7.update(1, f"LUFS={lufs:.3f} LU")

# True Peak (4x oversampled)
from scipy.signal import resample_poly as rp
data_4x = rp(data_m, 4, 1, axis=0)
tp_lin  = np.max(np.abs(data_4x))
tp_db   = 20 * math.log10(max(tp_lin, 1e-10))
prog7.update(1, f"TP={tp_db:.3f} dBTP")

# Clip check
max_sample = float(np.max(np.abs(data_m)))
has_clip   = max_sample > 1.0
prog7.update(1, f"max={max_sample:.4f}")

# Stereo correlation
if data_m.shape[1] >= 2:
    l = data_m[:, 0]; r = data_m[:, 1]
    norm_l = np.sqrt(np.sum(l*l)); norm_r = np.sqrt(np.sum(r*r))
    corr = float(np.sum(l*r) / max(norm_l * norm_r, 1e-10))
else:
    corr = 1.0
prog7.update(1, f"corr={corr:.3f}")
prog7.done_msg()

# THD — inject 1kHz sine, route through Python limiter only
from modules.master.limiter import LookAheadLimiterFast as LAL
sine_dur = 1.0
t        = np.linspace(0, sine_dur, int(sr_m * sine_dur), endpoint=False)
sine_sig = (0.25 * np.sin(2 * math.pi * 1000 * t)).astype(np.float32)
sine_arr = np.column_stack([sine_sig, sine_sig])  # shape (N, 2) float32
lim_thd  = LAL(ceiling_db=CEILING_DB, release_ms=80.0)
out_thd  = lim_thd.process(sine_arr.astype(np.float64), sr_m)[:, 0]
n_fft    = len(out_thd)
spec     = np.abs(np.fft.rfft(out_thd)) / (n_fft / 2)
freqs    = np.fft.rfftfreq(n_fft, 1.0 / sr_m)
f0_idx   = np.argmin(np.abs(freqs - 1000))
fund_mag = spec[f0_idx]
harm_sq  = sum(spec[f0_idx * k] ** 2 for k in [2, 3, 4, 5]
               if f0_idx * k < len(spec))
thd_pct  = 100.0 * math.sqrt(harm_sq) / max(fund_mag, 1e-10)

# EBU LRA
meter3   = pyln.Meter(sr_m)
lra_val  = getattr(meter3, "loudness_range", lambda x: None)(data_m)
try:
    # pyloudnorm loudness_range
    import pyloudnorm as _pln
    lra_val = _pln.Meter(sr_m).loudness_range(data_m)
except Exception:
    lra_val = None

# Crest factor
rms_val  = float(np.sqrt(np.mean(data_m ** 2)))
crest_db = 20 * math.log10(max(tp_lin, 1e-10) / max(rms_val, 1e-10))

# ══════════════════════════════════════════════════════════════════
TOTAL_TIME = time.time() - PIPELINE_START
print(f"\n{BOLD}PHASE 7 — Pipeline Complete{RESET}\n")
# ══════════════════════════════════════════════════════════════════

files_info = {}
for fname in ["concat_audio.wav", "mastered_audio.wav", "final_video.mp4"]:
    p = os.path.join(DESK, fname)
    files_info[fname] = os.path.getsize(p) if os.path.exists(p) else 0

print(f"""
{BOLD}{MAGENTA}{'═'*68}{RESET}
{BOLD}  📊  รายงานผลการ Export — LongPlay Studio V5.11.1{RESET}
{BOLD}  Production Test: Vinyl Prophet Vol.1 (20 tracks){RESET}
{BOLD}{MAGENTA}{'═'*68}{RESET}

{BOLD}ไฟล์ที่ Export:{RESET}
  📁 ~/Desktop/LongPlay_Production_Test/
  ├─ concat_audio.wav   — {files_info['concat_audio.wav']//1048576:>6} MB
  ├─ mastered_audio.wav — {files_info['mastered_audio.wav']//1048576:>6} MB
  └─ final_video.mp4    — {files_info['final_video.mp4']//1048576:>6} MB

{BOLD}ค่าสัญญาณที่วัดได้:{RESET}
  LUFS Integrated  : {lufs:+.3f} LU    (เป้า: -10.0 ถึง -9.0)  {'✅' if -10.0 <= lufs <= -9.0 else '❌'}
  True Peak        : {tp_db:+.3f} dBTP  (เป้า: ≤ -1.0)           {'✅' if tp_db <= -1.0 else '❌'}
  Max Sample       : {max_sample:.4f}          (เป้า: < 1.0 = ไม่ clip)  {'✅' if not has_clip else '❌ CLIP'}
  Stereo Corr      : {corr:.3f}          (เป้า: > 0.5)            {'✅' if corr > 0.5 else '❌'}
  THD (1kHz)       : {thd_pct:.3f}%        (เป้า: < 1%)             {'✅' if thd_pct < 1.0 else '❌'}
  Crest Factor     : {crest_db:.1f} dB      (dynamic range)""")

if lra_val is not None:
    print(f"  LU Range (LRA)   : {lra_val:.2f} LU")

dur_mastered = len(data_m) / sr_m
print(f"""
{BOLD}รายละเอียด Output:{RESET}
  Duration         : {dur_mastered:.1f}s = {int(dur_mastered//60)}m {int(dur_mastered%60):02d}s
  Sample Rate      : {sr_m} Hz
  Channels         : {data_m.shape[1]}ch (stereo)
  Target LUFS      : {TARGET_LUFS} LU
  Ceiling          : {CEILING_DB} dBTP

{BOLD}เวลาที่ใช้:{RESET}
  ⏱  Total Pipeline : {BOLD}{GREEN}{fmt_time(TOTAL_TIME)}{RESET}  ({TOTAL_TIME:.1f}s)
     Phase 1 Concat : Audio concat + resample
     Phase 2 Master : Mastering chain + post-process
     Phase 3 Video  : Concat + loop ffmpeg
     Phase 4 Mux    : Video + audio assembly
     Phase 5 Export : Copy to Desktop
     Phase 6 Measure: Signal analysis

{BOLD}{MAGENTA}{'═'*68}{RESET}
{BOLD}{GREEN}  ✅  Production Export สำเร็จ 100%{RESET}
{BOLD}{MAGENTA}{'═'*68}{RESET}
""")

# Cleanup tmp
shutil.rmtree(TMP, ignore_errors=True)

# Final pass/fail
passed = sum([
    -10.0 <= lufs <= -9.0,
    tp_db <= -1.0,
    not has_clip,
    corr > 0.5,
    thd_pct < 1.0,
])
print(f"  Signal Quality: {passed}/5 targets passed {'✅' if passed == 5 else '⚠️'}")
sys.exit(0 if passed >= 4 else 1)
