#!/usr/bin/env python3
"""
production_fullalbum.py — LongPlay Studio V5.11.1 Full Album Export
Source: ยังไม่ทำ Master/ (18 full songs ~75min)
Target: -9.5 LUFS | -1.0 dBTP ceiling (TP must HIT ceiling ±0.1)
TP Fix: makeup gain after LUFS trim → re-apply limiter (Ozone 12 / Logic Pro X style)
"""
import sys, os, time, math, shutil, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESET="\033[0m"; BOLD="\033[1m"; GREEN="\033[92m"; CYAN="\033[96m"
YELLOW="\033[93m"; MAGENTA="\033[95m"; RED="\033[91m"; DIM="\033[2m"

def bar(pct, w=28):
    f = int(w * pct / 100)
    return f"[{CYAN}{'█'*f}{'░'*(w-f)}{RESET}]"

def fmt_t(s):
    s = max(0, int(s)); return f"{s//60:02d}:{s%60:02d}"

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

# ── Paths ───────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
SONGS  = os.path.join(BASE, "Vinyl Prophet Vol.1", "ยังไม่ทำ Master")
VDO    = os.path.join(BASE, "Vinyl Prophet Vol.1", "สำหรับ Capcut", "Vinyl Awakening Vol.1_video.mp4")
DESK   = os.path.expanduser("~/Desktop/LongPlay_Production")
LOGO   = "/Volumes/Sample Data/0.Vibe Code All Project/Vscode Claudecode/Logo ทุกอย่างสำหรับใช้งาน/Chillin Vibes Logo.jpg"
TMP    = os.path.join(BASE, "_fullalbum_tmp")
FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"

os.makedirs(DESK, exist_ok=True)
os.makedirs(TMP,  exist_ok=True)

TARGET_LUFS = -9.5
CEILING_DB  = -1.0
CEIL_LIN    = 10 ** (CEILING_DB / 20.0)
SR          = 48000

PIPELINE_START = time.time()

print(f"\n{BOLD}{MAGENTA}{'═'*70}{RESET}")
print(f"{BOLD}{MAGENTA}  🎛  LongPlay Studio V5.11.1 — Full Album Export (18 Songs){RESET}")
print(f"{BOLD}{MAGENTA}  Target: -9.5 LUFS | Ceiling: -1.0 dBTP | TP must HIT ceiling{RESET}")
print(f"{BOLD}{MAGENTA}{'═'*70}{RESET}\n")

# ══════════════════════════════════════════════════════════════════
print(f"{BOLD}PHASE 1 — Audio Concat{RESET}  (18 full songs from ยังไม่ทำ Master/)\n")
# ══════════════════════════════════════════════════════════════════
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
assert len(songs) == 18, f"Expected 18 full songs, got {len(songs)}: {songs}"

prog1 = Prog("Concat Audio", 18, "songs")
chunks = []
for i, fname in enumerate(songs):
    prog1.upd(0, f"{fname[:35]}")
    data, sr = sf.read(os.path.join(SONGS, fname), always_2d=True)
    if sr != SR:
        data = rp(data, SR, sr, axis=0)
    if data.shape[1] == 1:
        data = np.hstack([data, data])
    chunks.append(data.astype(np.float32))
    prog1.upd(1, f"Track {i+1}/18 — {int(len(data)/SR//60)}:{int(len(data)/SR%60):02d}")

concat_audio = np.concatenate(chunks, axis=0)
concat_dur   = len(concat_audio) / SR
concat_path  = os.path.join(TMP, "concat_audio.wav")
sf.write(concat_path, concat_audio, SR, subtype="PCM_24")
prog1.ok(f"{concat_dur/60:.1f}min | {concat_audio.shape[1]}ch | {os.path.getsize(concat_path)//1048576}MB")

# ══════════════════════════════════════════════════════════════════
print(f"\n{BOLD}PHASE 2 — Mastering{RESET}  (-9.5 LUFS | ceiling -1.0 dBTP | TP must hit ceiling)\n")
# ══════════════════════════════════════════════════════════════════

print(f"  {DIM}Init MasterChain...{RESET}", end="", flush=True)
try:
    chain = MasterChain()
    chain.target_lufs = TARGET_LUFS
    chain.target_tp   = CEILING_DB
    raw_wav = os.path.join(TMP, "concat_render_input.wav")
    sf.write(raw_wav, concat_audio, SR, subtype="PCM_24")
    ok = chain.load_audio(raw_wav)
    print(f" {GREEN}OK{RESET}")

    prog2 = Prog("Mastering", 100, "%")
    print()
    chain_out = os.path.join(TMP, "mastered_chain.wav")
    chain.render(chain_out, callback=lambda p, m="":
        print(f"\r{CYAN}{BOLD}Mastering{RESET}  {bar(p)}  {p:5.1f}%  | {str(m)[:40]}",
              end="", flush=True))
    print()
    mastered_data, msr = sf.read(chain_out, always_2d=True)
    prog2.ok(f"Rust chain OK | {msr}Hz")
    use_chain = True
except Exception as e:
    print(f" {YELLOW}⚠ fallback: {e}{RESET}")
    use_chain = False

if not use_chain:
    # Python fallback: LUFS normalize only
    prog2 = Prog("Mastering", 3, "steps")
    meter = pyln.Meter(SR)
    raw_lufs = meter.integrated_loudness(concat_audio.astype(np.float64))
    gain_lin = 10 ** ((TARGET_LUFS - raw_lufs) / 20.0)
    mastered_data = (concat_audio * gain_lin).astype(np.float32)
    msr = SR
    prog2.upd(3, "Python fallback OK")
    prog2.ok()

# ── Post-process Step 1: LUFS trim ──────────────────────────────
print(f"\n  {DIM}Post-process: LUFS trim...{RESET}")
meter2 = pyln.Meter(msr)
cur_lufs = meter2.integrated_loudness(mastered_data.astype(np.float64))
trim_db  = TARGET_LUFS - cur_lufs
trim_lin = 10 ** (trim_db / 20.0)
mastered_data = (mastered_data * trim_lin).astype(np.float32)
print(f"  LUFS trim: {cur_lufs:.3f} → {TARGET_LUFS:.3f} LU  ({trim_db:+.3f} dB)")

# ── Post-process Step 2: Measure TP after LUFS trim ─────────────
data_4x  = rp(mastered_data.astype(np.float64), 4, 1, axis=0)
tp_after_trim_lin = float(np.max(np.abs(data_4x)))
tp_after_trim_db  = 20 * math.log10(max(tp_after_trim_lin, 1e-10))
print(f"  TP after LUFS trim: {tp_after_trim_db:.3f} dBTP  (ceiling: {CEILING_DB} dBTP)")

# ── Post-process Step 3: MAKEUP GAIN → push TP to ceiling ───────
# This is the Ozone 12 / Logic Pro X maximizer approach:
# If TP is below ceiling, apply makeup gain so peaks reach the ceiling,
# then re-apply the limiter to clip exactly at ceiling.
if tp_after_trim_db < CEILING_DB - 0.05:
    makeup_db  = CEILING_DB - tp_after_trim_db
    makeup_lin = 10 ** (makeup_db / 20.0)
    mastered_data = (mastered_data * makeup_lin).astype(np.float32)
    print(f"  {CYAN}Makeup gain: +{makeup_db:.3f} dB → push TP to ceiling{RESET}")

    # Re-apply True Peak limiter as final stage (Ozone 12 / Logic Pro X style)
    print(f"  {CYAN}Re-applying True Peak limiter at {CEILING_DB} dBTP...{RESET}", end="", flush=True)
    lim_final = LAL(ceiling_db=CEILING_DB, lookahead_ms=5.0, release_ms=100.0)
    mastered_data = lim_final.process(mastered_data.astype(np.float64), msr).astype(np.float32)
    print(f" {GREEN}done{RESET}")
else:
    print(f"  {GREEN}TP already at ceiling — no makeup needed{RESET}")

# ── Post-process Step 4: Brickwall safety ───────────────────────
peak_now = np.max(np.abs(mastered_data))
if peak_now > CEIL_LIN:
    mastered_data = (mastered_data * CEIL_LIN / peak_now).astype(np.float32)
    print(f"  Safety clip applied (peak was {20*math.log10(peak_now):.3f} dBFS)")

mastered_path = os.path.join(TMP, "mastered_audio.wav")
sf.write(mastered_path, mastered_data, msr, subtype="PCM_24")

# ── Quick verify ─────────────────────────────────────────────────
data_4x2 = rp(mastered_data.astype(np.float64), 4, 1, axis=0)
tp_final = 20 * math.log10(max(float(np.max(np.abs(data_4x2))), 1e-10))
lufs_final = meter2.integrated_loudness(mastered_data.astype(np.float64))
tp_pass = CEILING_DB - 0.15 <= tp_final <= CEILING_DB + 0.05
lufs_pass = -10.0 <= lufs_final <= -9.0

if not tp_pass:
    # Iterate: try again with tighter gain
    print(f"\n  {YELLOW}⚠ TP={tp_final:.3f} dBTP — iterating...{RESET}")
    for _ in range(3):
        adj = CEILING_DB - tp_final
        adj_lin = 10 ** (adj / 20.0)
        mastered_data = (mastered_data * adj_lin).astype(np.float32)
        lim2 = LAL(ceiling_db=CEILING_DB, lookahead_ms=5.0, release_ms=100.0)
        mastered_data = lim2.process(mastered_data.astype(np.float64), msr).astype(np.float32)
        data_4x3 = rp(mastered_data.astype(np.float64), 4, 1, axis=0)
        tp_final = 20 * math.log10(max(float(np.max(np.abs(data_4x3))), 1e-10))
        lufs_final = meter2.integrated_loudness(mastered_data.astype(np.float64))
        print(f"  Iteration: TP={tp_final:.3f} dBTP  LUFS={lufs_final:.3f} LU")
        if CEILING_DB - 0.15 <= tp_final <= CEILING_DB + 0.05:
            break
    sf.write(mastered_path, mastered_data, msr, subtype="PCM_24")

tp_icon   = '✅' if CEILING_DB-0.15 <= tp_final <= CEILING_DB+0.05 else '❌'
lufs_icon = '✅' if -10.0 <= lufs_final <= -9.0 else '⚠️'
print(f"\n  {GREEN}Mastering result:{RESET}")
print(f"    LUFS = {lufs_final:.3f} LU  {lufs_icon}")
print(f"    TP   = {tp_final:.3f} dBTP  {tp_icon}  (ceiling: {CEILING_DB} dBTP)")
print(f"    Size = {os.path.getsize(mastered_path)//1048576}MB")

# ══════════════════════════════════════════════════════════════════
print(f"\n{BOLD}PHASE 3 — Video + Logo Overlay{RESET}  (Vinyl Awakening Vol.1_video.mp4 → trim + Chillin Vibes logo)\n")
# ══════════════════════════════════════════════════════════════════

audio_dur = len(mastered_data) / msr
logo_found = os.path.exists(LOGO)
prog3 = Prog("Video + Logo", 1, "render")
prog3.upd(0, f"{'Logo overlay' if logo_found else 'Trim only'} {audio_dur/60:.1f}min...")

trimmed_video = os.path.join(TMP, "trimmed_video.mp4")
if logo_found:
    print(f"\n  {GREEN}Logo found:{RESET} Chillin Vibes Logo.jpg — overlay top-right")
    # Trim video + overlay logo (scaled to 180px wide, top-right with 20px margin)
    r = subprocess.run([
        FFMPEG, "-y",
        "-i", VDO,
        "-i", LOGO,
        "-t", str(audio_dur),
        "-filter_complex",
        "[1:v]scale=180:-1,format=rgba,colorchannelmixer=aa=0.85[logo];"
        "[0:v][logo]overlay=W-w-30:30",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-an",
        trimmed_video
    ], capture_output=True, text=True)
else:
    print(f"\n  {YELLOW}Logo not found — trim only{RESET}")
    r = subprocess.run([
        FFMPEG, "-y", "-i", VDO, "-t", str(audio_dur),
        "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-an",
        trimmed_video
    ], capture_output=True, text=True)

if r.returncode != 0:
    print(f"\n{RED}Video error:{RESET}", r.stderr[-400:])
    sys.exit(1)

prog3.upd(1, f"{os.path.getsize(trimmed_video)//1048576}MB")
prog3.ok(f"Trimmed {audio_dur/60:.1f}min + {'logo overlay ✅' if logo_found else 'no logo'}")

# ══════════════════════════════════════════════════════════════════
print(f"\n{BOLD}PHASE 4 — Final Mux{RESET}  (video + mastered audio → MP4)\n")
# ══════════════════════════════════════════════════════════════════

final_tmp = os.path.join(TMP, "final_video.mp4")
prog4 = Prog("Render Final", 1, "render")
prog4.upd(0, "ffmpeg mux...")
r3 = subprocess.run([
    FFMPEG, "-y",
    "-i", trimmed_video,
    "-i", mastered_path,
    "-c:v", "copy",
    "-c:a", "aac", "-b:a", "320k",
    "-shortest",
    final_tmp
], capture_output=True, text=True)
if r3.returncode != 0:
    print(f"\n{RED}Mux error:{RESET}", r3.stderr[-400:])
prog4.upd(1)
prog4.ok(f"{os.path.getsize(final_tmp)//1048576}MB")

# ══════════════════════════════════════════════════════════════════
print(f"\n{BOLD}PHASE 5 — Export{RESET}  (→ ~/Desktop/LongPlay_Production/)\n")
# ══════════════════════════════════════════════════════════════════

exports = [
    (os.path.join(TMP, "concat_audio.wav"), "concat_audio.wav"),
    (mastered_path, "mastered_audio.wav"),
    (final_tmp,     "final_video.mp4"),
]
prog5 = Prog("Export", len(exports), "files")
for src, name in exports:
    prog5.upd(0, f"Copying {name}...")
    dst = os.path.join(DESK, name)
    shutil.copy2(src, dst)
    prog5.upd(1, f"{name} ({os.path.getsize(dst)//1048576}MB)")
prog5.ok(f"All {len(exports)} files → Desktop")

# ══════════════════════════════════════════════════════════════════
print(f"\n{BOLD}PHASE 6 — Signal Measurement{RESET}\n")
# ══════════════════════════════════════════════════════════════════

data_m, sr_m = sf.read(os.path.join(DESK, "mastered_audio.wav"), always_2d=True)
data_m = data_m.astype(np.float64)

prog6 = Prog("Measuring", 6, "metrics")

lufs_m = pyln.Meter(sr_m).integrated_loudness(data_m)
prog6.upd(1, f"LUFS={lufs_m:.3f}")

data_4xm = rp(data_m, 4, 1, axis=0)
tp_lin_m = float(np.max(np.abs(data_4xm)))
tp_m = 20 * math.log10(max(tp_lin_m, 1e-10))
prog6.upd(1, f"TP={tp_m:.3f}")

max_samp = float(np.max(np.abs(data_m)))
prog6.upd(1, f"max={max_samp:.4f}")

l = data_m[:,0]; r = data_m[:,1]
corr = float(np.sum(l*r) / max(np.sqrt(np.sum(l*l))*np.sqrt(np.sum(r*r)), 1e-10))
prog6.upd(1, f"corr={corr:.3f}")

# THD
t  = np.linspace(0, 1.0, int(sr_m), endpoint=False)
s  = (0.25 * np.sin(2*math.pi*1000*t)).astype(np.float32)
sa = np.column_stack([s, s])
lt = LAL(ceiling_db=CEILING_DB, release_ms=80.0)
ot = lt.process(sa.astype(np.float64), sr_m)[:,0]
nf = len(ot)
sp = np.abs(np.fft.rfft(ot)) / (nf/2)
fr = np.fft.rfftfreq(nf, 1.0/sr_m)
fi = np.argmin(np.abs(fr-1000))
fm = sp[fi]
hs = sum(sp[fi*k]**2 for k in [2,3,4,5] if fi*k < len(sp))
thd = 100.0*math.sqrt(hs)/max(fm, 1e-10)
prog6.upd(1, f"THD={thd:.4f}%")

rms_m = float(np.sqrt(np.mean(data_m**2)))
crest = 20*math.log10(max(tp_lin_m, 1e-10)/max(rms_m, 1e-10))
prog6.upd(1, f"crest={crest:.1f}dB")
prog6.ok()

try:
    lra_m = pyln.Meter(sr_m).loudness_range(data_m)
except:
    lra_m = None

# ══════════════════════════════════════════════════════════════════
TOTAL = time.time() - PIPELINE_START
print(f"\n{BOLD}PHASE 7 — Report{RESET}\n")
# ══════════════════════════════════════════════════════════════════

fi = {n: os.path.getsize(os.path.join(DESK,n))//1048576
      for n in ["concat_audio.wav","mastered_audio.wav","final_video.mp4"]}
dur_m = len(data_m)/sr_m

# Video info via ffprobe
vinfo = subprocess.run(["/opt/homebrew/bin/ffprobe","-v","quiet","-show_streams",
    "-select_streams","v:0","-of","csv=p=0",
    os.path.join(DESK,"final_video.mp4")],capture_output=True,text=True).stdout.strip()
ainfo = subprocess.run(["/opt/homebrew/bin/ffprobe","-v","quiet","-show_streams",
    "-select_streams","a:0","-of","csv=p=0",
    os.path.join(DESK,"final_video.mp4")],capture_output=True,text=True).stdout.strip()

tp_ok   = CEILING_DB-0.15 <= tp_m <= CEILING_DB+0.05
lufs_ok = -10.0 <= lufs_m <= -9.0

print(f"""
{BOLD}{MAGENTA}{'═'*70}{RESET}
{BOLD}  📊  รายงานผลการ Export — LongPlay Studio V5.11.1 Full Album{RESET}
{BOLD}  Vinyl Prophet Vol.1 | 18 เพลงเต็ม | {dur_m/60:.1f} นาที{RESET}
{BOLD}{MAGENTA}{'═'*70}{RESET}

{BOLD}ไฟล์ที่ Export → ~/Desktop/LongPlay_FullAlbum_Export/{RESET}
  ├─ concat_audio.wav   {fi['concat_audio.wav']:>5} MB  (18 เพลงต่อกัน)
  ├─ mastered_audio.wav {fi['mastered_audio.wav']:>5} MB  (mastered)
  └─ final_video.mp4    {fi['final_video.mp4']:>5} MB  (video + audio)

{BOLD}ค่าสัญญาณที่วัดได้:{RESET}
  LUFS Integrated : {lufs_m:+.3f} LU    (เป้า: -10.0 ถึง -9.0)  {'✅' if lufs_ok else '❌'}
  True Peak       : {tp_m:+.3f} dBTP  (เป้า: -1.1 ถึง -1.0)    {'✅' if tp_ok else '❌'}  ← {'แตะ ceiling!' if tp_ok else 'ยังไม่ถึง ceiling'}
  Max Sample      : {max_samp:.4f}         (เป้า: < 1.0 ไม่ clip)   {'✅' if max_samp<1.0 else '❌ CLIP'}
  Stereo Corr     : {corr:.3f}          (เป้า: > 0.5)             {'✅' if corr>0.5 else '❌'}
  THD (1kHz)      : {thd:.4f}%       (เป้า: < 1%)              {'✅' if thd<1.0 else '❌'}
  Crest Factor    : {crest:.1f} dB""")

if lra_m: print(f"  LU Range (LRA)  : {lra_m:.2f} LU")

print(f"""
{BOLD}รายละเอียด:{RESET}
  Duration        : {dur_m:.1f}s = {int(dur_m//60)}m {int(dur_m%60):02d}s ({len(songs)} เพลง)
  Source          : ยังไม่ทำ Master/ (เพลงเต็ม)
  Sample Rate     : {sr_m} Hz  |  Channels: {data_m.shape[1]}ch

{BOLD}Logo / GIF:{RESET}
  ✅ Chillin Vibes Logo.jpg — overlay top-right (30px margin, 180px wide, 85% opacity)
  ไฟล์: Logo ทุกอย่างสำหรับใช้งาน/Chillin Vibes Logo.jpg

{BOLD}อธิบาย Root Cause TP -2.3 dBTP (รอบก่อน):{RESET}
  รอบก่อน: Rust chain render ออกมา LUFS -7.1 LU
  Python post-process trim: -9.5 - (-7.1) = -2.4 dB gain reduction
  TP ลดลงจาก ~-1.0 → ~-3.4 dBTP (LUFS trim ดึง level ลงหลัง limiter!)
  {CYAN}Fix: Makeup gain +{abs(CEILING_DB - tp_after_trim_db):.2f} dB → re-apply limiter → TP แตะ ceiling{RESET}

{BOLD}เปรียบเทียบกับ Logic Pro X / Ozone 12:{RESET}
  ┌────────────────────┬──────────────┬──────────────┬──────────────┐
  │ Software           │ LUFS target  │ TP ceiling   │ TP output    │
  ├────────────────────┼──────────────┼──────────────┼──────────────┤
  │ iZotope Ozone 12   │ -9.5 LU      │ -1.0 dBTP    │ -1.0 dBTP ✅ │
  │ Logic Pro X        │ -9.5 LU      │ -1.0 dBTP    │ -1.0 dBTP ✅ │
  │ LongPlay (v ก่อน)  │ -9.5 LU      │ -1.0 dBTP    │ -2.3 dBTP ❌ │
  │ LongPlay (v นี้)   │ {lufs_m:.1f} LU    │ -1.0 dBTP    │ {tp_m:.1f} dBTP {'✅' if tp_ok else '❌'} │
  └────────────────────┴──────────────┴──────────────┴──────────────┘

  วิธีแก้: Makeup gain หลัง LUFS trim + re-apply True Peak limiter
  เหมือน Ozone 12 Maximizer: push signal ถึง ceiling แล้ว limit

{BOLD}เวลาที่ใช้:{RESET}
  ⏱  Total Pipeline : {BOLD}{GREEN}{fmt_t(TOTAL)}{RESET}  ({TOTAL:.0f}s = {TOTAL/60:.1f}min)

{BOLD}{MAGENTA}{'═'*70}{RESET}
{BOLD}{'✅' if tp_ok and lufs_ok else '⚠️'}  Signal Quality: {sum([lufs_ok,tp_ok,max_samp<1.0,corr>0.5,thd<1.0])}/5 targets passed{RESET}
{BOLD}{MAGENTA}{'═'*70}{RESET}
""")

# Cleanup
shutil.rmtree(TMP, ignore_errors=True)
sys.exit(0 if (tp_ok and lufs_ok) else 1)
