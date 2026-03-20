# Project Brief — MRG LongPlay Studio V5.5

## Executive Summary

MRG LongPlay Studio เป็นโปรแกรม Desktop สำหรับ Music Production ที่รวม 3 โลกเข้าด้วยกัน:
1. **CapCut-class Video Editor** — Timeline, multi-track, transitions, effects
2. **Logic Pro X + iZotope Ozone 12 Mastering** — Full DSP chain ระดับ professional
3. **Waves WLM Plus Metering** — Clone สี ฟังก์ชัน และ layout ทั้งหมด

## Vision Statement

> สร้างโปรแกรมตัดต่อเพลง+วิดีโอระดับ Professional ที่ใช้ algorithm และ language เดียวกับ CapCut (Rust/C++ core + GUI layer) ผสม mastering engine ระดับ Logic Pro X + Ozone 12 พร้อม UI แบบ vintage hardware (Waves/SSL/Neve)

## Target Users

- **Primary:** Music Producers ที่ต้องการ all-in-one tool สำหรับ batch production → mastering → video → upload
- **Secondary:** AI Music Creators (Suno AI, Udio) ที่ต้องการ professional mastering + YouTube publishing pipeline
- **Tertiary:** Content Creators ที่ต้องการ CapCut-level editing + pro audio mastering ในโปรแกรมเดียว

## Technology Philosophy

| Principle | Implementation |
|-----------|---------------|
| **CapCut Architecture** | Rust native engine (DSP, I/O, Analysis) + Python GUI layer — เหมือน CapCut ที่ใช้ C++ engine + Flutter UI |
| **Audio-first Language** | Rust สำหรับ DSP — zero-cost abstractions, no GC pauses, SIMD-ready — เหมือนที่ pro audio ใช้ C++/Rust |
| **Ozone-quality DSP** | Real audio processing (not FFmpeg filter chains) — pedalboard + scipy fallback, Rust native เป็น primary |
| **Hardware UI** | Vintage analog aesthetic — gunmetal chassis, amber VU glow, brushed steel knobs, LED segments |

## Current State (Brownfield)

**Repository:** `github.com/mrarrangerteam/MRG-LONGPLAY-STUDIO-COMPLETE-2`

| Component | Status | LOC |
|-----------|--------|-----|
| Python GUI + Video Editor | Working (basic) | 12,978 |
| Python Mastering Engine | Working (bugs fixed) | 14,044 |
| Rust Native Engine | Code complete, NOT compiled | 13,191 |
| **Total** | | **40,213** |

### BMAD Audit Results (March 2026)

**Fixed in v5.5-complete-rewrite branch:**
- P0: ai_assist.py Maximizer gain push bug (was always 0 dB)
- P0: IRC mode legacy naming mismatch
- P1: PyQt6/PySide6 fallback, requirements.txt, bare except blocks, crash log leak, Rust sample rate

**Still Need Work:**
- Video Editor ขาด features มาก เทียบกับ CapCut (Multi-track, Keyframes, Effects, Undo/Redo, Text overlay, GPU preview)
- Rust backend ยังไม่ได้ compile เป็น .so/.dylib
- gui.py 12,978 บรรทัดในไฟล์เดียว ต้อง refactor
- ไม่มี unit tests
- Metering ยังไม่เป็น exact WLM Plus clone

## Key Constraints

- **Platform:** macOS first (Apple Silicon), Windows second
- **Build Tool:** Rust → maturin (PyO3), Python → pip
- **GUI:** PyQt6 (current), future migration to Tauri 2.0 + React possible
- **Audio I/O:** FFmpeg (required), soundfile (Python), hound/symphonia (Rust)
- **No Cloud Dependency:** ทุกอย่างทำงาน offline, ไม่มี API calls สำหรับ DSP

## Success Criteria

1. Mastering engine ให้ผลลัพธ์ True Peak ≤ -1.0 dBTP 100% ของเวลา
2. Video export เร็วกว่า real-time (ใช้ HW acceleration)
3. Rust backend compile + ทำงานได้ — benchmark เร็วกว่า Python ≥ 10x
4. GUI ไม่ crash เมื่อใช้งานทุก feature path
5. WLM Plus meter แสดง LUFS/TP/LRA ถูกต้องเทียบกับ reference tool ±0.5 LU

## Stakeholder

- **Product Owner:** mrarrangerteam (MRARRANGER AI Studio)
- **Development:** Claude Code (AI-assisted, story-driven)
- **QA:** BMAD QA Agent (post-story validation)
