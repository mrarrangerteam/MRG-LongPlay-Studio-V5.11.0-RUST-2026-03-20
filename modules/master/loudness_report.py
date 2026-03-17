"""
Loudness report export — CSV and PDF formats.

Story 4.6 — Epic 4: Pro Mastering.

Features:
    - Export loudness analysis to CSV file
    - Export loudness analysis to PDF file (text-based, no external lib required)
    - Platform compliance checklist
    - Timestamp and file metadata
    - Summary statistics (LUFS, True Peak, LRA)
"""

from __future__ import annotations

import csv
import datetime
import os
from typing import Dict, List, Optional, Tuple

from .loudness import LoudnessAnalysis
from .genre_profiles import PLATFORM_TARGETS


# ---------------------------------------------------------------------------
# Report data container
# ---------------------------------------------------------------------------
class LoudnessReportData:
    """Container for loudness report data."""

    def __init__(self) -> None:
        self.file_name: str = ""
        self.file_path: str = ""
        self.duration_sec: float = 0.0
        self.sample_rate: int = 44100
        self.channels: int = 2
        self.integrated_lufs: float = -70.0
        self.true_peak_dbtp: float = -70.0
        self.lra: float = 0.0
        self.momentary_max: float = -70.0
        self.short_term_max: float = -70.0
        self.target_platform: str = "YouTube"
        self.target_lufs: float = -14.0
        self.target_tp: float = -1.0
        self.timestamp: str = ""
        self.history: List[Dict[str, float]] = []   # time-series data

    @classmethod
    def from_analysis(cls, analysis: LoudnessAnalysis,
                      file_path: str = "",
                      platform: str = "YouTube") -> "LoudnessReportData":
        """Create report data from a LoudnessAnalysis object."""
        report = cls()
        report.file_path = file_path
        report.file_name = os.path.basename(file_path) if file_path else "Unknown"
        report.duration_sec = analysis.duration_sec
        report.sample_rate = analysis.sample_rate
        report.channels = analysis.channels
        report.integrated_lufs = analysis.integrated_lufs
        report.true_peak_dbtp = analysis.true_peak_dbtp
        report.lra = analysis.lra
        report.target_platform = platform
        target = PLATFORM_TARGETS.get(platform, PLATFORM_TARGETS.get("YouTube", {}))
        report.target_lufs = target.get("target_lufs", -14.0)
        report.target_tp = target.get("true_peak", -1.0)
        report.timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return report


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------
def export_csv(report: LoudnessReportData, output_path: str) -> bool:
    """
    Export loudness report as CSV file.

    Args:
        report: Report data to export.
        output_path: Path for the output CSV file.

    Returns:
        True on success.
    """
    try:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # header section
            writer.writerow(["MRG LongPlay Studio V5.5 — Loudness Report"])
            writer.writerow(["Generated", report.timestamp])
            writer.writerow([])

            # file info
            writer.writerow(["File Information"])
            writer.writerow(["File Name", report.file_name])
            writer.writerow(["File Path", report.file_path])
            writer.writerow(["Duration (s)", f"{report.duration_sec:.2f}"])
            writer.writerow(["Sample Rate", report.sample_rate])
            writer.writerow(["Channels", report.channels])
            writer.writerow([])

            # loudness measurements
            writer.writerow(["Loudness Measurements"])
            writer.writerow(["Integrated LUFS", f"{report.integrated_lufs:.1f}"])
            writer.writerow(["True Peak (dBTP)", f"{report.true_peak_dbtp:.1f}"])
            writer.writerow(["Loudness Range (LU)", f"{report.lra:.1f}"])
            writer.writerow(["Momentary Max (LUFS)", f"{report.momentary_max:.1f}"])
            writer.writerow(["Short-term Max (LUFS)", f"{report.short_term_max:.1f}"])
            writer.writerow([])

            # target compliance
            writer.writerow(["Target Compliance"])
            writer.writerow(["Target Platform", report.target_platform])
            writer.writerow(["Target LUFS", f"{report.target_lufs:.1f}"])
            writer.writerow(["Target True Peak", f"{report.target_tp:.1f}"])
            lufs_delta = report.integrated_lufs - report.target_lufs
            tp_ok = report.true_peak_dbtp <= report.target_tp + 0.1
            lufs_ok = abs(lufs_delta) <= 1.0
            writer.writerow(["LUFS Delta", f"{lufs_delta:+.1f} LU"])
            writer.writerow(["LUFS Compliance", "PASS" if lufs_ok else "FAIL"])
            writer.writerow(["True Peak Compliance", "PASS" if tp_ok else "FAIL"])
            writer.writerow(["Overall", "PASS" if (lufs_ok and tp_ok) else "FAIL"])
            writer.writerow([])

            # platform checklist
            writer.writerow(["Platform Compliance Checklist"])
            writer.writerow(["Platform", "Target LUFS", "Target TP", "LUFS OK", "TP OK", "Overall"])
            for name, target in PLATFORM_TARGETS.items():
                t_lufs = target.get("target_lufs", -14.0)
                t_tp = target.get("true_peak", -1.0)
                l_ok = abs(report.integrated_lufs - t_lufs) <= 1.0
                t_ok = report.true_peak_dbtp <= t_tp + 0.1
                writer.writerow([
                    name,
                    f"{t_lufs:.1f}",
                    f"{t_tp:.1f}",
                    "PASS" if l_ok else "FAIL",
                    "PASS" if t_ok else "FAIL",
                    "PASS" if (l_ok and t_ok) else "FAIL",
                ])
            writer.writerow([])

            # time-series data (if available)
            if report.history:
                writer.writerow(["Time-Series Data"])
                writer.writerow(["Time (s)", "Momentary LUFS", "Short-term LUFS",
                                 "True Peak L", "True Peak R"])
                for entry in report.history:
                    writer.writerow([
                        f"{entry.get('time', 0.0):.2f}",
                        f"{entry.get('momentary', -70.0):.1f}",
                        f"{entry.get('short_term', -70.0):.1f}",
                        f"{entry.get('tp_l', -70.0):.1f}",
                        f"{entry.get('tp_r', -70.0):.1f}",
                    ])

        return True
    except OSError as e:
        print(f"[REPORT] CSV export error: {e}")
        return False


# ---------------------------------------------------------------------------
# PDF export (simple text-based PDF — no external dependencies)
# ---------------------------------------------------------------------------
def export_pdf(report: LoudnessReportData, output_path: str) -> bool:
    """
    Export loudness report as a simple PDF file.

    Uses a minimal PDF generator (no external libraries required).
    For production use, consider reportlab or weasyprint.

    Args:
        report: Report data to export.
        output_path: Path for the output PDF file.

    Returns:
        True on success.
    """
    try:
        lines = _build_report_text(report)
        _write_simple_pdf(lines, output_path)
        return True
    except OSError as e:
        print(f"[REPORT] PDF export error: {e}")
        return False


def _build_report_text(report: LoudnessReportData) -> List[str]:
    """Build the report as a list of text lines."""
    lines: List[str] = []

    lines.append("MRG LongPlay Studio V5.5")
    lines.append("LOUDNESS REPORT")
    lines.append(f"Generated: {report.timestamp}")
    lines.append("")
    lines.append("=" * 50)
    lines.append("FILE INFORMATION")
    lines.append("=" * 50)
    lines.append(f"File:         {report.file_name}")
    lines.append(f"Duration:     {report.duration_sec:.2f} seconds")
    lines.append(f"Sample Rate:  {report.sample_rate} Hz")
    lines.append(f"Channels:     {report.channels}")
    lines.append("")
    lines.append("=" * 50)
    lines.append("LOUDNESS MEASUREMENTS")
    lines.append("=" * 50)
    lines.append(f"Integrated:   {report.integrated_lufs:.1f} LUFS")
    lines.append(f"True Peak:    {report.true_peak_dbtp:.1f} dBTP")
    lines.append(f"LRA:          {report.lra:.1f} LU")
    lines.append(f"Momentary Max:{report.momentary_max:.1f} LUFS")
    lines.append(f"Short-term Max:{report.short_term_max:.1f} LUFS")
    lines.append("")
    lines.append("=" * 50)
    lines.append(f"TARGET: {report.target_platform}")
    lines.append("=" * 50)
    lufs_delta = report.integrated_lufs - report.target_lufs
    lufs_ok = abs(lufs_delta) <= 1.0
    tp_ok = report.true_peak_dbtp <= report.target_tp + 0.1
    lines.append(f"Target LUFS:  {report.target_lufs:.1f}")
    lines.append(f"Target TP:    {report.target_tp:.1f} dBTP")
    lines.append(f"LUFS Delta:   {lufs_delta:+.1f} LU  {'PASS' if lufs_ok else 'FAIL'}")
    lines.append(f"TP Check:     {'PASS' if tp_ok else 'FAIL'}")
    lines.append(f"Overall:      {'PASS' if (lufs_ok and tp_ok) else 'FAIL'}")
    lines.append("")
    lines.append("=" * 50)
    lines.append("PLATFORM COMPLIANCE")
    lines.append("=" * 50)

    for name, target in PLATFORM_TARGETS.items():
        t_lufs = target.get("target_lufs", -14.0)
        t_tp = target.get("true_peak", -1.0)
        l_ok = abs(report.integrated_lufs - t_lufs) <= 1.0
        t_ok = report.true_peak_dbtp <= t_tp + 0.1
        status = "PASS" if (l_ok and t_ok) else "FAIL"
        lines.append(f"  {name:<15s} {t_lufs:>6.1f} LUFS  {t_tp:>5.1f} dBTP  [{status}]")

    return lines


def _write_simple_pdf(lines: List[str], output_path: str) -> None:
    """
    Write a minimal valid PDF with the given text lines.

    This produces a simple single-page PDF using raw PDF operators.
    No external libraries required.
    """
    font_size = 10
    line_height = 14
    margin_x = 50
    margin_y = 750
    page_width = 612   # US Letter
    page_height = 792

    # build content stream
    content_lines: List[str] = []
    content_lines.append("BT")
    content_lines.append(f"/F1 {font_size} Tf")

    y = margin_y
    for line in lines:
        if y < 50:
            break  # simple single-page limit
        safe_line = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        content_lines.append(f"{margin_x} {y} Td")
        content_lines.append(f"({safe_line}) Tj")
        content_lines.append(f"-{margin_x} -{line_height} Td")
        y -= line_height

    content_lines.append("ET")
    content_stream = "\n".join(content_lines)

    # assemble PDF objects
    objects: List[str] = []

    # obj 1: catalog
    objects.append("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj")
    # obj 2: pages
    objects.append("2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj")
    # obj 3: page
    objects.append(
        f"3 0 obj\n<< /Type /Page /Parent 2 0 R "
        f"/MediaBox [0 0 {page_width} {page_height}] "
        f"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj"
    )
    # obj 4: content stream
    objects.append(
        f"4 0 obj\n<< /Length {len(content_stream)} >>\nstream\n"
        f"{content_stream}\nendstream\nendobj"
    )
    # obj 5: font
    objects.append(
        "5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>\nendobj"
    )

    # write PDF
    with open(output_path, "wb") as f:
        f.write(b"%PDF-1.4\n")

        offsets: List[int] = []
        for obj in objects:
            offsets.append(f.tell())
            f.write(obj.encode("latin-1") + b"\n")

        xref_pos = f.tell()
        f.write(b"xref\n")
        f.write(f"0 {len(objects) + 1}\n".encode())
        f.write(b"0000000000 65535 f \n")
        for offset in offsets:
            f.write(f"{offset:010d} 00000 n \n".encode())

        f.write(b"trailer\n")
        f.write(f"<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode())
        f.write(b"startxref\n")
        f.write(f"{xref_pos}\n".encode())
        f.write(b"%%EOF\n")
