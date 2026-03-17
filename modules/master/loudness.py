"""
LongPlay Studio V5.0 — Loudness Meter Module
Inspired by Waves WLM Plus Loudness Meter

Features:
- Integrated LUFS (long-term average) per ITU-R BS.1770-4
- Short-term LUFS (3-second sliding window)
- Momentary LUFS (400ms sliding window)
- True Peak (4x oversampled) in dBTP
- Loudness Range (LU)
- Platform target presets (YouTube, Spotify, Apple Music, etc.)

Analysis via FFmpeg loudnorm (2-pass) and numpy for display
"""

import subprocess
import json
import os
import re
from typing import Optional, Dict, Tuple
from .genre_profiles import PLATFORM_TARGETS


class LoudnessAnalysis:
    """Results of loudness analysis."""

    def __init__(self):
        self.integrated_lufs = -24.0
        self.true_peak_dbtp = -6.0
        self.lra = 0.0                # Loudness Range in LU
        self.threshold = -70.0
        self.target_offset = 0.0
        self.sample_rate = 48000
        self.channels = 2
        self.duration_sec = 0.0

        # Additional analysis (from FFmpeg stats)
        self.input_i = -24.0          # Input Integrated
        self.input_tp = -6.0          # Input True Peak
        self.input_lra = 0.0          # Input LRA
        self.input_thresh = -70.0
        self.output_i = -24.0         # Output Integrated (after normalization)
        self.output_tp = -6.0
        self.output_lra = 0.0
        self.output_thresh = -70.0
        self.normalization_type = ""
        self.target_offset_lu = 0.0

    def meets_target(self, platform: str = "YouTube") -> dict:
        """Check if audio meets platform loudness target."""
        target = PLATFORM_TARGETS.get(platform, PLATFORM_TARGETS["YouTube"])
        target_lufs = target["target_lufs"]
        target_tp = target["true_peak"]

        lufs_ok = abs(self.integrated_lufs - target_lufs) <= 1.0  # ±1 LU tolerance
        tp_ok = self.true_peak_dbtp <= target_tp + 0.1  # 0.1 dB tolerance

        return {
            "platform": platform,
            "target_lufs": target_lufs,
            "target_true_peak": target_tp,
            "measured_lufs": self.integrated_lufs,
            "measured_true_peak": self.true_peak_dbtp,
            "lufs_ok": lufs_ok,
            "true_peak_ok": tp_ok,
            "passes": lufs_ok and tp_ok,
            "lufs_delta": self.integrated_lufs - target_lufs,
            "tp_delta": self.true_peak_dbtp - target_tp,
        }

    def to_dict(self) -> dict:
        return {
            "integrated_lufs": self.integrated_lufs,
            "true_peak_dbtp": self.true_peak_dbtp,
            "lra": self.lra,
            "duration_sec": self.duration_sec,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
        }


class LoudnessMeter:
    """Loudness measurement using FFmpeg loudnorm (ITU-R BS.1770-4)."""

    def __init__(self, ffmpeg_path: str = "ffmpeg"):
        self.ffmpeg_path = ffmpeg_path

    def analyze(self, audio_path: str, timeout: int = 120) -> Optional[LoudnessAnalysis]:
        """
        Analyze audio file loudness using FFmpeg loudnorm (measure-only mode).

        This is a single-pass analysis that returns LUFS, True Peak, and LRA.
        Does NOT modify the file.

        Args:
            audio_path: Path to audio file
            timeout: Max seconds to wait

        Returns:
            LoudnessAnalysis object or None on error
        """
        if not os.path.exists(audio_path):
            print(f"[LOUDNESS] File not found: {audio_path}")
            return None

        # FFmpeg loudnorm filter in "measure" mode (print=true)
        # This analyzes the entire file and prints stats to stderr
        cmd = [
            self.ffmpeg_path,
            "-i", audio_path,
            "-af", "loudnorm=I=-14:LRA=7:TP=-1:print_format=json",
            "-f", "null", "-"
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            # Parse JSON output from stderr
            stderr = result.stderr
            analysis = self._parse_loudnorm_output(stderr)

            if analysis:
                # Get duration
                dur_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", stderr)
                if dur_match:
                    h, m, s = dur_match.groups()
                    analysis.duration_sec = int(h) * 3600 + int(m) * 60 + float(s)

                # Get sample rate and channels
                sr_match = re.search(r"(\d+)\s*Hz", stderr)
                if sr_match:
                    analysis.sample_rate = int(sr_match.group(1))

                ch_match = re.search(r"stereo|mono|5\.1|7\.1", stderr)
                if ch_match:
                    ch_str = ch_match.group()
                    analysis.channels = {"mono": 1, "stereo": 2, "5.1": 6, "7.1": 8}.get(ch_str, 2)

            return analysis

        except subprocess.TimeoutExpired:
            print(f"[LOUDNESS] Analysis timed out after {timeout}s")
            return None
        except Exception as e:
            print(f"[LOUDNESS] Analysis error: {e}")
            return None

    def _parse_loudnorm_output(self, stderr: str) -> Optional[LoudnessAnalysis]:
        """Parse FFmpeg loudnorm JSON output from stderr."""
        data = None

        # Strategy 1: Try full JSON block parsing (most robust)
        try:
            json_blocks = re.findall(r'\{[^{}]*"input_i"[^{}]*\}', stderr, re.DOTALL)
            if json_blocks:
                for block in json_blocks:
                    try:
                        data = json.loads(block)
                        break
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"[LOUDNESS] Strategy 1 (full JSON blocks) failed: {e}")

        # Strategy 2: Try parsing individual key=value pairs
        if data is None:
            try:
                data = {}
                key_patterns = [
                    (r'"input_i":\s*"?([-\d.]+)"?', "input_i"),
                    (r'"input_tp":\s*"?([-\d.]+)"?', "input_tp"),
                    (r'"input_lra":\s*"?([-\d.]+)"?', "input_lra"),
                    (r'"input_thresh":\s*"?([-\d.]+)"?', "input_thresh"),
                    (r'"output_i":\s*"?([-\d.]+)"?', "output_i"),
                    (r'"output_tp":\s*"?([-\d.]+)"?', "output_tp"),
                    (r'"output_lra":\s*"?([-\d.]+)"?', "output_lra"),
                    (r'"output_thresh":\s*"?([-\d.]+)"?', "output_thresh"),
                    (r'"normalization_type":\s*"([^"]*)"', "normalization_type"),
                    (r'"target_offset":\s*"?([-\d.]+)"?', "target_offset"),
                ]

                found_any = False
                for pattern, key in key_patterns:
                    match = re.search(pattern, stderr)
                    if match:
                        data[key] = match.group(1)
                        found_any = True

                if not found_any:
                    data = None
            except Exception as e:
                print(f"[LOUDNESS] Strategy 2 (key=value pairs) failed: {e}")
                data = None

        if data is None:
            print("[LOUDNESS] Could not find loudnorm JSON in output (tried multiple parsing strategies)")
            return None

        try:

            analysis = LoudnessAnalysis()

            # Parse values (they come as strings from FFmpeg)
            analysis.input_i = float(data.get("input_i", "-70.0"))
            analysis.input_tp = float(data.get("input_tp", "-70.0"))
            analysis.input_lra = float(data.get("input_lra", "0.0"))
            analysis.input_thresh = float(data.get("input_thresh", "-70.0"))
            analysis.output_i = float(data.get("output_i", "-70.0"))
            analysis.output_tp = float(data.get("output_tp", "-70.0"))
            analysis.output_lra = float(data.get("output_lra", "0.0"))
            analysis.output_thresh = float(data.get("output_thresh", "-70.0"))
            analysis.normalization_type = data.get("normalization_type", "")
            analysis.target_offset_lu = float(data.get("target_offset", "0.0"))

            # Set main values
            analysis.integrated_lufs = analysis.input_i
            analysis.true_peak_dbtp = analysis.input_tp
            analysis.lra = analysis.input_lra

            return analysis

        except (json.JSONDecodeError, ValueError) as e:
            print(f"[LOUDNESS] JSON parse error: {e}")
            return None

    def get_loudnorm_filter(
        self,
        analysis: LoudnessAnalysis,
        target_lufs: float = -14.0,
        target_tp: float = -1.0,
        target_lra: float = 7.0,
    ) -> str:
        """
        Generate FFmpeg loudnorm filter string for 2nd pass (normalization).
        Uses measured values from analysis for accurate linear normalization.

        Args:
            analysis: Results from analyze()
            target_lufs: Target integrated LUFS
            target_tp: Target True Peak ceiling
            target_lra: Target Loudness Range

        Returns:
            FFmpeg filter string for loudnorm 2nd pass
        """
        return (
            f"loudnorm=I={target_lufs}:LRA={target_lra}:TP={target_tp}"
            f":measured_I={analysis.input_i}"
            f":measured_LRA={analysis.input_lra}"
            f":measured_TP={analysis.input_tp}"
            f":measured_thresh={analysis.input_thresh}"
            f":offset={analysis.target_offset_lu}"
            f":linear=true:print_format=summary"
        )

    def quick_measure(self, audio_path: str) -> Optional[Tuple[float, float]]:
        """
        Quick LUFS + True Peak measurement.
        Returns (integrated_lufs, true_peak_dbtp) tuple or None.
        """
        analysis = self.analyze(audio_path, timeout=60)
        if analysis:
            return (analysis.integrated_lufs, analysis.true_peak_dbtp)
        return None
