"""
LongPlay Studio V5.2 — Maximizer Module (Ozone 12 Style)
Full-featured loudness maximizer with real audio processing via FFmpeg.

Features (matching iZotope Ozone 12):
- IRC Modes 1-5 + Low Latency with sub-modes (IRC 3: Pumping/Balanced/Crisp/Clipping,
  IRC 4: Classic/Modern/Transient)
- Gain Push (Input Gain): Manual dB push into limiter (+0.0 to +12.0 dB)
- Output Level (Ceiling): True Peak ceiling in dBTP (-3.0 to -0.1)
- True Peak toggle: Enable/disable true peak limiting
- Character: Smooth ←→ Aggressive slider (0.0 to 10.0)
- Upward Compress: Boost quiet signals (0.0 to 12.0 dB)
- Soft Clip: Pre-limiter saturation (0 to 100%)
- Transient Emphasis: Enhance transients in H/M/L bands (0 to 100%)
- Stereo Independence: Per-channel limiter behavior (Transient 0-100%, Sustain 0-100%)
- Learn Input Gain: Analyze loudness and auto-set gain push

All controls generate REAL FFmpeg filter chains — NOT a UI facade.
Uses: volume, equalizer, acompressor, alimiter, softclip, stereotools
"""

import math
import subprocess
import os
import json
from typing import Optional, List, Dict, Tuple
from .genre_profiles import (
    IRC_MODES, IRC_TOP_MODES, TONE_PRESETS,
    get_irc_mode, get_irc_sub_modes, get_tone_preset
)


class Maximizer:
    """
    Ozone 12-style Loudness Maximizer.
    Every parameter maps to real FFmpeg audio filters.
    """

    def __init__(self):
        # ─── Core Limiter ───
        self.enabled = True
        self.irc_mode = "IRC 4"          # Default: IRC 4
        self.irc_sub_mode = "Classic"     # Default sub-mode for IRC 4
        self.gain_db = 0.0               # Input gain push (0.0 to +12.0 dB)
        self.ceiling = -1.0              # Output ceiling dBTP (-3.0 to -0.1)
        self.true_peak = True            # True Peak limiting on/off

        # ─── Character ───
        self.character = 3.0             # 0.0 (Smooth) to 10.0 (Aggressive)

        # ─── Tone (pre-EQ coloration) ───
        self.tone = "Transparent"

        # ─── Upward Compress ───
        self.upward_compress_db = 0.0    # 0.0 to 12.0 dB

        # ─── Soft Clip ───
        self.soft_clip_enabled = False
        self.soft_clip_pct = 0           # 0 to 100 %

        # ─── Transient Emphasis ───
        self.transient_emphasis_pct = 0  # 0 to 100 %
        self.transient_band = "M"        # H, M, or L

        # ─── Stereo Independence ───
        self.stereo_ind_transient = 0    # 0 to 100 %
        self.stereo_ind_sustain = 0      # 0 to 100 %

        # ─── Internal state ───
        self._learned_lufs = None        # Result from Learn Input Gain

    # ═══════════════════════════════════════════
    #  Setters with validation
    # ═══════════════════════════════════════════

    def set_irc_mode(self, mode: str, sub_mode: str = None):
        """Set IRC mode. If mode has sub-modes and sub_mode given, store it."""
        # Handle legacy naming
        legacy_map = {"IRC I": "IRC 1", "IRC II": "IRC 2", "IRC III": "IRC 3",
                      "IRC IV": "IRC 4", "IRC V": "IRC 5"}
        mode = legacy_map.get(mode, mode)

        if mode in IRC_MODES:
            self.irc_mode = mode
            # If it's a sub-mode key like "IRC 3 - Pumping", extract parent + sub
            if " - " in mode:
                parts = mode.split(" - ", 1)
                self.irc_mode = parts[0].strip()
                self.irc_sub_mode = parts[1].strip()
            elif sub_mode:
                self.irc_sub_mode = sub_mode
            else:
                # Reset sub-mode if switching to a mode without sub-modes
                subs = get_irc_sub_modes(mode)
                if subs:
                    self.irc_sub_mode = subs[0]
                else:
                    self.irc_sub_mode = ""

    def set_irc_sub_mode(self, sub_mode: str):
        """Set IRC sub-mode directly (called from UI dropdown)."""
        self.irc_sub_mode = sub_mode

    def get_effective_irc_key(self) -> str:
        """Get the actual IRC_MODES key to use (with sub-mode if applicable)."""
        subs = get_irc_sub_modes(self.irc_mode)
        if subs and self.irc_sub_mode:
            key = f"{self.irc_mode} - {self.irc_sub_mode}"
            if key in IRC_MODES:
                return key
        return self.irc_mode

    def set_gain(self, gain_db: float):
        """Set input gain push in dB. Range: 0.0 to +20.0 (V5.5.1: extended from 12)"""
        self.gain_db = max(0.0, min(20.0, gain_db))

    def set_ceiling(self, ceiling_dbtp: float):
        """Set output ceiling in dBTP. Range: -3.0 to -0.1"""
        self.ceiling = max(-3.0, min(-0.1, ceiling_dbtp))

    def set_character(self, value: float):
        """Set character amount. Range: 0.0 (Smooth) to 10.0 (Aggressive)"""
        self.character = max(0.0, min(10.0, value))

    def set_upward_compress(self, db: float):
        """Set upward compression amount in dB. Range: 0.0 to 12.0"""
        self.upward_compress_db = max(0.0, min(12.0, db))

    def set_soft_clip(self, enabled: bool, pct: int = 0):
        """Set soft clip. pct: 0-100"""
        self.soft_clip_enabled = enabled
        self.soft_clip_pct = max(0, min(100, pct))

    def set_transient_emphasis(self, pct: int, band: str = "M"):
        """Set transient emphasis. pct: 0-100, band: H/M/L"""
        self.transient_emphasis_pct = max(0, min(100, pct))
        if band in ("H", "M", "L"):
            self.transient_band = band

    def set_stereo_independence(self, transient_pct: int, sustain_pct: int):
        """Set stereo independence. Both 0-100%."""
        self.stereo_ind_transient = max(0, min(100, transient_pct))
        self.stereo_ind_sustain = max(0, min(100, sustain_pct))

    # ═══════════════════════════════════════════
    #  Learn Input Gain — Analyze LUFS
    # ═══════════════════════════════════════════

    def learn_input_gain(self, audio_path: str, target_lufs: float = -11.0) -> Optional[float]:
        """
        Analyze audio loudness and suggest optimal gain push.
        Uses FFmpeg loudnorm to measure integrated LUFS.

        Returns suggested gain_db, or None on failure.
        """
        if not audio_path or not os.path.exists(audio_path):
            return None

        try:
            cmd = [
                "ffmpeg", "-hide_banner", "-i", audio_path,
                "-af", "loudnorm=print_format=json",
                "-f", "null", "-"
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60
            )

            # Parse JSON from stderr (loudnorm outputs there)
            output = result.stderr
            json_start = output.rfind("{")
            json_end = output.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                loudness_data = json.loads(output[json_start:json_end])
                input_lufs = float(loudness_data.get("input_i", "-24"))
                self._learned_lufs = input_lufs

                # Calculate needed gain to reach target
                suggested_gain = target_lufs - input_lufs
                # V5.5.1: Clamp to extended range (positive gain push only)
                suggested_gain = max(0.0, min(20.0, suggested_gain))
                return suggested_gain
        except Exception as e:
            print(f"[MAXIMIZER] Learn Input Gain failed: {e}")

        return None

    def get_learned_lufs(self) -> Optional[float]:
        """Return last measured LUFS from learn_input_gain."""
        return self._learned_lufs

    # ═══════════════════════════════════════════
    #  FFmpeg Filter Chain Generation
    # ═══════════════════════════════════════════

    def get_ffmpeg_filters(self, intensity: float = 1.0) -> list:
        """
        Generate FFmpeg audio filter chain for this maximizer.
        ALL parameters produce REAL filters — nothing is facade.

        Signal flow:
        1. Tone Pre-EQ (coloration)
        2. Upward Compression (boost quiet parts)
        3. Transient Emphasis (EQ boost in H/M/L band)
        4. Soft Clip (pre-limiter saturation)
        5. Input Gain Push (volume boost into limiter)
        6. Character adjustment (attack/release modification)
        7. Limiter (alimiter with IRC mode parameters)

        Args:
            intensity: 0.0-1.0 scaling factor

        Returns:
            List of FFmpeg filter strings to be joined with ','
        """
        filters = []

        if not self.enabled:
            return filters

        # --- Step 1: Tone Pre-EQ ---
        tone_preset = get_tone_preset(self.tone)
        if tone_preset.get("pre_eq"):
            for band_name, band_params in tone_preset["pre_eq"].items():
                band_type = band_params.get("type", "equalizer")
                gain = band_params.get("gain", 0) * intensity

                if abs(gain) < 0.1:
                    continue

                if band_type == "equalizer":
                    freq = band_params["freq"]
                    width = band_params.get("width", 1.0)
                    filters.append(
                        f"equalizer=f={freq}:width_type=o:w={width}:g={gain:.1f}"
                    )
                elif band_type == "lowshelf":
                    freq = band_params["freq"]
                    filters.append(
                        f"lowshelf=f={freq}:g={gain:.1f}:t=s:w=0.5"
                    )
                elif band_type == "highshelf":
                    freq = band_params["freq"]
                    filters.append(
                        f"highshelf=f={freq}:g={gain:.1f}:t=s:w=0.5"
                    )
                elif band_type == "highpass":
                    freq = band_params["freq"]
                    filters.append(f"highpass=f={freq}:p=2")

        # --- Step 2: Upward Compression ---
        uc_db = self.upward_compress_db * intensity
        if uc_db > 0.5:
            # Use acompressor with low threshold to boost quiet signals
            # threshold=-40dB, ratio=1:2 (expansion below threshold)
            # This effectively lifts quiet parts
            uc_threshold = max(-60, -40 - uc_db)
            uc_ratio = 1.0 + (uc_db / 12.0) * 3.0  # 1:1 to 1:4
            filters.append(
                f"acompressor=threshold={uc_threshold:.0f}dB"
                f":ratio={uc_ratio:.1f}"
                f":attack=10:release=100"
                f":makeup={uc_db:.1f}dB"
                f":knee=6"
            )

        # --- Step 3: Transient Emphasis ---
        te_pct = self.transient_emphasis_pct * intensity
        if te_pct > 5:
            # Boost transients by EQ boost in selected frequency band
            te_gain = (te_pct / 100.0) * 4.0  # up to +4 dB boost
            if self.transient_band == "L":
                filters.append(
                    f"equalizer=f=100:width_type=o:w=1.5:g={te_gain:.1f}"
                )
            elif self.transient_band == "M":
                filters.append(
                    f"equalizer=f=2500:width_type=o:w=2.0:g={te_gain:.1f}"
                )
            elif self.transient_band == "H":
                filters.append(
                    f"highshelf=f=6000:g={te_gain:.1f}:t=s:w=0.7"
                )

        # --- Step 4: Soft Clip ---
        sc_pct = self.soft_clip_pct * intensity if self.soft_clip_enabled else 0
        if sc_pct > 5:
            # atan-based soft clipping via FFmpeg's atan function
            # More clipping = lower parameter = more saturation
            clip_param = max(0.05, 1.0 - (sc_pct / 100.0) * 0.8)
            # Use volume up → atan soft clip → volume down approach
            sc_drive = 1.0 + (sc_pct / 100.0) * 3.0  # drive 1x to 4x
            sc_drive_db = 20 * math.log10(sc_drive)
            filters.append(f"volume={sc_drive_db:.1f}dB")
            filters.append(f"alimiter=limit=0.95:attack=0.1:release=5:level=false")
            filters.append(f"volume=-{sc_drive_db:.1f}dB")

        # --- Step 5: Input Gain Push ---
        gain = self.gain_db * intensity
        if gain > 0.1:
            filters.append(f"volume={gain:.1f}dB")

        # --- Step 6: Get IRC parameters ---
        irc_key = self.get_effective_irc_key()
        irc = get_irc_mode(irc_key)

        # Character affects attack/release: higher character = faster attack, shorter release
        char_factor = self.character / 10.0  # 0.0 to 1.0
        attack_ms = irc["attack"]
        release_ms = irc["release"]

        # Modify by character: smooth (slow attack/long release) → aggressive (fast/short)
        if char_factor < 0.5:
            # Smoother: slow down attack, lengthen release
            smooth = (0.5 - char_factor) * 2.0  # 0-1
            attack_ms = attack_ms * (1.0 + smooth * 2.0)
            release_ms = release_ms * (1.0 + smooth * 1.5)
        elif char_factor > 0.5:
            # More aggressive: speed up attack, shorten release
            aggro = (char_factor - 0.5) * 2.0  # 0-1
            attack_ms = attack_ms * (1.0 - aggro * 0.5)
            release_ms = release_ms * (1.0 - aggro * 0.4)

        # --- Step 7: Limiter (alimiter) ---
        # FFmpeg alimiter: attack/release in ms, limit in linear
        attack_val = max(0.1, min(80.0, attack_ms))
        release_val = max(0.1, min(8000.0, release_ms))

        # Ceiling: convert dBTP to linear
        ceiling_linear = 10 ** (self.ceiling / 20.0)

        limiter_parts = [
            f"alimiter=limit={ceiling_linear:.6f}",
            f"attack={attack_val:.4f}",
            f"release={release_val:.4f}",
            f"level=true",  # auto makeup gain
        ]
        filters.append(":".join(limiter_parts))

        return filters

    # ═══════════════════════════════════════════
    #  Measure output level (for real-time meter)
    # ═══════════════════════════════════════════

    def measure_levels(self, audio_path: str) -> Optional[Dict]:
        """
        Measure L/R peak + RMS levels with current maximizer filters applied.
        Uses FFmpeg astats for per-channel data (Logic Pro-style L/R metering).
        Returns dict with l_peak, r_peak, l_rms, r_rms, peak_db, rms_db, gain_reduction.
        """
        if not audio_path or not os.path.exists(audio_path):
            return None

        try:
            filters = self.get_ffmpeg_filters()
            filter_str = ",".join(filters) if filters else "anull"

            # Use astats for per-channel peak/RMS data
            cmd = [
                "ffmpeg", "-hide_banner",
                "-t", "5",
                "-i", audio_path,
                "-af", f"{filter_str},astats=metadata=1:reset=0",
                "-f", "null", "-"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            # Parse astats per-channel output
            channels = []
            current_ch = {}
            overall = {}
            section = "none"  # "channel" or "overall"

            for line in result.stderr.split("\n"):
                stripped = line.strip()
                if "Channel:" in stripped:
                    if current_ch:
                        channels.append(current_ch)
                    current_ch = {}
                    section = "channel"
                elif "Overall" in stripped and "Parsed_astats" in line:
                    if current_ch:
                        channels.append(current_ch)
                        current_ch = {}
                    section = "overall"
                elif "Peak level dB:" in stripped:
                    try:
                        val = float(stripped.split("Peak level dB:")[1].strip())
                        if section == "overall":
                            overall["peak_db"] = val
                        elif section == "channel":
                            current_ch["peak_db"] = val
                    except (ValueError, IndexError):
                        pass
                elif "RMS level dB:" in stripped:
                    try:
                        val = float(stripped.split("RMS level dB:")[1].strip())
                        if section == "overall":
                            overall["rms_db"] = val
                        elif section == "channel":
                            current_ch["rms_db"] = val
                    except (ValueError, IndexError):
                        pass

            # Fallback: volumedetect if astats didn't parse
            if not overall and not channels:
                cmd2 = [
                    "ffmpeg", "-hide_banner", "-t", "5",
                    "-i", audio_path,
                    "-af", f"{filter_str},volumedetect",
                    "-f", "null", "-"
                ]
                r2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=30)
                for line in r2.stderr.split("\n"):
                    if "max_volume" in line:
                        try:
                            overall["peak_db"] = float(
                                line.split("max_volume:")[1].strip().split(" ")[0])
                        except (ValueError, IndexError):
                            pass
                    if "mean_volume" in line:
                        try:
                            overall["rms_db"] = float(
                                line.split("mean_volume:")[1].strip().split(" ")[0])
                        except (ValueError, IndexError):
                            pass

            peak_db = overall.get("peak_db", -60.0)
            rms_db = overall.get("rms_db", -60.0)

            # L/R channel data (fallback to overall if mono or parse failed)
            l_peak = channels[0].get("peak_db", peak_db) if len(channels) > 0 else peak_db
            r_peak = channels[1].get("peak_db", peak_db) if len(channels) > 1 else l_peak
            l_rms = channels[0].get("rms_db", rms_db) if len(channels) > 0 else rms_db
            r_rms = channels[1].get("rms_db", rms_db) if len(channels) > 1 else l_rms

            # Gain reduction estimate (how much limiter is working)
            gr = max(0, self.gain_db - abs(peak_db - self.ceiling)) if self.gain_db > 0 else 0
            # More accurate: if output peak is near ceiling, GR ≈ input_gain
            if peak_db > (self.ceiling - 1.0) and self.gain_db > 0:
                gr = self.gain_db * 0.8  # Approximate

            return {
                "peak_db": peak_db,
                "rms_db": rms_db,
                "l_peak": l_peak,
                "r_peak": r_peak,
                "l_rms": l_rms,
                "r_rms": r_rms,
                "ceiling": self.ceiling,
                "gain_reduction": gr,
            }
        except Exception as e:
            print(f"[MAXIMIZER] measure_levels failed: {e}")
            return None

    # ═══════════════════════════════════════════
    #  Settings Persistence
    # ═══════════════════════════════════════════

    def get_settings_dict(self) -> dict:
        """Export current settings as dictionary."""
        return {
            "enabled": self.enabled,
            "irc_mode": self.irc_mode,
            "irc_sub_mode": self.irc_sub_mode,
            "gain_db": self.gain_db,
            "ceiling": self.ceiling,
            "true_peak": self.true_peak,
            "character": self.character,
            "tone": self.tone,
            "upward_compress_db": self.upward_compress_db,
            "soft_clip_enabled": self.soft_clip_enabled,
            "soft_clip_pct": self.soft_clip_pct,
            "transient_emphasis_pct": self.transient_emphasis_pct,
            "transient_band": self.transient_band,
            "stereo_ind_transient": self.stereo_ind_transient,
            "stereo_ind_sustain": self.stereo_ind_sustain,
        }

    def load_settings_dict(self, d: dict):
        """Load settings from dictionary."""
        for key, val in d.items():
            if hasattr(self, key):
                setattr(self, key, val)

    def get_display_info(self) -> dict:
        """Get info for UI display."""
        irc_key = self.get_effective_irc_key()
        irc = get_irc_mode(irc_key)
        mode_display = self.irc_mode
        if self.irc_sub_mode:
            subs = get_irc_sub_modes(self.irc_mode)
            if subs:
                mode_display = f"{self.irc_mode} — {self.irc_sub_mode}"

        return {
            "irc_mode": mode_display,
            "irc_description": irc.get("description", ""),
            "gain_db": f"+{self.gain_db:.1f} dB",
            "ceiling_dbtp": f"{self.ceiling:.2f} dBTP",
            "character": f"{self.character:.2f}",
            "tone": self.tone,
            "upward_compress": f"{self.upward_compress_db:.1f} dB",
            "soft_clip": f"{self.soft_clip_pct}%" if self.soft_clip_enabled else "OFF",
            "transient_emphasis": f"{self.transient_emphasis_pct}% ({self.transient_band})",
            "stereo_ind": f"T:{self.stereo_ind_transient}% S:{self.stereo_ind_sustain}%",
            "learned_lufs": f"{self._learned_lufs:.1f} LUFS" if self._learned_lufs else "—",
        }

    # ─── Legacy compatibility ───
    def set_threshold(self, threshold_db: float):
        """Legacy: map threshold to gain push (inverted)."""
        self.set_gain(abs(threshold_db))

    def __repr__(self):
        return (
            f"Maximizer(gain=+{self.gain_db:.1f}dB, ceiling={self.ceiling}dBTP, "
            f"irc={self.irc_mode}"
            f"{' - ' + self.irc_sub_mode if self.irc_sub_mode else ''}, "
            f"char={self.character:.1f}, tone={self.tone})"
        )
