"""
LongPlay Studio V5.0 — AI Assist Module
Genre-based AI recommendations engine.

Flow: Analyze Audio → Select Genre → Generate Recommendations → Preview → Apply

Features:
- Analyze audio and suggest optimal mastering settings
- Genre-aware recommendations (30+ genres)
- Intensity slider (0-100%) to scale all parameters
- One-click "AI Master" button
- Per-module recommendations with explanations
"""

from typing import Optional, Dict

import numpy as np

from .analyzer import AudioAnalyzer, AudioAnalysis
from .loudness import LoudnessMeter, LoudnessAnalysis
from .match_eq import _extract_mono_pcm, _compute_avg_spectrum, _spectrum_to_bands, THIRD_OCTAVE_CENTERS
from .genre_profiles import (
    GENRE_PROFILES, PLATFORM_TARGETS, IRC_MODES,
    get_genre_profile, get_genre_list, get_irc_mode,
)
from .maximizer import Maximizer
from .equalizer import Equalizer
from .dynamics import Dynamics
from .imager import Imager


class MasterRecommendation:
    """AI-generated mastering recommendations."""

    def __init__(self):
        self.genre = "All-Purpose Mastering"
        self.intensity = 50         # 0-100%
        self.platform = "YouTube"

        # Per-module recommendations
        self.maximizer = Maximizer()
        self.equalizer = Equalizer()
        self.dynamics = Dynamics()
        self.imager = Imager()

        # Explanations for user
        self.explanations = []

        # Confidence score (0-100)
        self.confidence = 0

    def to_dict(self) -> dict:
        return {
            "genre": self.genre,
            "intensity": self.intensity,
            "platform": self.platform,
            "confidence": self.confidence,
            "explanations": self.explanations,
            "maximizer": self.maximizer.get_settings_dict(),
            "equalizer": self.equalizer.get_settings_dict(),
            "dynamics": self.dynamics.get_settings_dict(),
            "imager": self.imager.get_settings_dict(),
        }


class AIAssist:
    """AI Assistant for automatic mastering recommendations."""

    def __init__(self, ffmpeg_path: str = "ffmpeg"):
        self.ffmpeg_path = ffmpeg_path
        self.analyzer = AudioAnalyzer(ffmpeg_path)
        self.loudness_meter = LoudnessMeter(ffmpeg_path)

    def analyze_and_recommend(
        self,
        audio_path: str,
        genre: str = "All-Purpose Mastering",
        platform: str = "YouTube",
        intensity: int = 50,
    ) -> Optional[MasterRecommendation]:
        """
        Analyze audio and generate mastering recommendations.

        Args:
            audio_path: Path to audio file
            genre: Target genre name
            platform: Target platform for loudness
            intensity: 0-100% processing intensity

        Returns:
            MasterRecommendation object or None
        """
        # Step 1: Analyze audio
        print(f"[AI ASSIST] Analyzing: {audio_path}")
        audio_analysis = self.analyzer.analyze(audio_path)
        if not audio_analysis:
            print("[AI ASSIST] Audio analysis failed")
            return None

        # Step 2: Measure loudness
        print("[AI ASSIST] Measuring loudness...")
        loudness = self.loudness_meter.analyze(audio_path)

        # Step 3: Get genre profile
        profile = get_genre_profile(genre)
        platform_target = PLATFORM_TARGETS.get(platform, PLATFORM_TARGETS["YouTube"])

        # Step 4: Generate recommendations
        rec = MasterRecommendation()
        rec.genre = genre
        rec.intensity = intensity
        rec.platform = platform

        intensity_factor = intensity / 100.0

        # --- Maximizer Recommendations ---
        rec.maximizer.enabled = True
        # V5.5 FIX: Use set_irc_mode() to trigger legacy name mapping
        # Default "IRC 2" (not legacy "IRC II") matches IRC_MODES dict keys
        irc_mode = profile.get("irc_mode", "IRC 2")
        irc_sub_mode = profile.get("irc_sub_mode", None)
        rec.maximizer.set_irc_mode(irc_mode, irc_sub_mode)
        rec.maximizer.tone = profile.get("tone", "Transparent")
        rec.maximizer.set_ceiling(
            profile.get("true_peak_ceiling", platform_target["true_peak"]))

        # Adjust GAIN PUSH based on how loud the audio already is
        # V5.5 FIX: Use set_gain() instead of non-existent .threshold attribute.
        # The Maximizer uses gain_db (0 to +20 dB) to push audio into the limiter.
        if loudness:
            current_lufs = loudness.integrated_lufs
            target_lufs = max(profile.get("target_lufs", -14), platform_target["target_lufs"])
            lufs_diff = target_lufs - current_lufs

            if lufs_diff > 0:
                # Audio is quieter than target — need gain push into limiter
                gain_push = min(20.0, abs(lufs_diff) * 1.2)
                rec.maximizer.set_gain(gain_push)
                rec.explanations.append(
                    f"Audio is {abs(lufs_diff):.1f} LU below target — "
                    f"Maximizer gain push: +{gain_push:.1f} dB"
                )
            else:
                # Audio is already loud enough — gentle limiting only
                rec.maximizer.set_gain(2.0)
                rec.explanations.append(
                    f"Audio is already {abs(lufs_diff):.1f} LU above target — "
                    f"Light limiting applied for peak control (+2.0 dB)"
                )

        # --- EQ Recommendations ---
        rec.equalizer.enabled = True
        rec.equalizer.load_genre_preset(genre)

        # Adjust EQ based on spectral analysis
        spectral = audio_analysis.spectral
        if spectral.brightness > 1.5:
            rec.explanations.append(
                "Audio is bright — EQ will reduce high frequencies slightly"
            )
            # Reduce high-end boost in preset
            for band in rec.equalizer.bands:
                if band.freq > 5000 and band.gain > 0:
                    band.gain *= 0.5
        elif spectral.brightness < 0.7:
            rec.explanations.append(
                "Audio is dark — EQ will add presence and air"
            )
            # Boost highs more
            for band in rec.equalizer.bands:
                if band.freq > 3000:
                    band.gain += 1.0

        # --- Dynamics Recommendations ---
        rec.dynamics.enabled = True
        rec.dynamics.load_genre_preset(genre)

        dynamic = audio_analysis.dynamic
        if dynamic.crest_factor_db < 8:
            rec.explanations.append(
                f"Audio is already compressed (crest factor: {dynamic.crest_factor_db:.1f} dB) — "
                f"Light compression only"
            )
            rec.dynamics.single_band.ratio = max(1.5, rec.dynamics.single_band.ratio * 0.5)
            rec.dynamics.single_band.threshold -= 4
        elif dynamic.crest_factor_db > 18:
            rec.explanations.append(
                f"Audio is very dynamic (crest factor: {dynamic.crest_factor_db:.1f} dB) — "
                f"More compression recommended"
            )
            rec.dynamics.single_band.ratio = min(6.0, rec.dynamics.single_band.ratio * 1.3)

        # --- Imager Recommendations ---
        rec.imager.enabled = True
        rec.imager.load_genre_preset(genre)

        stereo = audio_analysis.stereo
        if stereo.is_mono:
            rec.explanations.append(
                "Audio is mono — Imager will add stereo width"
            )
            rec.imager.width = min(rec.imager.width + 30, 180)
        elif stereo.correlation > 0.9:
            rec.explanations.append(
                "Audio has narrow stereo image — Imager will widen slightly"
            )
            rec.imager.width = min(rec.imager.width + 15, 160)
        elif stereo.correlation < 0.2:
            rec.explanations.append(
                "Audio has wide/phase-y stereo — Imager will narrow to improve mono compatibility"
            )
            rec.imager.width = max(rec.imager.width - 20, 80)

        # --- Confidence Score ---
        rec.confidence = self._calculate_confidence(audio_analysis, loudness)

        print(f"[AI ASSIST] Recommendation ready (confidence: {rec.confidence}%)")
        return rec

    def _calculate_confidence(
        self,
        audio_analysis: AudioAnalysis,
        loudness: Optional[LoudnessAnalysis],
    ) -> int:
        """Calculate confidence score for recommendations."""
        score = 50  # Base

        # Audio analysis quality
        if audio_analysis.duration_sec > 30:
            score += 10
        if audio_analysis.duration_sec > 120:
            score += 5

        # Loudness data available
        if loudness:
            score += 15
            if loudness.integrated_lufs > -50:
                score += 10

        # Spectral data quality
        spec = audio_analysis.spectral
        if spec.low_energy + spec.mid_energy + spec.high_energy > 0.5:
            score += 10

        return min(100, score)

    def get_genre_list(self) -> Dict:
        """Get available genres grouped by category."""
        return get_genre_list()

    def get_platform_list(self) -> Dict:
        """Get available platform targets."""
        return PLATFORM_TARGETS

    def compute_tonal_correction(self, audio_path: str,
                                  target_curve: str = "neutral") -> Optional[Dict[float, float]]:
        """Analyze tonal balance and compute corrective EQ.

        Compares the audio's average spectrum to a target frequency response
        curve and returns per-band corrections in dB.

        Args:
            audio_path: Path to the audio file to analyze.
            target_curve: One of "neutral", "warm", "bright", "bass_heavy".

        Returns:
            Dict mapping frequency (Hz) to correction in dB, or None on failure.
        """
        # ── Target curve definitions (offsets in dB per 1/3-octave band) ──
        # Bands: 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160,
        #        200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600,
        #        2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000
        n = len(THIRD_OCTAVE_CENTERS)

        # ISO 226 equal-loudness approximation: ears are less sensitive
        # at low and very high frequencies. "Neutral" compensates for this
        # so the perceived response sounds flat.
        iso226_offset = np.array([
            -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, -0.3, -0.1,
             0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.2,  0.3,
             0.2,  0.0, -0.2, -0.5, -0.8, -1.2, -1.5, -2.0, -3.0, -4.0,
            -5.0,
        ], dtype=np.float64)[:n]

        target_curves = {
            "neutral": iso226_offset,
            "warm": iso226_offset + np.concatenate([
                np.zeros(6),                          # below 63 Hz: no extra
                np.full(4, 0.5),                      # 80-160 Hz: +0.5
                np.full(6, 1.0),                      # 200-630 Hz: +1.0 low-mids
                np.full(4, 0.0),                      # 800-1600 Hz: flat
                np.full(5, -0.5),                     # 2k-5k Hz: -0.5
                np.full(6, -1.0),                     # 6.3k-20k Hz: -1.0 highs
            ])[:n],
            "bright": iso226_offset + np.concatenate([
                np.zeros(6),                          # below 63 Hz: no extra
                np.full(4, -0.5),                     # 80-160 Hz: -0.5
                np.full(6, -1.0),                     # 200-630 Hz: -1.0 low-mids
                np.full(4, 0.0),                      # 800-1600 Hz: flat
                np.full(5, 0.5),                      # 2k-5k Hz: +0.5
                np.full(6, 1.0),                      # 6.3k-20k Hz: +1.0 highs
            ])[:n],
            "bass_heavy": iso226_offset + np.concatenate([
                np.full(4, 2.0),                      # 20-40 Hz: +2 dB
                np.full(3, 2.0),                      # 50-80 Hz: +2 dB (below 100 Hz)
                np.full(1, 1.0),                      # 100 Hz: +1 dB taper
                np.zeros(n - 8),                      # rest: flat
            ])[:n],
        }

        if target_curve not in target_curves:
            print(f"[AI ASSIST] Unknown target curve: {target_curve}")
            return None

        target = target_curves[target_curve]

        # Step 1: Compute average spectrum of the audio (FFT over full track)
        samples = _extract_mono_pcm(audio_path, self.ffmpeg_path)
        if samples is None or len(samples) < 8192:
            print(f"[AI ASSIST] Could not extract audio from: {audio_path}")
            return None

        full_spectrum = _compute_avg_spectrum(samples)
        band_spectrum = _spectrum_to_bands(full_spectrum)

        # Step 2: Compare to target curve — normalize both to mean
        band_mean = np.mean(band_spectrum)
        target_mean = np.mean(target)
        normalized_spectrum = band_spectrum - band_mean
        normalized_target = target - target_mean

        # Step 3: Compute per-band correction in dB
        corrections = normalized_target - normalized_spectrum

        # Clamp corrections to a sensible range
        corrections = np.clip(corrections, -6.0, 6.0)

        # Step 4: Return corrections dict: {freq: correction_db}
        result: Dict[float, float] = {}
        for i, freq in enumerate(THIRD_OCTAVE_CENTERS):
            result[float(freq)] = round(float(corrections[i]), 2)

        print(f"[AI ASSIST] Tonal correction computed (curve={target_curve}, "
              f"bands={len(result)})")
        return result
