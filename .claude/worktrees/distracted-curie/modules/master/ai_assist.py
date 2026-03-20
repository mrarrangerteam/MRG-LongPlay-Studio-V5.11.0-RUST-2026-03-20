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
from .analyzer import AudioAnalyzer, AudioAnalysis
from .loudness import LoudnessMeter, LoudnessAnalysis
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
