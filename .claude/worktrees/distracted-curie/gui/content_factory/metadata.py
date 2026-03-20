"""
Content Factory — Metadata Generator

Auto-generates YouTube-optimized metadata:
  - Titles (SEO-friendly)
  - Descriptions (with tracklist + timestamps for chapters)
  - Tags (genre-aware)
  - #Shorts hashtag for Short videos

No AI API required — uses templates + channel/genre context.
"""

from __future__ import annotations

import datetime
from typing import List, Optional, Tuple

from .models import LongVideoPlan, ShortVideoPlan, ContentFactoryConfig


class MetadataGenerator:
    """
    Generates YouTube metadata (title, description, tags) for content.

    Usage:
        gen = MetadataGenerator(config)
        gen.generate_long_metadata(plan)
        gen.generate_short_metadata(plan)
    """

    # Genre → relevant tags mapping
    GENRE_TAGS = {
        "Chill": ["chill", "relaxing", "chill vibes", "chill music", "lofi", "study music"],
        "Lofi": ["lofi", "lo-fi", "lofi hip hop", "chill beats", "study beats", "lofi chill"],
        "Jazz": ["jazz", "smooth jazz", "jazz music", "jazz vibes", "relaxing jazz"],
        "R&B": ["rnb", "r&b", "soul", "r&b music", "smooth r&b"],
        "Pop": ["pop", "pop music", "pop hits", "pop vibes"],
        "Rock": ["rock", "rock music", "rock vibes", "indie rock"],
        "EDM": ["edm", "electronic", "dance music", "edm mix", "electronic music"],
        "Hip-Hop": ["hip hop", "rap", "hip hop mix", "rap music"],
        "Classical": ["classical", "classical music", "orchestra", "piano"],
        "Ambient": ["ambient", "ambient music", "relaxing", "meditation", "sleep"],
    }

    # Common base tags
    BASE_TAGS = ["music", "playlist", "mix", "compilation"]
    SHORT_BASE_TAGS = ["shorts", "music shorts", "short video"]

    def __init__(self, config: Optional[ContentFactoryConfig] = None):
        self._config = config
        self._channel_name = config.channel_name if config else ""
        self._genre = config.channel_genre if config else "Chill"

    # ─── Long Video Metadata ─────────────────────────────────────

    def generate_long_metadata(self, plan: LongVideoPlan) -> None:
        """Generate title, description, tags for a long video."""
        plan.title = self._long_title(plan)
        plan.description = self._long_description(plan)
        plan.tags = self._long_tags(plan)

    def _long_title(self, plan: LongVideoPlan) -> str:
        """Generate SEO-friendly title for long compilation."""
        genre = self._genre
        n_songs = len(plan.songs)
        duration_hr = plan.estimated_duration / 3600

        if duration_hr >= 1.0:
            dur_str = f"{duration_hr:.0f} Hour"
        else:
            dur_str = f"{plan.estimated_duration / 60:.0f} Min"

        # Template variations
        today = datetime.date.today()
        month_year = today.strftime("%B %Y")

        templates = [
            f"{genre} Music Mix {month_year} — {dur_str} | {n_songs} Songs",
            f"{dur_str} {genre} Playlist — Best {genre} Music {today.year}",
            f"Best {genre} Mix {month_year} — {n_songs} Songs, {dur_str}",
        ]

        # Pick based on plan index (simple rotation)
        idx = hash(plan.video_id) % len(templates)
        return templates[idx]

    def _long_description(self, plan: LongVideoPlan) -> str:
        """Generate description with tracklist + timestamps."""
        lines = []

        # Header
        genre = self._genre
        lines.append(f"🎵 {genre} Music Compilation — {len(plan.songs)} Songs")
        lines.append("")

        if self._channel_name:
            lines.append(f"Welcome to {self._channel_name}!")
            lines.append(f"Enjoy this curated {genre.lower()} music mix. "
                         "Perfect for studying, working, relaxing, or just vibing.")
        lines.append("")

        # Tracklist with timestamps (YouTube chapters)
        lines.append("📋 Tracklist:")
        if plan.chapters:
            for timestamp, title in plan.chapters:
                ts_str = self._format_timestamp(timestamp)
                lines.append(f"  {ts_str} {title}")
        else:
            # Generate from songs
            pos = 0.0
            for i, song in enumerate(plan.songs):
                ts_str = self._format_timestamp(pos)
                label = f"{song.artist} — {song.title}" if song.artist else song.title
                lines.append(f"  {ts_str} {label}")
                pos += song.duration_sec
                if i < len(plan.songs) - 1:
                    pos -= plan.crossfade_sec

        lines.append("")

        # Footer
        lines.append("─" * 40)
        if self._channel_name:
            lines.append(f"🔔 Subscribe to {self._channel_name} for more {genre.lower()} music!")
        lines.append("")
        lines.append(f"#{''.join(genre.split())} #Music #Playlist #Mix #Compilation")

        return "\n".join(lines)

    def _long_tags(self, plan: LongVideoPlan) -> List[str]:
        """Generate tags for long compilation."""
        tags = list(self.BASE_TAGS)
        tags.extend(self.GENRE_TAGS.get(self._genre, []))

        # Add genre variations
        genre_lower = self._genre.lower()
        tags.extend([
            f"{genre_lower} mix",
            f"{genre_lower} playlist",
            f"{genre_lower} compilation",
            f"best {genre_lower}",
            f"{genre_lower} music {datetime.date.today().year}",
        ])

        if self._channel_name:
            tags.append(self._channel_name.lower())

        # Deduplicate
        seen = set()
        unique_tags = []
        for t in tags:
            if t.lower() not in seen:
                seen.add(t.lower())
                unique_tags.append(t)

        return unique_tags[:30]  # YouTube max 500 chars, ~30 tags safe

    # ─── Short Video Metadata ────────────────────────────────────

    def generate_short_metadata(self, plan: ShortVideoPlan) -> None:
        """Generate title, description, tags for a Short."""
        plan.title = self._short_title(plan)
        plan.description = self._short_description(plan)
        plan.tags = self._short_tags(plan)

    def _short_title(self, plan: ShortVideoPlan) -> str:
        """Generate Short title."""
        if not plan.song:
            return "Music Short"

        song = plan.song
        if song.artist:
            return f"{song.title} — {song.artist} #Shorts"
        return f"{song.title} #Shorts"

    def _short_description(self, plan: ShortVideoPlan) -> str:
        """Generate Short description."""
        lines = []
        if plan.song:
            song = plan.song
            lines.append(f"🎵 {song.title}")
            if song.artist:
                lines.append(f"🎤 {song.artist}")
            lines.append("")

        genre = self._genre
        lines.append(f"Enjoy this {genre.lower()} music clip!")
        lines.append("")

        if self._channel_name:
            lines.append(f"🔔 Subscribe to {self._channel_name} for more!")
        lines.append("")
        lines.append(f"#Shorts #{genre} #Music #{''.join(genre.split())}Vibes")

        return "\n".join(lines)

    def _short_tags(self, plan: ShortVideoPlan) -> List[str]:
        """Generate tags for Short."""
        tags = list(self.SHORT_BASE_TAGS)
        tags.extend(self.GENRE_TAGS.get(self._genre, []))

        genre_lower = self._genre.lower()
        tags.extend([
            f"{genre_lower} shorts",
            f"{genre_lower} short",
            f"{genre_lower} clip",
        ])

        if plan.song:
            if plan.song.artist:
                tags.append(plan.song.artist.lower())

        if self._channel_name:
            tags.append(self._channel_name.lower())

        # Deduplicate
        seen = set()
        unique_tags = []
        for t in tags:
            if t.lower() not in seen:
                seen.add(t.lower())
                unique_tags.append(t)

        return unique_tags[:20]

    # ─── Utility ─────────────────────────────────────────────────

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """Format seconds as HH:MM:SS or MM:SS."""
        total_sec = int(max(0, seconds))
        h = total_sec // 3600
        m = (total_sec % 3600) // 60
        s = total_sec % 60
        if h > 0:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"
