#!/usr/bin/env python3
"""
AI DJ Module for LongPlay Studio V4.25
Features:
1. Audio analysis (BPM, Key, Energy, Mood)
2. Smart playlist ordering
3. Best #1 track suggestion
4. Multiple shuffle versions
"""

import os
import subprocess
import json
import random
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# Try Rust backend for AI DJ analysis
_RUST_AIDJ = False
try:
    from longplay import PyAIDJ as _RustAIDJ, PyAudioAnalysis as _RustAudioAnalysis
    _RUST_AIDJ = True
except ImportError:
    pass


@dataclass
class AudioAnalysis:
    """Audio analysis result for a single track"""
    file_path: str
    filename: str
    duration_sec: float = 0.0
    bpm: float = 0.0
    key: str = ""
    energy: float = 0.0  # 0-1 scale
    loudness_db: float = -14.0
    intro_score: float = 0.0  # How good is this track as opener? 0-100
    
    # Computed properties
    @property
    def energy_bars(self) -> str:
        """Return energy as visual bars"""
        filled = int(self.energy * 6)
        return "█" * filled + "░" * (6 - filled)
    
    @property
    def bpm_category(self) -> str:
        """Categorize BPM"""
        if self.bpm < 80:
            return "slow"
        elif self.bpm < 110:
            return "medium"
        elif self.bpm < 140:
            return "upbeat"
        else:
            return "fast"


class AIDJ:
    """AI DJ - Analyze and arrange tracks intelligently"""
    
    # Musical key compatibility (Camelot wheel)
    KEY_COMPATIBILITY = {
        "C": ["C", "G", "F", "Am", "Em", "Dm"],
        "G": ["G", "D", "C", "Em", "Bm", "Am"],
        "D": ["D", "A", "G", "Bm", "F#m", "Em"],
        "A": ["A", "E", "D", "F#m", "C#m", "Bm"],
        "E": ["E", "B", "A", "C#m", "G#m", "F#m"],
        "B": ["B", "F#", "E", "G#m", "D#m", "C#m"],
        "F#": ["F#", "C#", "B", "D#m", "A#m", "G#m"],
        "C#": ["C#", "G#", "F#", "A#m", "E#m", "D#m"],
        "F": ["F", "C", "Bb", "Dm", "Am", "Gm"],
        "Bb": ["Bb", "F", "Eb", "Gm", "Dm", "Cm"],
        "Eb": ["Eb", "Bb", "Ab", "Cm", "Gm", "Fm"],
        "Ab": ["Ab", "Eb", "Db", "Fm", "Cm", "Bbm"],
        "Am": ["Am", "Em", "Dm", "C", "G", "F"],
        "Em": ["Em", "Bm", "Am", "G", "D", "C"],
        "Bm": ["Bm", "F#m", "Em", "D", "A", "G"],
        "F#m": ["F#m", "C#m", "Bm", "A", "E", "D"],
        "C#m": ["C#m", "G#m", "F#m", "E", "B", "A"],
        "G#m": ["G#m", "D#m", "C#m", "B", "F#", "E"],
        "D#m": ["D#m", "A#m", "G#m", "F#", "C#", "B"],
        "Dm": ["Dm", "Am", "Gm", "F", "C", "Bb"],
        "Gm": ["Gm", "Dm", "Cm", "Bb", "F", "Eb"],
        "Cm": ["Cm", "Gm", "Fm", "Eb", "Bb", "Ab"],
        "Fm": ["Fm", "Cm", "Bbm", "Ab", "Eb", "Db"],
        "Bbm": ["Bbm", "Fm", "Ebm", "Db", "Ab", "Gb"],
    }
    
    def __init__(self):
        self.analyses: Dict[str, AudioAnalysis] = {}
        self.shuffle_history: List[List[str]] = []  # Store previous shuffles
        self.current_shuffle_index = -1
        
    def analyze_track(self, file_path: str) -> AudioAnalysis:
        """Analyze a single audio track using librosa for accurate BPM/Key detection"""
        if file_path in self.analyses:
            return self.analyses[file_path]
            
        filename = os.path.basename(file_path)
        analysis = AudioAnalysis(file_path=file_path, filename=filename)
        
        try:
            # Get duration and basic info using ffprobe
            result = subprocess.run([
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_format", "-show_streams",
                file_path
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                if "format" in data:
                    analysis.duration_sec = float(data["format"].get("duration", 0))
                    
            # Get loudness using ffmpeg volumedetect
            result = subprocess.run([
                "ffmpeg", "-i", file_path,
                "-af", "volumedetect",
                "-f", "null", "-"
            ], capture_output=True, text=True, timeout=60)
            
            # Parse mean_volume from stderr
            import re
            match = re.search(r"mean_volume:\s*([-\d.]+)\s*dB", result.stderr)
            if match:
                analysis.loudness_db = float(match.group(1))
                
            # Estimate energy from loudness
            normalized_loudness = (analysis.loudness_db + 30) / 25
            analysis.energy = max(0, min(1, normalized_loudness))
            
            # Use librosa for accurate BPM and Key detection
            try:
                import librosa
                import numpy as np
                
                # Load audio (first 60 seconds for faster analysis)
                y, sr = librosa.load(file_path, duration=60, sr=22050)
                
                # Detect BPM using librosa
                tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
                if isinstance(tempo, np.ndarray):
                    tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
                analysis.bpm = round(float(tempo))
                
                # Detect Key using chroma features
                chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
                chroma_mean = np.mean(chroma, axis=1)
                
                # Map to key names
                key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                key_index = int(np.argmax(chroma_mean))
                
                # Determine major or minor
                # Simple heuristic: check relative minor/major strength
                major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
                minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
                
                # Rotate profiles to match detected key
                major_corr = np.corrcoef(chroma_mean, np.roll(major_profile, key_index))[0, 1]
                minor_corr = np.corrcoef(chroma_mean, np.roll(minor_profile, key_index))[0, 1]
                
                if minor_corr > major_corr:
                    analysis.key = key_names[key_index] + "m"
                else:
                    analysis.key = key_names[key_index]
                    
            except ImportError:
                # Fallback if librosa not available
                print(f"librosa not available, using fallback for {filename}")
                analysis.bpm = 100 + int(analysis.energy * 40)
                keys = ["C", "G", "D", "A", "Am", "Em", "F", "Dm"]
                analysis.key = keys[hash(filename) % len(keys)]
            except Exception as e:
                print(f"librosa analysis error for {filename}: {e}")
                analysis.bpm = 100 + int(analysis.energy * 40)
                keys = ["C", "G", "D", "A", "Am", "Em", "F", "Dm"]
                analysis.key = keys[hash(filename) % len(keys)]
            
            # Calculate intro score
            energy_score = 100 - abs(analysis.energy - 0.6) * 100
            bpm_score = 100 - abs(analysis.bpm - 100) * 0.5
            loudness_score = 100 - abs(analysis.loudness_db + 14) * 2
            
            analysis.intro_score = (energy_score * 0.4 + bpm_score * 0.3 + loudness_score * 0.3)
            analysis.intro_score = max(0, min(100, analysis.intro_score))
            
        except Exception as e:
            print(f"Analysis error for {filename}: {e}")
            analysis.duration_sec = 180
            analysis.bpm = 100
            analysis.key = "C"
            analysis.energy = 0.5
            analysis.intro_score = 50
            
        self.analyses[file_path] = analysis
        return analysis
    
    def analyze_all(self, file_paths: List[str], progress_callback=None) -> List[AudioAnalysis]:
        """Analyze all tracks with optional progress callback"""
        results = []
        total = len(file_paths)
        
        for i, path in enumerate(file_paths):
            analysis = self.analyze_track(path)
            results.append(analysis)
            
            if progress_callback:
                progress_callback(i + 1, total, analysis.filename)
                
        return results
    
    def suggest_order(self, file_paths: List[str], strategy: str = "smooth") -> List[str]:
        """
        Suggest optimal track order
        
        Strategies:
        - "smooth": Gradual energy flow, key compatibility
        - "energy_up": Start low, build up energy
        - "energy_down": Start high, wind down
        - "random_smart": Random but with key compatibility
        """
        if len(file_paths) <= 1:
            return file_paths
            
        # Ensure all tracks are analyzed
        analyses = [self.analyze_track(p) for p in file_paths]
        
        if strategy == "smooth":
            return self._order_smooth(analyses)
        elif strategy == "energy_up":
            return self._order_energy_up(analyses)
        elif strategy == "energy_down":
            return self._order_energy_down(analyses)
        elif strategy == "random_smart":
            return self._order_random_smart(analyses)
        else:
            return file_paths
    
    def _order_smooth(self, analyses: List[AudioAnalysis]) -> List[str]:
        """Order for smooth transitions - key and energy compatibility"""
        if not analyses:
            return []
            
        # Find best opener
        sorted_by_intro = sorted(analyses, key=lambda x: x.intro_score, reverse=True)
        ordered = [sorted_by_intro[0]]
        remaining = [a for a in analyses if a != ordered[0]]
        
        while remaining:
            current = ordered[-1]
            
            # Score each remaining track
            best_score = -1
            best_track = None
            
            for candidate in remaining:
                score = self._transition_score(current, candidate)
                if score > best_score:
                    best_score = score
                    best_track = candidate
                    
            if best_track:
                ordered.append(best_track)
                remaining.remove(best_track)
            else:
                # Fallback: add first remaining
                ordered.append(remaining.pop(0))
                
        return [a.file_path for a in ordered]
    
    def _order_energy_up(self, analyses: List[AudioAnalysis]) -> List[str]:
        """Order by increasing energy"""
        sorted_analyses = sorted(analyses, key=lambda x: x.energy)
        return [a.file_path for a in sorted_analyses]
    
    def _order_energy_down(self, analyses: List[AudioAnalysis]) -> List[str]:
        """Order by decreasing energy"""
        sorted_analyses = sorted(analyses, key=lambda x: x.energy, reverse=True)
        return [a.file_path for a in sorted_analyses]
    
    def _order_random_smart(self, analyses: List[AudioAnalysis]) -> List[str]:
        """Random order but respecting key compatibility"""
        if not analyses:
            return []
            
        shuffled = analyses.copy()
        random.shuffle(shuffled)
        
        # Find best opener from top 3 random picks
        opener = max(shuffled[:3], key=lambda x: x.intro_score)
        shuffled.remove(opener)
        
        ordered = [opener]
        
        while shuffled:
            current = ordered[-1]
            
            # Find compatible tracks
            compatible = [a for a in shuffled if self._keys_compatible(current.key, a.key)]
            
            if compatible:
                # Pick random from compatible
                next_track = random.choice(compatible)
            else:
                # Pick any random
                next_track = random.choice(shuffled)
                
            ordered.append(next_track)
            shuffled.remove(next_track)
            
        return [a.file_path for a in ordered]
    
    def _transition_score(self, from_track: AudioAnalysis, to_track: AudioAnalysis) -> float:
        """Calculate how good a transition between two tracks would be"""
        score = 0.0
        
        # Key compatibility (0-40 points)
        if self._keys_compatible(from_track.key, to_track.key):
            score += 40
        elif from_track.key == to_track.key:
            score += 35
            
        # Energy flow (0-30 points) - prefer gradual changes
        energy_diff = abs(from_track.energy - to_track.energy)
        score += 30 * (1 - energy_diff)
        
        # BPM compatibility (0-20 points) - prefer within 10 BPM
        bpm_diff = abs(from_track.bpm - to_track.bpm)
        if bpm_diff < 5:
            score += 20
        elif bpm_diff < 10:
            score += 15
        elif bpm_diff < 20:
            score += 10
            
        # Loudness compatibility (0-10 points)
        loudness_diff = abs(from_track.loudness_db - to_track.loudness_db)
        score += max(0, 10 - loudness_diff)
        
        return score
    
    def _keys_compatible(self, key1: str, key2: str) -> bool:
        """Check if two musical keys are compatible"""
        if key1 == key2:
            return True
        compatible = self.KEY_COMPATIBILITY.get(key1, [])
        return key2 in compatible
    
    def get_best_opener(self, file_paths: List[str], top_n: int = 3) -> List[Tuple[str, float]]:
        """Get top N best tracks for opening position"""
        analyses = [self.analyze_track(p) for p in file_paths]
        sorted_analyses = sorted(analyses, key=lambda x: x.intro_score, reverse=True)
        
        return [(a.file_path, a.intro_score) for a in sorted_analyses[:top_n]]
    
    def shuffle_again(self, file_paths: List[str]) -> List[str]:
        """Generate a new shuffle, different from previous ones"""
        max_attempts = 10
        
        for _ in range(max_attempts):
            # Random smart shuffle
            new_order = self._order_random_smart([self.analyze_track(p) for p in file_paths])
            
            # Check if it's different from recent shuffles
            is_unique = True
            for prev_order in self.shuffle_history[-5:]:  # Check last 5
                if new_order == prev_order:
                    is_unique = False
                    break
                    
            if is_unique:
                self.shuffle_history.append(new_order)
                self.current_shuffle_index = len(self.shuffle_history) - 1
                return new_order
                
        # Fallback: return last attempt
        return new_order
    
    def get_previous_shuffle(self) -> Optional[List[str]]:
        """Navigate to previous shuffle in history"""
        if self.current_shuffle_index > 0:
            self.current_shuffle_index -= 1
            return self.shuffle_history[self.current_shuffle_index]
        return None
    
    def get_next_shuffle(self) -> Optional[List[str]]:
        """Navigate to next shuffle in history"""
        if self.current_shuffle_index < len(self.shuffle_history) - 1:
            self.current_shuffle_index += 1
            return self.shuffle_history[self.current_shuffle_index]
        return None
    
    def get_playlist_stats(self, file_paths: List[str]) -> Dict:
        """Get statistics for current playlist order"""
        analyses = [self.analyze_track(p) for p in file_paths]
        
        if not analyses:
            return {}
            
        # Calculate flow smoothness
        total_transition_score = 0
        for i in range(len(analyses) - 1):
            total_transition_score += self._transition_score(analyses[i], analyses[i+1])
            
        max_possible = (len(analyses) - 1) * 100
        smoothness = (total_transition_score / max_possible * 100) if max_possible > 0 else 0
        
        # Energy balance
        energies = [a.energy for a in analyses]
        avg_energy = sum(energies) / len(energies)
        energy_variance = sum((e - avg_energy) ** 2 for e in energies) / len(energies)
        energy_balance = 100 - (energy_variance * 200)  # Lower variance = better balance
        
        return {
            "smoothness": round(smoothness, 1),
            "energy_balance": round(max(0, min(100, energy_balance)), 1),
            "avg_bpm": round(sum(a.bpm for a in analyses) / len(analyses), 1),
            "avg_energy": round(avg_energy, 2),
            "total_duration_sec": sum(a.duration_sec for a in analyses),
            "track_count": len(analyses),
        }


class YouTubeGenerator:
    """Generate YouTube metadata from playlist"""
    
    # SEO Keywords Database from SEO_KEYWORD_AGENT_V2.1 - Full Version
    SEO_KEYWORDS = {
        "high_volume": [
            # Top 20 highest search volume keywords
            "chill music", "relaxing music", "study music", "sleep music",
            "lofi beats", "cafe music", "jazz music", "piano music",
            "background music", "calm music", "peaceful music", "ambient music",
            "soft music", "soothing music", "meditation music", "yoga music",
            "spa music", "nature sounds", "rain sounds", "ocean waves",
            # Thai high volume
            "เพลงเพราะๆ", "เพลงฟังสบาย", "เพลงชิลๆ", "เพลงผ่อนคลาย"
        ],
        "relax_chill": [
            # English chill keywords
            "chill vibes", "relaxing music no ads", "soft music", "gentle music",
            "soothing music", "tranquil music", "peaceful sounds", "calm vibes",
            "easy listening", "mellow music", "laid back music", "chill beats",
            "relaxing instrumental", "stress relief music", "unwind music",
            # Thai chill keywords
            "เพลงเพราะๆ ฟังสบาย", "เพลงฟังชิลๆ", "เพลงฟังสบายๆ", "เพลงเพราะๆ ผ่อนคลาย",
            "เพลงฟังเพลินๆ", "เพลงเบาๆ", "เพลงนั่งชิล", "เพลงพักผ่อน"
        ],
        "study_work": [
            # English study/work keywords
            "study music", "work music", "focus music", "concentration music",
            "lofi beats to study", "productivity music", "deep focus music",
            "music for studying", "homework music", "reading music",
            "office music", "work from home music", "coding music",
            "music for concentration", "brain power music",
            # Thai study/work keywords
            "เพลงฟังตอนทำงาน", "เพลงทำงานไม่มีโฆษณา", "เพลงเพิ่มสมาธิ",
            "เพลงอ่านหนังสือ", "เพลงทำการบ้าน", "เพลงโฟกัส", "เพลงเรียน"
        ],
        "sleep": [
            # English sleep keywords
            "sleep music", "deep sleep music", "relaxing sleep music",
            "sleeping music 8 hours", "peaceful sleep music", "calming music for sleep",
            "insomnia music", "bedtime music", "night music", "lullaby for adults",
            "sleep meditation", "sleep sounds", "white noise sleep",
            # Thai sleep keywords
            "เพลงก่อนนอน", "เพลงหลับสบาย", "เพลงกล่อมนอน", "เพลงนอนหลับ",
            "เพลงฟังก่อนนอน", "เพลงหลับลึก", "เพลงผ่อนคลายก่อนนอน"
        ],
        "cafe_jazz": [
            # English cafe/jazz keywords
            "cafe music", "coffee shop music", "jazz music", "bossa nova",
            "cafe jazz", "morning coffee music", "smooth jazz", "cafe ambience",
            "relaxing cafe music", "coffee house music", "jazz cafe",
            "cozy cafe music", "rainy cafe", "paris cafe music",
            "jazz instrumental", "lounge music", "cocktail music",
            # Thai cafe keywords
            "เพลงร้านกาแฟ", "เพลงคาเฟ่", "เพลงเปิดร้านกาแฟ", "เพลงแจ๊ส",
            "เพลงบอสซาโนวา", "เพลงคาเฟ่ชิลๆ"
        ],
        "piano_instrumental": [
            # English piano keywords
            "piano music", "relaxing piano", "soft piano music", "calm piano",
            "piano instrumental", "beautiful piano music", "peaceful piano",
            "piano for studying", "emotional piano", "sad piano",
            "romantic piano", "classical piano", "piano covers",
            "piano sleep music", "piano meditation",
            # Thai piano keywords
            "เพลงเปียโนเพราะๆ", "เพลงเปียโนผ่อนคลาย", "เพลงเปียโนฟังสบาย",
            "เพลงบรรเลงเปียโน", "เปียโนคลาสสิค"
        ]
    }
    
    THEMES = {
        "cafe": {
            "emoji": "☕",
            "thai_name": "เพลงคาเฟ่",
            "english_name": "Cafe Music",
            "keywords_th": ["เพลงเปิดร้านกาแฟ", "เพลงคาเฟ่", "ฟังสบาย", "เพลงทำงาน", "เพลงอ่านหนังสือ"],
            "keywords_en": ["cafe music", "coffee shop music", "work music", "study music", "chill vibes"],
        },
        "driving": {
            "emoji": "🚗",
            "thai_name": "เพลงขับรถ",
            "english_name": "Driving Music",
            "keywords_th": ["เพลงขับรถ", "เพลงเดินทาง", "กลับบ้าน", "เพลงทางไกล", "เพลงชิลๆ"],
            "keywords_en": ["driving music", "road trip", "travel music", "chill drive", "highway music"],
        },
        "sleep": {
            "emoji": "🌙",
            "thai_name": "เพลงก่อนนอน",
            "english_name": "Sleep Music",
            "keywords_th": ["เพลงก่อนนอน", "เพลงผ่อนคลาย", "นอนหลับ", "เพลงสงบ", "เพลงนอน"],
            "keywords_en": ["sleep music", "relaxing music", "calm music", "bedtime music", "peaceful"],
        },
        "workout": {
            "emoji": "💪",
            "thai_name": "เพลงออกกำลังกาย",
            "english_name": "Workout Music",
            "keywords_th": ["เพลงออกกำลังกาย", "เพลงวิ่ง", "เพลงฟิตเนส", "เพลงปลุกพลัง", "เพลงมันส์"],
            "keywords_en": ["workout music", "gym music", "fitness music", "running music", "motivation"],
        },
        "focus": {
            "emoji": "🎯",
            "thai_name": "เพลงโฟกัส",
            "english_name": "Focus Music",
            "keywords_th": ["เพลงทำงาน", "เพลงโฟกัส", "เพลงมีสมาธิ", "เพลงอ่านหนังสือ", "เพลงเรียน"],
            "keywords_en": ["focus music", "concentration", "study music", "productivity", "deep work"],
        },
        "chill": {
            "emoji": "🌴",
            "thai_name": "เพลงชิลๆ",
            "english_name": "Chill Vibes",
            "keywords_th": ["เพลงชิลๆ", "ฟังสบาย", "เพลงผ่อนคลาย", "เพลงพักผ่อน", "เพลงเบาๆ"],
            "keywords_en": ["chill vibes", "lofi", "relaxing", "easy listening", "ambient"],
        },
    }
    
    def __init__(self, channel_name: str = "Chillin' Vibes"):
        self.channel_name = channel_name
        self.max_tags_length = 500
        
    def generate_title(self, volume: int, theme: str, duration_str: str) -> str:
        """Generate YouTube title"""
        theme_data = self.THEMES.get(theme, self.THEMES["chill"])
        emoji = theme_data["emoji"]
        thai_name = theme_data["thai_name"]
        
        return f"Vol.{volume} {emoji} {thai_name} ฟังสบายไม่มีสะดุด {duration_str}"
    
    def generate_description(self, volume: int, theme: str, tracks: List[Dict], 
                            duration_str: str, playlist_url: str = "") -> str:
        """Generate full YouTube description with timestamps"""
        theme_data = self.THEMES.get(theme, self.THEMES["chill"])
        emoji = theme_data["emoji"]
        thai_name = theme_data["thai_name"]
        english_name = theme_data["english_name"]
        
        # Title line
        title = f"Vol.{volume} {emoji} {thai_name} ฟังสบายไม่มีสะดุด {duration_str}"
        
        # Extended description
        extended = f"""
{title}

รวมเพลงฟังสบายๆ เอาไว้ฟังตอนทำงาน อ่านหนังสือ หรือขับรถ
ไม่ว่าจะเช้า กลางวัน หรือดึกดื่น ก็ฟังได้ตลอด 🎵

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔔 กดติดตาม & กดกระดิ่ง เพื่อไม่พลาดเพลงใหม่ๆ ทุกสัปดาห์!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎵 TRACKLIST

"""
        
        # Add timestamps
        for track in tracks:
            timestamp = track.get("timestamp", "0:00")
            name = track.get("name", "Unknown")
            extended += f"{timestamp} {name}\n"
            
        # Add footer
        extended += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎶 Playlist: {playlist_url if playlist_url else 'Coming soon!'}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#{''.join(theme_data['keywords_th'][:3]).replace(' ', '')} #{english_name.replace(' ', '')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
© {self.channel_name}
"""
        
        return extended
    
    def get_seo_keywords(self, seo_type: str) -> list:
        """Get SEO keywords by type"""
        return self.SEO_KEYWORDS.get(seo_type, [])
        
    def generate_tags(self, theme: str, seo_type: str = "auto", extra_tags: List[str] = None) -> str:
        """Generate tags within 500 character limit with SEO keywords"""
        theme_data = self.THEMES.get(theme, self.THEMES["chill"])
        
        all_tags = []
        
        # Add SEO keywords first (higher priority)
        if seo_type != "auto":
            seo_keywords = self.get_seo_keywords(seo_type)
            all_tags.extend(seo_keywords[:10])  # Top 10 SEO keywords
        
        # Add theme keywords
        all_tags.extend(theme_data["keywords_th"])
        all_tags.extend(theme_data["keywords_en"])
        
        # Add common tags
        common_tags = [
            "longplay", "1hour", "nonstop", "playlist", 
            "เพลงไม่มีโฆษณา", "เพลงยาว", "เพลงรวม",
            self.channel_name.replace(" ", "").lower()
        ]
        all_tags.extend(common_tags)
        
        # Add extra tags
        if extra_tags:
            all_tags.extend(extra_tags)
            
        # Build tag string within limit
        result = []
        current_length = 0
        
        for tag in all_tags:
            tag_text = tag if tag.startswith("#") else f"#{tag}"
            tag_length = len(tag_text) + 1  # +1 for space
            
            if current_length + tag_length <= self.max_tags_length:
                result.append(tag_text)
                current_length += tag_length
            else:
                break
                
        return " ".join(result)
    
    def generate_timestamps(self, tracks: List[Dict]) -> List[Dict]:
        """Add timestamps to tracks based on duration"""
        timestamped = []
        current_time_sec = 0
        
        for track in tracks:
            # Format timestamp
            minutes = int(current_time_sec // 60)
            seconds = int(current_time_sec % 60)
            timestamp = f"{minutes}:{seconds:02d}"
            
            timestamped.append({
                "name": track.get("name", "Unknown"),
                "timestamp": timestamp,
                "duration_sec": track.get("duration_sec", 180)
            })
            
            # Add duration for next track
            current_time_sec += track.get("duration_sec", 180)
            
        return timestamped
    
    def format_duration(self, total_seconds: float) -> str:
        """Format total duration as human readable"""
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        
        if hours > 0:
            return f"{hours} ชั่วโมง" if minutes == 0 else f"{hours} ชั่วโมง {minutes} นาที"
        else:
            return f"{minutes} นาที"


# Test
if __name__ == "__main__":
    print("🎧 AI DJ Module Test")
    
    # Test YouTube Generator
    yt = YouTubeGenerator()
    
    # Sample tracks
    tracks = [
        {"name": "Polaroid Summers", "duration_sec": 226},
        {"name": "Letters Never Sent", "duration_sec": 234},
        {"name": "Radio Silence", "duration_sec": 218},
        {"name": "Tangled in December", "duration_sec": 245},
    ]
    
    timestamped = yt.generate_timestamps(tracks)
    print("\n📋 Timestamps:")
    for t in timestamped:
        print(f"  {t['timestamp']} - {t['name']}")
        
    print("\n📝 Title:")
    print(yt.generate_title(24, "cafe", "1 ชั่วโมง"))
    
    print("\n🏷️ Tags:")
    print(yt.generate_tags("cafe"))
    print(f"Length: {len(yt.generate_tags('cafe'))} chars")
