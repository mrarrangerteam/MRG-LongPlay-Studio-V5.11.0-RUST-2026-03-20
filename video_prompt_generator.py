#!/usr/bin/env python3
"""
Video Prompt Generator - Midjourney Style for meta.ai Video
LongPlay Studio V4.26

Features:
1. Analyze video content and extract visual elements
2. Generate Midjourney-style prompts for AI video generation
3. Support meta.ai video format
4. Cinematic, mood, lighting, camera movement descriptions
5. Multiple prompt styles (Cinematic, Anime, Documentary, etc.)
"""

import os
import subprocess
import json
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path


@dataclass
class VideoAnalysis:
    """Analysis result for a video file"""
    file_path: str
    filename: str
    duration_sec: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    codec: str = ""
    
    # Visual analysis
    dominant_colors: List[str] = field(default_factory=list)
    brightness: str = "medium"  # dark, medium, bright
    contrast: str = "normal"  # low, normal, high
    motion_level: str = "medium"  # static, slow, medium, fast, dynamic
    
    # Scene analysis
    scene_type: str = ""  # indoor, outdoor, abstract, nature, urban, etc.
    time_of_day: str = ""  # dawn, day, golden_hour, dusk, night
    weather: str = ""  # clear, cloudy, rainy, foggy, etc.
    
    # Generated prompts
    prompts: Dict[str, str] = field(default_factory=dict)


# ==================== Prompt Templates ====================
MIDJOURNEY_STYLES = {
    "cinematic": {
        "prefix": "cinematic shot,",
        "camera": ["wide angle", "close-up", "tracking shot", "dolly zoom", "crane shot", "steadicam"],
        "lighting": ["dramatic lighting", "rim lighting", "volumetric lighting", "natural lighting", "neon glow"],
        "mood": ["epic", "emotional", "tense", "serene", "mysterious", "nostalgic"],
        "suffix": "--ar 16:9 --v 6 --style raw"
    },
    "anime": {
        "prefix": "anime style,",
        "camera": ["dynamic angle", "action shot", "portrait", "scenic view"],
        "lighting": ["soft glow", "dramatic shadows", "sunset lighting", "moonlight"],
        "mood": ["vibrant", "melancholic", "action-packed", "peaceful", "dramatic"],
        "suffix": "--ar 16:9 --niji 6"
    },
    "documentary": {
        "prefix": "documentary footage,",
        "camera": ["handheld", "interview shot", "b-roll", "aerial view", "time-lapse"],
        "lighting": ["natural light", "available light", "soft diffused"],
        "mood": ["authentic", "intimate", "observational", "journalistic"],
        "suffix": "--ar 16:9 --v 6"
    },
    "music_video": {
        "prefix": "music video aesthetic,",
        "camera": ["slow motion", "quick cuts", "360 spin", "dutch angle", "fisheye"],
        "lighting": ["neon lights", "strobe", "colored gels", "silhouette", "lens flare"],
        "mood": ["energetic", "dreamy", "hypnotic", "rebellious", "romantic"],
        "suffix": "--ar 16:9 --v 6 --stylize 750"
    },
    "abstract": {
        "prefix": "abstract visual,",
        "camera": ["macro", "kaleidoscope", "fluid motion", "geometric patterns"],
        "lighting": ["ethereal glow", "prismatic", "bioluminescent", "aurora"],
        "mood": ["surreal", "meditative", "psychedelic", "minimalist", "cosmic"],
        "suffix": "--ar 16:9 --v 6 --chaos 50"
    },
    "lofi": {
        "prefix": "lofi aesthetic,",
        "camera": ["static shot", "window view", "desk scene", "cozy interior"],
        "lighting": ["warm lamp light", "sunset through window", "rainy day light", "soft ambient"],
        "mood": ["cozy", "nostalgic", "peaceful", "studious", "melancholic"],
        "suffix": "--ar 16:9 --v 6 --stylize 500"
    },
    "vaporwave": {
        "prefix": "vaporwave aesthetic,",
        "camera": ["glitch effect", "VHS distortion", "retro computer", "mall interior"],
        "lighting": ["pink and cyan neon", "sunset gradient", "CRT glow", "holographic"],
        "mood": ["nostalgic", "surreal", "retrofuturistic", "dreamlike", "ironic"],
        "suffix": "--ar 16:9 --v 6 --stylize 1000"
    },
    "nature": {
        "prefix": "nature footage,",
        "camera": ["aerial drone", "macro lens", "time-lapse", "underwater", "wildlife"],
        "lighting": ["golden hour", "blue hour", "dappled sunlight", "overcast soft"],
        "mood": ["serene", "majestic", "intimate", "wild", "pristine"],
        "suffix": "--ar 16:9 --v 6 --style raw"
    }
}

# Color mood mappings
COLOR_MOODS = {
    "warm": ["orange", "red", "yellow", "amber", "gold"],
    "cool": ["blue", "cyan", "teal", "navy", "ice"],
    "neutral": ["gray", "white", "black", "beige", "silver"],
    "vibrant": ["magenta", "lime", "electric blue", "hot pink", "neon green"],
    "earth": ["brown", "olive", "rust", "terracotta", "forest green"],
    "pastel": ["pink", "lavender", "mint", "peach", "baby blue"]
}

# Scene keywords
SCENE_KEYWORDS = {
    "indoor": ["room", "interior", "inside", "studio", "office", "home"],
    "outdoor": ["outside", "exterior", "open air", "street", "park"],
    "nature": ["forest", "mountain", "ocean", "river", "field", "sky"],
    "urban": ["city", "building", "downtown", "street", "traffic", "skyline"],
    "abstract": ["pattern", "geometric", "fluid", "particles", "light trails"]
}


class VideoPromptGenerator:
    """Generate Midjourney-style prompts from video analysis"""
    
    def __init__(self):
        self.analyses: Dict[str, VideoAnalysis] = {}
        
    def analyze_video(self, file_path: str) -> VideoAnalysis:
        """Analyze a video file and extract visual information"""
        if file_path in self.analyses:
            return self.analyses[file_path]
            
        filename = os.path.basename(file_path)
        analysis = VideoAnalysis(file_path=file_path, filename=filename)
        
        try:
            # Get video metadata using ffprobe
            result = subprocess.run([
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_format", "-show_streams",
                file_path
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                
                # Extract format info
                if "format" in data:
                    analysis.duration_sec = float(data["format"].get("duration", 0))
                    
                # Extract video stream info
                for stream in data.get("streams", []):
                    if stream.get("codec_type") == "video":
                        analysis.width = stream.get("width", 0)
                        analysis.height = stream.get("height", 0)
                        analysis.codec = stream.get("codec_name", "")
                        
                        # Parse fps
                        fps_str = stream.get("r_frame_rate", "30/1")
                        if "/" in fps_str:
                            num, den = fps_str.split("/")
                            analysis.fps = float(num) / float(den) if float(den) > 0 else 30.0
                        else:
                            analysis.fps = float(fps_str)
                        break
            
            # Analyze visual properties using ffmpeg
            self._analyze_visual_properties(file_path, analysis)
            
            # Infer scene type from filename and visual analysis
            self._infer_scene_type(analysis)
            
        except Exception as e:
            print(f"Video analysis error for {filename}: {e}")
            # Set defaults
            analysis.duration_sec = 60
            analysis.width = 1920
            analysis.height = 1080
            analysis.fps = 30.0
            analysis.brightness = "medium"
            analysis.motion_level = "medium"
            analysis.scene_type = "abstract"
            
        self.analyses[file_path] = analysis
        return analysis
    
    def _analyze_visual_properties(self, file_path: str, analysis: VideoAnalysis):
        """Analyze visual properties of the video"""
        try:
            # Get average brightness using ffmpeg
            result = subprocess.run([
                "ffmpeg", "-i", file_path,
                "-vf", "select='eq(n,0)+eq(n,30)+eq(n,60)',showinfo",
                "-f", "null", "-"
            ], capture_output=True, text=True, timeout=30)
            
            # Parse brightness from output (simplified)
            stderr = result.stderr.lower()
            
            # Estimate brightness from common patterns
            if "dark" in stderr or "night" in analysis.filename.lower():
                analysis.brightness = "dark"
                analysis.time_of_day = "night"
            elif "bright" in stderr or "day" in analysis.filename.lower():
                analysis.brightness = "bright"
                analysis.time_of_day = "day"
            else:
                analysis.brightness = "medium"
                analysis.time_of_day = "golden_hour"
            
            # Estimate motion from fps and filename
            if analysis.fps >= 60:
                analysis.motion_level = "fast"
            elif analysis.fps >= 30:
                analysis.motion_level = "medium"
            else:
                analysis.motion_level = "slow"
                
            # Check filename for motion hints
            fname_lower = analysis.filename.lower()
            if any(word in fname_lower for word in ["static", "still", "calm"]):
                analysis.motion_level = "static"
            elif any(word in fname_lower for word in ["fast", "action", "dynamic"]):
                analysis.motion_level = "dynamic"
                
            # Estimate dominant colors from filename or default
            analysis.dominant_colors = self._estimate_colors(analysis)
            
        except Exception as e:
            print(f"Visual analysis error: {e}")
            analysis.brightness = "medium"
            analysis.motion_level = "medium"
            analysis.dominant_colors = ["blue", "purple", "black"]
    
    def _estimate_colors(self, analysis: VideoAnalysis) -> List[str]:
        """Estimate dominant colors based on filename and scene type"""
        fname_lower = analysis.filename.lower()
        colors = []
        
        # Check for color keywords in filename
        for mood, color_list in COLOR_MOODS.items():
            for color in color_list:
                if color in fname_lower:
                    colors.append(color)
        
        # Default colors based on time of day
        if not colors:
            if analysis.time_of_day == "night":
                colors = ["deep blue", "purple", "black", "neon"]
            elif analysis.time_of_day == "golden_hour":
                colors = ["orange", "gold", "warm amber", "soft pink"]
            elif analysis.time_of_day == "dawn":
                colors = ["soft pink", "pale blue", "lavender", "gold"]
            else:
                colors = ["natural", "blue sky", "green", "earth tones"]
        
        return colors[:4]  # Limit to 4 colors
    
    def _infer_scene_type(self, analysis: VideoAnalysis):
        """Infer scene type from filename and analysis"""
        fname_lower = analysis.filename.lower()
        
        for scene_type, keywords in SCENE_KEYWORDS.items():
            if any(kw in fname_lower for kw in keywords):
                analysis.scene_type = scene_type
                break
        
        if not analysis.scene_type:
            # Default based on aspect ratio
            if analysis.width > analysis.height * 2:
                analysis.scene_type = "outdoor"  # Ultra-wide = likely landscape
            else:
                analysis.scene_type = "abstract"  # Default
    
    def generate_prompt(self, analysis: VideoAnalysis, style: str = "cinematic", 
                       custom_subject: str = "", custom_mood: str = "") -> str:
        """Generate a Midjourney-style prompt for the video"""
        
        if style not in MIDJOURNEY_STYLES:
            style = "cinematic"
            
        template = MIDJOURNEY_STYLES[style]
        
        # Build prompt components
        components = []
        
        # Prefix
        components.append(template["prefix"])
        
        # Subject (custom or inferred)
        if custom_subject:
            components.append(custom_subject)
        else:
            # Generate subject from analysis
            subject = self._generate_subject(analysis)
            components.append(subject)
        
        # Camera movement based on motion level
        camera = self._select_camera(analysis, template["camera"])
        components.append(camera)
        
        # Lighting based on brightness and time
        lighting = self._select_lighting(analysis, template["lighting"])
        components.append(lighting)
        
        # Mood (custom or from template)
        if custom_mood:
            components.append(custom_mood)
        else:
            mood = random.choice(template["mood"])
            components.append(f"{mood} mood")
        
        # Colors
        if analysis.dominant_colors:
            color_str = " and ".join(analysis.dominant_colors[:2])
            components.append(f"{color_str} color palette")
        
        # Technical specs
        components.append(f"4K quality")
        
        # Suffix
        prompt = ", ".join(components) + " " + template["suffix"]
        
        return prompt
    
    def _generate_subject(self, analysis: VideoAnalysis) -> str:
        """Generate subject description from analysis"""
        subjects = {
            "indoor": ["cozy interior scene", "modern room", "atmospheric space"],
            "outdoor": ["sweeping landscape", "open vista", "natural environment"],
            "nature": ["pristine wilderness", "organic beauty", "natural wonder"],
            "urban": ["city atmosphere", "urban landscape", "metropolitan scene"],
            "abstract": ["flowing abstract forms", "geometric patterns", "visual poetry"]
        }
        
        scene_subjects = subjects.get(analysis.scene_type, subjects["abstract"])
        return random.choice(scene_subjects)
    
    def _select_camera(self, analysis: VideoAnalysis, options: List[str]) -> str:
        """Select camera movement based on motion level"""
        motion_map = {
            "static": ["static shot", "locked off", "portrait"],
            "slow": ["slow pan", "gentle tracking", "steady"],
            "medium": options[:3],  # Use first 3 options
            "fast": ["quick cuts", "dynamic angle", "action shot"],
            "dynamic": options[-3:]  # Use last 3 options
        }
        
        camera_options = motion_map.get(analysis.motion_level, options)
        # Filter to only include options that exist in the template
        valid_options = [opt for opt in camera_options if any(opt in o or o in opt for o in options)]
        if not valid_options:
            valid_options = options
        
        return random.choice(valid_options)
    
    def _select_lighting(self, analysis: VideoAnalysis, options: List[str]) -> str:
        """Select lighting based on brightness and time"""
        brightness_map = {
            "dark": ["dramatic shadows", "rim lighting", "neon glow", "moonlight"],
            "medium": options,
            "bright": ["natural lighting", "soft diffused", "golden hour", "bright and airy"]
        }
        
        time_lighting = {
            "night": ["neon lights", "moonlight", "city lights", "starlight"],
            "dawn": ["soft pink light", "gentle glow", "misty light"],
            "day": ["natural daylight", "bright sun", "clear light"],
            "golden_hour": ["golden hour", "warm sunset", "magic hour light"],
            "dusk": ["twilight", "blue hour", "fading light"]
        }
        
        # Combine brightness and time-based options
        lighting_options = brightness_map.get(analysis.brightness, options)
        if analysis.time_of_day in time_lighting:
            lighting_options = lighting_options + time_lighting[analysis.time_of_day]
        
        # Filter to valid options
        valid_options = [opt for opt in lighting_options if any(opt in o or o in opt for o in options + lighting_options)]
        if not valid_options:
            valid_options = options
            
        return random.choice(valid_options)
    
    def generate_all_styles(self, analysis: VideoAnalysis, 
                           custom_subject: str = "", 
                           custom_mood: str = "") -> Dict[str, str]:
        """Generate prompts for all available styles"""
        prompts = {}
        for style_name in MIDJOURNEY_STYLES.keys():
            prompts[style_name] = self.generate_prompt(
                analysis, style_name, custom_subject, custom_mood
            )
        
        analysis.prompts = prompts
        return prompts
    
    def generate_meta_ai_prompt(self, analysis: VideoAnalysis, 
                                duration_sec: int = 5,
                                custom_description: str = "") -> str:
        """Generate prompt optimized for meta.ai video generation"""
        
        # Meta.ai prefers clear, descriptive prompts
        components = []
        
        # Duration hint
        if duration_sec <= 4:
            components.append("short clip")
        elif duration_sec <= 8:
            components.append("medium length video")
        else:
            components.append("extended sequence")
        
        # Main description
        if custom_description:
            components.append(custom_description)
        else:
            # Generate from analysis
            scene_desc = self._generate_subject(analysis)
            components.append(scene_desc)
        
        # Visual style
        if analysis.brightness == "dark":
            components.append("moody and atmospheric")
        elif analysis.brightness == "bright":
            components.append("bright and vibrant")
        else:
            components.append("balanced lighting")
        
        # Motion
        motion_desc = {
            "static": "minimal movement, contemplative",
            "slow": "slow, graceful motion",
            "medium": "natural movement",
            "fast": "energetic, dynamic motion",
            "dynamic": "high energy, rapid movement"
        }
        components.append(motion_desc.get(analysis.motion_level, "natural movement"))
        
        # Colors
        if analysis.dominant_colors:
            components.append(f"featuring {', '.join(analysis.dominant_colors[:2])} tones")
        
        # Quality
        components.append("high quality, smooth motion, professional look")
        
        return ", ".join(components)
    
    def batch_analyze(self, file_paths: List[str], 
                     progress_callback=None) -> List[VideoAnalysis]:
        """Analyze multiple videos with progress callback"""
        results = []
        total = len(file_paths)
        
        for i, path in enumerate(file_paths):
            analysis = self.analyze_video(path)
            results.append(analysis)
            
            if progress_callback:
                progress_callback(i + 1, total, analysis.filename)
                
        return results


# ==================== Export Functions ====================
def export_prompts_to_file(analyses: List[VideoAnalysis], output_path: str, 
                          include_all_styles: bool = True):
    """Export generated prompts to a text file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Video Prompt Generator - Midjourney Style Prompts\n")
        f.write("# Generated by LongPlay Studio V4.26\n\n")
        
        for analysis in analyses:
            f.write(f"## {analysis.filename}\n")
            f.write(f"Duration: {analysis.duration_sec:.1f}s | ")
            f.write(f"Resolution: {analysis.width}x{analysis.height} | ")
            f.write(f"FPS: {analysis.fps:.1f}\n\n")
            
            if analysis.prompts:
                for style, prompt in analysis.prompts.items():
                    f.write(f"### {style.title()}\n")
                    f.write(f"```\n{prompt}\n```\n\n")
            
            f.write("---\n\n")


# ==================== Test ====================
if __name__ == "__main__":
    # Test with sample video
    generator = VideoPromptGenerator()
    
    # Create mock analysis for testing
    test_analysis = VideoAnalysis(
        file_path="/test/video.mp4",
        filename="CityLightsFade.mp4",
        duration_sec=180,
        width=1920,
        height=1080,
        fps=30.0,
        brightness="dark",
        motion_level="medium",
        scene_type="urban",
        time_of_day="night",
        dominant_colors=["neon blue", "purple", "orange"]
    )
    
    print("=== Video Prompt Generator Test ===\n")
    
    # Generate all style prompts
    prompts = generator.generate_all_styles(test_analysis)
    
    for style, prompt in prompts.items():
        print(f"[{style.upper()}]")
        print(prompt)
        print()
    
    # Generate meta.ai prompt
    print("[META.AI]")
    meta_prompt = generator.generate_meta_ai_prompt(test_analysis, duration_sec=5)
    print(meta_prompt)
