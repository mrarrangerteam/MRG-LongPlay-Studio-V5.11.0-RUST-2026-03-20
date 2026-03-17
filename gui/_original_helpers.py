"""
Original top-level helper functions extracted from gui.py.

These utility functions (ffmpeg path setup, hardware detection, quality presets,
smart temp directory, etc.) are used by the main window and export logic.
"""

import os
import platform
import re
import subprocess
import tempfile


# ==================== FFmpeg Path Setup for macOS ====================

def setup_ffmpeg_path():
    """Add Homebrew paths to PATH for macOS .app bundles"""
    if platform.system() == "Darwin":
        homebrew_paths = [
            "/opt/homebrew/bin",  # Apple Silicon
            "/usr/local/bin",     # Intel Mac
        ]
        current_path = os.environ.get("PATH", "")
        paths_to_add = []
        for path in homebrew_paths:
            if os.path.exists(path) and path not in current_path:
                paths_to_add.append(path)
        if paths_to_add:
            os.environ["PATH"] = ":".join(paths_to_add) + ":" + current_path
            print(f"Added to PATH: {', '.join(paths_to_add)}")

# Run IMMEDIATELY on import
setup_ffmpeg_path()


def _natural_sort_key(filepath):
    """Natural sort key: sort '2.Song' before '10.Song' (numeric-aware)"""
    basename = os.path.basename(filepath)
    parts = re.split(r'(\d+)', basename)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


# ==================== Hardware Acceleration Detection ====================

def detect_hw_encoder():
    """Detect best available hardware encoder for the system"""
    system = platform.system()

    if system == "Darwin":
        try:
            result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"],
                capture_output=True, text=True, timeout=5
            )
            if "h264_videotoolbox" in result.stdout:
                print("Hardware Acceleration: VideoToolbox (Apple Silicon/Intel Mac)")
                return "h264_videotoolbox"
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass

    elif system == "Windows":
        try:
            result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"],
                capture_output=True, text=True, timeout=5
            )
            if "h264_nvenc" in result.stdout:
                return "h264_nvenc"
            if "h264_amf" in result.stdout:
                return "h264_amf"
            if "h264_qsv" in result.stdout:
                return "h264_qsv"
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass

    elif system == "Linux":
        try:
            result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"],
                capture_output=True, text=True, timeout=5
            )
            if "h264_vaapi" in result.stdout:
                return "h264_vaapi"
            if "h264_nvenc" in result.stdout:
                return "h264_nvenc"
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass

    print("No Hardware Acceleration - using CPU (libx264)")
    return "libx264"


# Global encoder setting
HW_ENCODER = detect_hw_encoder()


# ==================== Smart Temp Directory Selection ====================

def get_smart_temp_dir(min_free_gb: float = 5.0) -> str:
    """Smart temp directory selection - chooses location with most free space."""
    import shutil as sh

    min_free_bytes = min_free_gb * 1024 * 1024 * 1024
    candidates = []

    env_tmpdir = os.environ.get('TMPDIR') or os.environ.get('TEMP') or os.environ.get('TMP')
    if env_tmpdir and os.path.exists(env_tmpdir):
        try:
            usage = sh.disk_usage(env_tmpdir)
            if usage.free >= min_free_bytes:
                print(f"[SMART TEMP] Using TMPDIR: {env_tmpdir} ({usage.free / (1024**3):.1f}GB free)")
                return tempfile.mkdtemp(dir=env_tmpdir)
            candidates.append((env_tmpdir, usage.free))
        except OSError as e:
            print(f"[SMART TEMP] Warning: Cannot check TMPDIR: {e}")

    system = platform.system()

    if system == "Darwin":
        volumes_path = "/Volumes"
        if os.path.exists(volumes_path):
            for volume in os.listdir(volumes_path):
                vol_path = os.path.join(volumes_path, volume)
                if os.path.isdir(vol_path) and volume != "Macintosh HD":
                    try:
                        usage = sh.disk_usage(vol_path)
                        if usage.free >= min_free_bytes:
                            temp_base = os.path.join(vol_path, "longplay_temp")
                            os.makedirs(temp_base, exist_ok=True)
                            candidates.append((temp_base, usage.free))
                    except OSError:
                        pass

        home_temp = os.path.expanduser("~/Library/Caches/LongPlayStudio")
        try:
            os.makedirs(home_temp, exist_ok=True)
            usage = sh.disk_usage(home_temp)
            candidates.append((home_temp, usage.free))
        except OSError:
            pass

    elif system == "Windows":
        import string
        for letter in string.ascii_uppercase:
            drive = f"{letter}:\\"
            if os.path.exists(drive):
                try:
                    usage = sh.disk_usage(drive)
                    if usage.free >= min_free_bytes:
                        temp_base = os.path.join(drive, "LongPlayTemp")
                        os.makedirs(temp_base, exist_ok=True)
                        candidates.append((temp_base, usage.free))
                except OSError:
                    pass

    else:  # Linux
        for mount in ["/mnt", "/media", os.path.expanduser("~")]:
            if os.path.exists(mount):
                try:
                    usage = sh.disk_usage(mount)
                    if usage.free >= min_free_bytes:
                        temp_base = os.path.join(mount, "longplay_temp")
                        os.makedirs(temp_base, exist_ok=True)
                        candidates.append((temp_base, usage.free))
                except OSError:
                    pass

    candidates.sort(key=lambda x: x[1], reverse=True)

    if candidates:
        best_path, best_free = candidates[0]
        print(f"[SMART TEMP] Selected: {best_path} ({best_free / (1024**3):.1f}GB free)")
        return tempfile.mkdtemp(dir=best_path)

    default_temp = tempfile.mkdtemp()
    print(f"[SMART TEMP] Fallback to system default: {default_temp}")
    return default_temp


def get_hwaccel_input_params():
    """Get FFmpeg hardware acceleration input parameters for faster decoding"""
    system = platform.system()

    if system == "Darwin" and HW_ENCODER == "h264_videotoolbox":
        return ["-hwaccel", "videotoolbox"]
    elif system == "Windows":
        if HW_ENCODER == "h264_nvenc":
            return ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        elif HW_ENCODER == "h264_qsv":
            return ["-hwaccel", "qsv"]
    elif system == "Linux":
        if HW_ENCODER == "h264_vaapi":
            return ["-hwaccel", "vaapi", "-hwaccel_device", "/dev/dri/renderD128"]
        elif HW_ENCODER == "h264_nvenc":
            return ["-hwaccel", "cuda"]

    return []


# Get hardware input params once at startup
HW_INPUT_PARAMS = get_hwaccel_input_params()
if HW_INPUT_PARAMS:
    print(f"Hardware Decoding: {HW_INPUT_PARAMS[1]}")


# ==================== Quality Mode Settings ====================

QUALITY_PRESETS = {
    "fast": {
        "description": "Fast - Quick preview, smaller file",
        "x264_preset": "ultrafast",
        "x264_crf": "28",
        "videotoolbox_q": "45",
        "nvenc_preset": "p1",
        "nvenc_cq": "28",
    },
    "balanced": {
        "description": "Balanced - Good for YouTube",
        "x264_preset": "fast",
        "x264_crf": "23",
        "videotoolbox_q": "60",
        "nvenc_preset": "p4",
        "nvenc_cq": "23",
    },
    "quality": {
        "description": "Quality - Best quality, slower",
        "x264_preset": "slow",
        "x264_crf": "18",
        "videotoolbox_q": "75",
        "nvenc_preset": "p7",
        "nvenc_cq": "18",
    },
}

CURRENT_QUALITY_MODE = "balanced"


def set_quality_mode(mode: str):
    """Set the global quality mode"""
    global CURRENT_QUALITY_MODE
    if mode in QUALITY_PRESETS:
        CURRENT_QUALITY_MODE = mode
        print(f"Quality mode set to: {QUALITY_PRESETS[mode]['description']}")


def get_quality_mode() -> str:
    """Get the current quality mode"""
    return CURRENT_QUALITY_MODE


def get_encoder_params(force_software=False, quality_mode=None):
    """Get FFmpeg encoder parameters based on detected hardware and quality mode"""
    mode = quality_mode or CURRENT_QUALITY_MODE
    preset = QUALITY_PRESETS.get(mode, QUALITY_PRESETS["balanced"])

    if force_software:
        return [
            "-c:v", "libx264",
            "-crf", preset["x264_crf"],
            "-preset", preset["x264_preset"],
            "-pix_fmt", "yuv420p",
        ]
    if HW_ENCODER == "h264_videotoolbox":
        return [
            "-c:v", "h264_videotoolbox",
            "-q:v", preset["videotoolbox_q"],
            "-pix_fmt", "yuv420p",
        ]
    elif HW_ENCODER == "h264_nvenc":
        return [
            "-c:v", "h264_nvenc",
            "-preset", preset["nvenc_preset"],
            "-cq", preset["nvenc_cq"],
            "-pix_fmt", "yuv420p",
        ]
    elif HW_ENCODER == "h264_amf":
        quality_setting = "speed" if mode == "fast" else ("balanced" if mode == "balanced" else "quality")
        return [
            "-c:v", "h264_amf",
            "-quality", quality_setting,
            "-rc", "cqp",
            "-qp", preset["x264_crf"],
            "-pix_fmt", "yuv420p",
        ]
    elif HW_ENCODER == "h264_qsv":
        qsv_preset = "veryfast" if mode == "fast" else ("fast" if mode == "balanced" else "slow")
        return [
            "-c:v", "h264_qsv",
            "-preset", qsv_preset,
            "-global_quality", preset["x264_crf"],
            "-pix_fmt", "nv12",
        ]
    elif HW_ENCODER == "h264_vaapi":
        return [
            "-c:v", "h264_vaapi",
            "-qp", preset["x264_crf"],
        ]
    else:
        return [
            "-c:v", "libx264",
            "-crf", preset["x264_crf"],
            "-preset", preset["x264_preset"],
            "-pix_fmt", "yuv420p",
        ]


def run_ffmpeg_with_progress(cmd: list, duration: float, progress_callback=None):
    """Run FFmpeg command with real-time progress tracking"""
    cmd_with_progress = cmd.copy()
    idx = 1
    cmd_with_progress.insert(idx, "-progress")
    cmd_with_progress.insert(idx + 1, "pipe:1")
    cmd_with_progress.insert(idx + 2, "-stats_period")
    cmd_with_progress.insert(idx + 3, "0.5")

    process = subprocess.Popen(
        cmd_with_progress,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )

    time_pattern = re.compile(r'out_time_ms=(\d+)')

    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break

        match = time_pattern.search(line)
        if match and duration > 0:
            current_ms = int(match.group(1))
            current_sec = current_ms / 1000000
            progress = min(100, int((current_sec / duration) * 100))
            if progress_callback:
                progress_callback(progress)

    process.wait()
    if process.returncode != 0:
        stderr = process.stderr.read()
        raise subprocess.CalledProcessError(process.returncode, cmd, stderr=stderr)
