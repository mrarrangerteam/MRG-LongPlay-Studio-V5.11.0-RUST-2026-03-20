"""
Video preview card and thread stub.

Classes:
    VideoThread      — Thread stub for backward compat (no-op)
    VideoPreviewCard — QTimer-based video frame renderer with GIF overlay
"""

import os
import re
import subprocess
import shutil
import tempfile
from typing import List, Optional, Dict

from gui.utils.compat import (
    QFrame, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QSizePolicy, Qt, QTimer,
    QPixmap, QImage, QApplication, QThread, pyqtSignal, QFont,
)
from gui.styles import Colors
from gui.audio_player import MediaFile
from gui.video.detached import DetachedVideoWindow

# OpenCV availability
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class VideoThread(QThread):
    """Thread stub for backward compatibility - frame rendering now handled by QTimer in VideoPreviewCard"""
    change_pixmap_signal = pyqtSignal(object)  # np.ndarray
    position_changed = pyqtSignal(int)  # position in ms
    duration_changed = pyqtSignal(int)  # duration in ms

    def __init__(self, video_path: str = None):
        super().__init__()
        self.video_path = video_path
        self._run_flag = False
        self._pause_flag = False
        self._speed = 1.0
        self._seek_position = -1
        self.duration_ms = 0
        self.current_position_ms = 0

    def set_video(self, path: str):
        self.video_path = path
        self._seek_position = 0

    def set_speed(self, speed: float):
        self._speed = speed

    def seek(self, position_ms: int):
        self._seek_position = position_ms

    def run(self):
        return

    def pause(self):
        self._pause_flag = True

    def resume(self):
        self._pause_flag = False

    def stop(self):
        self._run_flag = False
        self._pause_flag = False
        self.wait()


class VideoPreviewCard(QFrame):
    """Video/GIF preview with QTimer-based frame rendering - no threading overhead"""

    speed_changed = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(280)
        self.setStyleSheet(f"""
            QFrame {{
                background: {Colors.BG_CARD};
                border-radius: 12px;
                border: 1px solid {Colors.BORDER};
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)

        # Header with play button
        header = QHBoxLayout()
        title = QLabel("\U0001f3ac Video Preview")
        title.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; font-size: 13px; font-weight: bold;")
        header.addWidget(title)
        header.addStretch()

        self.play_btn = QPushButton("\u25b6")
        self.play_btn.setFixedSize(32, 32)
        self.play_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.ACCENT};
                color: white;
                border: none;
                border-radius: 16px;
                font-size: 14px;
            }}
            QPushButton:hover {{
                background: {Colors.ACCENT_DIM};
            }}
        """)
        self.play_btn.clicked.connect(self._toggle_play)
        header.addWidget(self.play_btn)

        # Speed controls
        self.speed_buttons = {}
        self.current_speed = 1.0
        for speed_text, speed_value in [("1x", 1.0), ("1.5x", 1.5), ("2x", 2.0)]:
            btn = QPushButton(speed_text)
            btn.setFixedSize(32, 24)
            btn.setProperty("speed_value", speed_value)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: {Colors.BG_TERTIARY if speed_value != 1.0 else Colors.ACCENT};
                    color: {Colors.TEXT_SECONDARY if speed_value != 1.0 else 'white'};
                    border: none;
                    border-radius: 4px;
                    font-size: 10px;
                }}
                QPushButton:hover {{
                    background: {Colors.BORDER};
                }}
            """)
            btn.clicked.connect(lambda checked, sv=speed_value: self._set_playback_speed(sv))
            header.addWidget(btn)
            self.speed_buttons[speed_value] = btn

        # Pop Out button
        self.popout_btn = QPushButton("Pop Out \u2b1c")
        self.popout_btn.setFixedSize(80, 32)
        self.popout_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_SECONDARY};
                border: none;
                border-radius: 4px;
                font-size: 10px;
            }}
            QPushButton:hover {{
                background: {Colors.BORDER};
            }}
        """)
        self.popout_btn.clicked.connect(self._toggle_detached_window)
        header.addWidget(self.popout_btn)

        layout.addLayout(header)

        # Realtime display label
        self.realtime_label = QLabel("00:00 / 00:00")
        self.realtime_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.realtime_label.setStyleSheet(f"""
            QLabel {{
                color: {Colors.ACCENT};
                font-size: 14px;
                font-weight: bold;
                font-family: 'Menlo', 'Courier New', monospace;
                background: {Colors.BG_TERTIARY};
                border-radius: 6px;
                padding: 4px 8px;
            }}
        """)
        layout.addWidget(self.realtime_label)

        # Container for video preview
        self.preview_container = QWidget()
        self.preview_container.setMinimumHeight(180)
        self.preview_container.setStyleSheet("background: #000000; border-radius: 8px;")
        self.preview_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        container_layout = QGridLayout(self.preview_container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)

        # Video display label
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background: #000000;")
        self.video_label.setMinimumSize(320, 180)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_label.setScaledContents(False)
        container_layout.addWidget(self.video_label, 0, 0)

        # GIF label OVERLAY
        self.gif_label = QLabel()
        self.gif_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.gif_label.setStyleSheet("background: transparent;")
        self.gif_label.setScaledContents(False)
        self.gif_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        container_layout.addWidget(self.gif_label, 0, 0)
        self.gif_label.raise_()
        self.gif_label.hide()

        layout.addWidget(self.preview_container, 1)

        # Caption
        self.caption = QLabel("No video loaded")
        self.caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.caption.setStyleSheet(f"color: {Colors.TEXT_TERTIARY}; font-size: 10px;")
        self.caption.setWordWrap(True)
        layout.addWidget(self.caption)

        # Video thread stub
        self.video_thread = None

        # QTimer for frame rendering
        self._render_timer = QTimer()
        self._render_timer.timeout.connect(self._render_frame)
        self._render_timer.setInterval(83)  # ~12fps

        # PRE-CACHED FRAMES
        self._frame_cache: Dict[int, list] = {}
        self._frame_fps: Dict[int, float] = {}
        self._video_durations: Dict[int, int] = {}
        self._current_audio_pos_ms = 0
        self._force_render = False
        self._last_rendered_frame = (-1, -1)

        # State
        self.video_files: List[MediaFile] = []
        self.current_type = "video"
        self.is_playing = False
        self.gif_movie = None
        self._gif_frames: list = []
        self._gif_delays: list = []
        self._gif_frame_idx: int = 0
        self._gif_timer: Optional[QTimer] = None
        self.use_opencv = CV2_AVAILABLE

        # Detached window
        self.detached_window: Optional[DetachedVideoWindow] = None

    def _preload_video_frames(self, video_idx: int):
        """Pre-load ALL frames from a short video into memory as QImage."""
        if video_idx in self._frame_cache:
            return

        if not (0 <= video_idx < len(self.video_files)):
            print(f"[VIDEO PREVIEW] video_idx {video_idx} out of range (have {len(self.video_files)} videos)")
            return

        video = self.video_files[video_idx]
        print(f"[VIDEO PREVIEW] Preloading video {video_idx}: {video.path}")

        if CV2_AVAILABLE:
            frames = self._preload_via_cv2(video_idx, video.path)
            if frames:
                self._frame_cache[video_idx] = frames
                print(f"[VIDEO PREVIEW] cv2 cached video {video_idx}: {len(frames)} frames")
                return

        print(f"[VIDEO PREVIEW] cv2 failed, trying ffmpeg fallback...")
        frames = self._preload_via_ffmpeg(video_idx, video.path)
        if frames:
            self._frame_cache[video_idx] = frames
            print(f"[VIDEO PREVIEW] ffmpeg cached video {video_idx}: {len(frames)} frames")
        else:
            print(f"[VIDEO PREVIEW] Both cv2 and ffmpeg failed! Creating test pattern...")
            self._create_test_frames(video_idx)

    def _preload_via_cv2(self, video_idx, path):
        try:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                print(f"[VIDEO PREVIEW] cv2 cannot open: {path}")
                return None

            fps = cap.get(cv2.CAP_PROP_FPS) or 24
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_ms = int((total_frames / fps) * 1000) if fps > 0 else 5000

            self._video_durations[video_idx] = duration_ms
            self._frame_fps[video_idx] = fps
            print(f"[VIDEO PREVIEW] cv2 opened: fps={fps}, total={total_frames}, duration={duration_ms}ms")

            target_w, target_h = 640, 360
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                small = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
                if not qimg.isNull():
                    frames.append(qimg)

            cap.release()
            if frames:
                print(f"[VIDEO PREVIEW] cv2: {len(frames)} frames, first={frames[0].width()}x{frames[0].height()}")
            else:
                print(f"[VIDEO PREVIEW] cv2: 0 frames read (codec issue?)")
            return frames if frames else None

        except Exception as e:
            print(f"[VIDEO PREVIEW] cv2 error: {e}")
            return None

    def _preload_via_ffmpeg(self, video_idx, path):
        import glob as glob_mod
        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries", "stream=r_frame_rate,duration,nb_frames",
                 "-select_streams", "v:0", "-of", "csv=p=0", path],
                capture_output=True, text=True, timeout=10
            )
            info = probe.stdout.strip().split(',') if probe.stdout.strip() else []

            if len(info) >= 1 and '/' in info[0]:
                num, den = info[0].split('/')
                fps = float(num) / float(den) if float(den) > 0 else 24
            else:
                fps = 24

            tmpdir = tempfile.mkdtemp(prefix="longplay_frames_")
            subprocess.run(
                ["ffmpeg", "-i", path, "-vf", "scale=640:360", "-q:v", "3",
                 "-f", "image2", f"{tmpdir}/frame_%04d.jpg"],
                capture_output=True, text=True, timeout=30
            )

            frame_files = sorted(glob_mod.glob(f"{tmpdir}/frame_*.jpg"))
            frames = []
            for fpath in frame_files:
                qimg = QImage(fpath)
                if not qimg.isNull():
                    frames.append(qimg.copy())

            shutil.rmtree(tmpdir, ignore_errors=True)

            if frames:
                duration_ms = int((len(frames) / fps) * 1000)
                self._video_durations[video_idx] = duration_ms
                self._frame_fps[video_idx] = fps
                print(f"[VIDEO PREVIEW] ffmpeg: {len(frames)} frames, fps={fps}")
            return frames if frames else None

        except Exception as e:
            print(f"[VIDEO PREVIEW] ffmpeg error: {e}")
            return None

    def _create_test_frames(self, video_idx):
        from gui.utils.compat import QPainter as _QPainter, QFont as _QFont
        target_w, target_h = 640, 360
        frames = []
        from gui.utils.compat import QColor as _QC
        test_colors = [_QC(255, 100, 50), _QC(50, 200, 100), _QC(50, 100, 255)]
        for i in range(24):
            img = QImage(target_w, target_h, QImage.Format.Format_RGB888)
            img.fill(test_colors[i % len(test_colors)])
            p = _QPainter(img)
            p.setPen(_QC(255, 255, 255))
            f = _QFont("Arial", 24)
            p.setFont(f)
            p.drawText(img.rect(), Qt.AlignmentFlag.AlignCenter,
                       f"TEST FRAME {i}\nVideo {video_idx}\nDisplay OK!")
            p.end()
            frames.append(img)
        self._frame_cache[video_idx] = frames
        self._video_durations[video_idx] = 1000
        self._frame_fps[video_idx] = 24
        print(f"[VIDEO PREVIEW] Created {len(frames)} test frames for video {video_idx}")

    _render_debug_count = 0

    def _render_frame(self):
        if not self.is_playing and not self._force_render:
            return
        if not self.video_files:
            return

        pos = self._current_audio_pos_ms
        video_idx = self._get_video_for_position(pos)

        if video_idx not in self._frame_cache:
            self._preload_video_frames(video_idx)

        frames = self._frame_cache.get(video_idx)
        if not frames:
            if VideoPreviewCard._render_debug_count < 5:
                print(f"[VIDEO PREVIEW] No frames in cache for video {video_idx}")
                VideoPreviewCard._render_debug_count += 1
            self._force_render = False
            return

        duration_ms = self._video_durations.get(video_idx, 5000)
        video_pos_ms = pos % duration_ms if duration_ms > 0 else 0
        fps = self._frame_fps.get(video_idx, 24)
        frame_idx = int((video_pos_ms / 1000.0) * fps)
        frame_idx = max(0, min(frame_idx, len(frames) - 1))

        cache_key = (video_idx, frame_idx)
        if cache_key == self._last_rendered_frame and not self._force_render:
            return
        self._last_rendered_frame = cache_key

        if VideoPreviewCard._render_debug_count < 10:
            print(f"[VIDEO PREVIEW] Render: video={video_idx}, frame={frame_idx}/{len(frames)}, pos={pos}ms")
            VideoPreviewCard._render_debug_count += 1

        qimg = frames[frame_idx]
        self._display_qimage(qimg)
        self._force_render = False

    def _display_qimage(self, qimage):
        if qimage is None or qimage.isNull():
            print(f"[VIDEO PREVIEW] Null QImage, cannot display")
            return

        try:
            pixmap = QPixmap.fromImage(qimage)
            if pixmap.isNull():
                print(f"[VIDEO PREVIEW] QPixmap.fromImage returned null!")
                return

            label_size = self.video_label.size()
            if label_size.width() > 0 and label_size.height() > 0:
                scaled = pixmap.scaled(
                    label_size.width(),
                    label_size.height(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.video_label.setPixmap(scaled)
            else:
                self.video_label.setPixmap(pixmap)

            if self.detached_window and self.detached_window.isVisible():
                det_size = self.detached_window.video_label.size()
                if det_size.width() > 0 and det_size.height() > 0:
                    det_scaled = pixmap.scaled(
                        det_size.width(),
                        det_size.height(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    )
                    self.detached_window.video_label.setPixmap(det_scaled)
                else:
                    self.detached_window.video_label.setPixmap(pixmap)
        except Exception as e:
            import traceback
            print(f"[VIDEO PREVIEW] Display error: {e}")
            traceback.print_exc()

    def setVideos(self, videos: List[MediaFile]):
        self.video_files = videos
        if videos:
            video = videos[0]
            print(f"[VIDEO PREVIEW] ========================================")
            print(f"[VIDEO PREVIEW] Loading video: {video.path}")
            print(f"[VIDEO PREVIEW] Total videos loaded: {len(videos)}")
            print(f"[VIDEO PREVIEW] File exists: {os.path.exists(video.path)}")
            print(f"[VIDEO PREVIEW] Using OpenCV: {self.use_opencv}")

            if not os.path.exists(video.path):
                print(f"[VIDEO PREVIEW] ERROR: File not found: {video.path}")
                self.caption.setText(f"File not found: {video.name}")
                return

            if len(videos) > 1:
                self.caption.setText(f"\U0001f3ac {os.path.basename(video.path)} (+{len(videos)-1} more)")
            else:
                self.caption.setText(os.path.basename(video.path))
            self.current_type = "video"

            self.gif_label.hide()
            VideoPreviewCard._render_debug_count = 0

            for i, v in enumerate(videos):
                self.caption.setText(f"Loading video {i+1}/{len(videos)}...")
                QApplication.processEvents()
                self._preload_video_frames(i)

            total_cached = sum(len(f) for f in self._frame_cache.values())
            if total_cached > 0:
                self.caption.setText(f"\U0001f3ac {len(videos)} videos ({total_cached} frames)")
            else:
                self.caption.setText("Cannot load video frames")
            print(f"[VIDEO PREVIEW] All videos cached: {total_cached} total frames")

            self._force_render = True
            self._current_audio_pos_ms = 0
            self._render_frame()

            self.is_playing = True
            self.play_btn.setText("\u23f8")
            QTimer.singleShot(100, self._render_timer.start)
            print(f"[VIDEO PREVIEW] Frame rendering scheduled!")

    def setGIF(self, gif_path: str):
        import glob as _glob
        print(f"[GIF setGIF] START: {gif_path}")
        try:
            print(f"[GIF setGIF] Step 1: Stopping existing animation...")
            if self._gif_timer:
                self._gif_timer.stop()
                self._gif_timer = None
            self._gif_frames = []
            self._gif_delays = []
            self._gif_frame_idx = 0
            self.gif_movie = None
            print(f"[GIF setGIF] Step 1: OK")

            if not os.path.exists(gif_path):
                print(f"[GIF setGIF] File not found: {gif_path}")
                return

            print(f"[GIF setGIF] Step 2: Creating temp dir...")
            tmpdir = tempfile.mkdtemp(prefix="longplay_gif_")
            print(f"[GIF setGIF] Step 2: tmpdir={tmpdir}")
            try:
                label_size = self.video_label.size()
                scale_w = max(320, label_size.width())
                scale_h = max(180, label_size.height())
                print(f"[GIF setGIF] Step 3: Extracting frames via ffmpeg ({scale_w}x{scale_h})...")

                out_path = os.path.join(tmpdir, "frame_%04d.png")
                cmd = [
                    "ffmpeg", "-i", gif_path,
                    "-vf", f"scale={scale_w}:{scale_h}:force_original_aspect_ratio=decrease",
                    "-q:v", "2", "-v", "quiet",
                    "-f", "image2", out_path
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                print(f"[GIF setGIF] Step 3: ffmpeg returned {result.returncode}")

                if result.returncode != 0:
                    print(f"[GIF setGIF] Step 3b: ffmpeg stderr: {result.stderr[:200]}")
                    cmd2 = ["ffmpeg", "-i", gif_path, "-v", "quiet",
                            "-f", "image2", os.path.join(tmpdir, "f_%04d.jpg")]
                    subprocess.run(cmd2, capture_output=True, text=True, timeout=30)

                print(f"[GIF setGIF] Step 4: Loading frames from disk...")
                frame_files = sorted(_glob.glob(os.path.join(tmpdir, "frame_*.png")))
                if not frame_files:
                    frame_files = sorted(_glob.glob(os.path.join(tmpdir, "f_*.jpg")))
                print(f"[GIF setGIF] Step 4: Found {len(frame_files)} frame files")

                for fpath in frame_files[:200]:
                    qimg = QImage(fpath)
                    if not qimg.isNull():
                        self._gif_frames.append(qimg.copy())

                print(f"[GIF setGIF] Step 4: Loaded {len(self._gif_frames)} QImage frames")

            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

            if not self._gif_frames:
                print(f"[GIF setGIF] No frames extracted, skipping display")
                return

            print(f"[GIF setGIF] Step 5: Displaying first frame...")
            pixmap = QPixmap.fromImage(self._gif_frames[0])
            if pixmap.isNull():
                print(f"[GIF setGIF] Step 5: WARNING - pixmap is null!")
                return
            self.gif_label.setPixmap(pixmap)
            self.gif_label.show()
            print(f"[GIF setGIF] Step 5: OK")

            print(f"[GIF setGIF] Step 6: Starting animation timer...")
            self._gif_timer = QTimer(self)
            self._gif_timer.timeout.connect(self._cycle_gif_frame)
            self._gif_timer.start(83)

            print(f"[GIF setGIF] DONE: {len(self._gif_frames)} frames loaded")

            if self.video_files:
                self.caption.setText(f"Video + GIF: {os.path.basename(gif_path)}")
            else:
                self.caption.setText(os.path.basename(gif_path))

            if not self.video_files:
                self.current_type = "gif"
        except Exception as e:
            import traceback
            print(f"[GIF setGIF] ERROR: {e}")
            traceback.print_exc()

    def _cycle_gif_frame(self):
        try:
            if not self._gif_frames:
                return
            self._gif_frame_idx = (self._gif_frame_idx + 1) % len(self._gif_frames)
            pixmap = QPixmap.fromImage(self._gif_frames[self._gif_frame_idx])
            self.gif_label.setPixmap(pixmap)
        except Exception:
            pass  # Silently skip frame on error

    def _toggle_play(self):
        if self.is_playing:
            self.pause()
        else:
            self.play()

    def play(self):
        self.is_playing = True
        self.play_btn.setText("\u23f8")

        if self.current_type == "gif" and self._gif_frames:
            if self._gif_timer:
                self._gif_timer.start(83)
        elif self.use_opencv:
            self._render_timer.start()
            print(f"[VIDEO PREVIEW] Playing video...")

    def pause(self):
        self.is_playing = False
        self.play_btn.setText("\u25b6")

        if self.current_type == "gif" and self._gif_timer:
            self._gif_timer.stop()
        elif self.use_opencv:
            self._render_timer.stop()
            print(f"[VIDEO PREVIEW] Paused video")

    def seek(self, position_ms: int):
        self._current_audio_pos_ms = position_ms
        self._force_render = True
        self._render_frame()
        self._update_realtime_display(position_ms)

    def set_audio_context(self, audio_files, crossfade_sec=5):
        self._audio_files = audio_files
        self._crossfade_sec = crossfade_sec

    def _get_video_for_position(self, position_ms: int):
        audio_files = getattr(self, '_audio_files', None)
        if not audio_files or not self.video_files:
            return 0

        crossfade_sec = getattr(self, '_crossfade_sec', 5)
        position_sec = position_ms / 1000.0

        cumulative = 0
        current_track_idx = 0
        for i, af in enumerate(audio_files):
            if i == 0:
                track_end = cumulative + af.duration
            else:
                track_end = cumulative + max(0, af.duration - crossfade_sec)

            if position_sec < track_end or i == len(audio_files) - 1:
                current_track_idx = i
                break
            cumulative = track_end

        all_default = all(getattr(af, 'video_assignment', 0) == 0 for af in audio_files)
        num_videos = len(self.video_files)

        if all_default and num_videos > 1:
            tracks_per_video = max(1, len(audio_files) // num_videos)
            video_idx = min(current_track_idx // tracks_per_video, num_videos - 1)
            return video_idx
        else:
            video_assignment = getattr(audio_files[current_track_idx], 'video_assignment', 0)
            if video_assignment < num_videos:
                return video_assignment
            return current_track_idx % num_videos

    def stop_playback(self):
        self._render_timer.stop()
        self.is_playing = False
        self.play_btn.setText("\u25b6")

    def _set_playback_speed(self, speed: float):
        self.current_speed = speed
        self.speed_changed.emit(speed)
        print(f"[VIDEO PREVIEW] Speed set to {speed}x")

        for speed_val, btn in self.speed_buttons.items():
            if speed_val == speed:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background: {Colors.ACCENT};
                        color: white;
                        border: none;
                        border-radius: 4px;
                        font-size: 10px;
                    }}
                    QPushButton:hover {{
                        background: {Colors.ACCENT_DIM};
                    }}
                """)
            else:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background: {Colors.BG_TERTIARY};
                        color: {Colors.TEXT_SECONDARY};
                        border: none;
                        border-radius: 4px;
                        font-size: 10px;
                    }}
                    QPushButton:hover {{
                        background: {Colors.BORDER};
                    }}
                """)

    def _toggle_detached_window(self):
        if self.detached_window is None or not self.detached_window.isVisible():
            self._open_detached_window()
        else:
            self._close_detached_window()

    def _open_detached_window(self):
        self.detached_window = DetachedVideoWindow()
        self.detached_window.dock_btn.clicked.connect(self._close_detached_window)
        self.detached_window.closed.connect(self._on_detached_window_closed)
        self.detached_window.show()

        self.popout_btn.setText("Pop In \u2b1c")
        self.popout_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.ACCENT};
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 10px;
            }}
            QPushButton:hover {{
                background: {Colors.ACCENT_DIM};
            }}
        """)

        self._force_render = True
        self._render_frame()

    def _close_detached_window(self):
        if self.detached_window:
            self.detached_window.close()
            self.detached_window = None

        self.popout_btn.setText("Pop Out \u2b1c")
        self.popout_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_SECONDARY};
                border: none;
                border-radius: 4px;
                font-size: 10px;
            }}
            QPushButton:hover {{
                background: {Colors.BORDER};
            }}
        """)

    def _on_detached_window_closed(self):
        self._close_detached_window()

    def _update_realtime_display(self, position_ms: int):
        audio_files = getattr(self, '_audio_files', None)
        crossfade_sec = getattr(self, '_crossfade_sec', 5)

        if audio_files:
            total_ms = 0
            for i, af in enumerate(audio_files):
                if i == 0:
                    total_ms += af.duration * 1000
                else:
                    total_ms += max(0, af.duration - crossfade_sec) * 1000
            total_ms = int(total_ms)
        else:
            total_ms = 0

        pos_min = position_ms // 60000
        pos_sec = (position_ms % 60000) // 1000
        pos_ms_frac = (position_ms % 1000) // 10

        total_min = total_ms // 60000
        total_sec = (total_ms % 60000) // 1000

        time_str = f"{pos_min:02d}:{pos_sec:02d}.{pos_ms_frac:02d} / {total_min:02d}:{total_sec:02d}"

        if self.current_speed != 1.0:
            time_str += f" [{self.current_speed}x]"

        self.realtime_label.setText(time_str)

        if audio_files:
            position_sec = position_ms / 1000.0
            cumulative = 0
            current_name = ""
            current_idx = 0
            for i, af in enumerate(audio_files):
                if i == 0:
                    track_end = cumulative + af.duration
                else:
                    track_end = cumulative + max(0, af.duration - crossfade_sec)

                if position_sec < track_end or i == len(audio_files) - 1:
                    current_name = af.name
                    current_idx = i
                    break
                cumulative = track_end

            display_name = re.sub(r'^[\d]+[\.\-\s]+', '', current_name)
            display_name = os.path.splitext(display_name)[0]
            self.caption.setText(f"\U0001f3b5 {current_idx + 1}/{len(audio_files)}: {display_name}")

            if self.detached_window:
                self.detached_window.set_track_info(display_name, time_str)
