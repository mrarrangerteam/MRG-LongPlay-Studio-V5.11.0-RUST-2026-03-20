"""
AI Video dialog — smart video ordering with shuffle/reorder.

Classes:
    AIVideoDialog — Dialog for AI Video ordering
"""

import os
import random
import re
import shutil
import uuid
from typing import List

from gui.utils.compat import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QComboBox, QSlider, QMessageBox, Qt,
    QMediaPlayer, QAudioOutput, QVideoWidget, QUrl, pyqtSignal, QApplication,
)
from gui.styles import Colors
from gui.timeline.track_list import DraggableTrackListWidget


class AIVideoDialog(QDialog):
    """Dialog for AI Video ordering - shuffle/reorder videos"""
    orderApplied = pyqtSignal(list)  # emits new order of file paths
    
    def __init__(self, video_files: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🎬 AI VDO - Smart Video Order")
        self.setMinimumSize(900, 600)
        self.setStyleSheet(f"""
            QDialog {{
                background: {Colors.BG_PRIMARY};
            }}
        """)
        
        self.video_files = video_files  # List of MediaFile objects
        self.current_order = []
        self.track_data = []
        self.shuffle_history = []
        self.current_shuffle_index = -1
        
        self._setup_ui()
        self._analyze_videos()
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Header
        header_layout = QHBoxLayout()
        
        header = QLabel("🎬 AI VDO - Smart Video Ordering")
        header.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; font-size: 18px; font-weight: bold;")
        header_layout.addWidget(header)
        
        header_layout.addStretch()
        
        # Strategy selector
        strategy_label = QLabel("Strategy:")
        strategy_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        header_layout.addWidget(strategy_label)
        
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems([
            "📏 By Duration (Short → Long)",
            "📐 By Duration (Long → Short)",
            "🔤 By Name (A → Z)",
            "🎲 Smart Random"
        ])
        self.strategy_combo.setStyleSheet(f"""
            QComboBox {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 6px;
                padding: 8px;
                min-width: 200px;
            }}
        """)
        header_layout.addWidget(self.strategy_combo)
        
        layout.addLayout(header_layout)
        
        # Stats panel
        self.stats_frame = QFrame()
        self.stats_frame.setStyleSheet(f"""
            QFrame {{
                background: {Colors.BG_SECONDARY};
                border-radius: 8px;
                padding: 10px;
            }}
        """)
        stats_layout = QHBoxLayout(self.stats_frame)
        
        self.video_count_label = QLabel(f"📹 Videos: {len(self.video_files)}")
        self.video_count_label.setStyleSheet(f"color: {Colors.ACCENT}; font-weight: bold;")
        stats_layout.addWidget(self.video_count_label)
        
        stats_layout.addStretch()
        
        self.total_duration_label = QLabel("⏱️ Total: --")
        self.total_duration_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        stats_layout.addWidget(self.total_duration_label)
        
        layout.addWidget(self.stats_frame)
        
        # Buttons row
        btn_layout = QHBoxLayout()
        
        self.suggest_btn = QPushButton("🤖 AI Suggest")
        self.suggest_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.ACCENT};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: {Colors.ACCENT_DIM};
            }}
        """)
        self.suggest_btn.clicked.connect(self._ai_suggest)
        btn_layout.addWidget(self.suggest_btn)
        
        self.shuffle_btn = QPushButton("🔄 Shuffle Again")
        self.shuffle_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.VIDEO_COLOR};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: #e07b00;
            }}
        """)
        self.shuffle_btn.clicked.connect(self._shuffle_again)
        btn_layout.addWidget(self.shuffle_btn)
        
        self.random_btn = QPushButton("🎲 Pure Random")
        self.random_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
            }}
            QPushButton:hover {{
                background: {Colors.BORDER};
            }}
        """)
        self.random_btn.clicked.connect(self._pure_random)
        btn_layout.addWidget(self.random_btn)
        
        btn_layout.addStretch()
        
        # Navigation buttons
        self.prev_btn = QPushButton("⬅️ Previous")
        self.prev_btn.setStyleSheet(self.random_btn.styleSheet())
        self.prev_btn.clicked.connect(self._previous_shuffle)
        self.prev_btn.setEnabled(False)
        btn_layout.addWidget(self.prev_btn)
        
        self.next_btn = QPushButton("➡️ Next")
        self.next_btn.setStyleSheet(self.random_btn.styleSheet())
        self.next_btn.clicked.connect(self._next_shuffle)
        self.next_btn.setEnabled(False)
        btn_layout.addWidget(self.next_btn)
        
        layout.addLayout(btn_layout)
        
        # Video Preview Widget (actual video display)
        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumHeight(200)
        self.video_widget.setStyleSheet("background: #000000; border-radius: 8px;")
        layout.addWidget(self.video_widget)
        
        # Video Preview Controls (like AI DJ audio player)
        preview_layout = QHBoxLayout()
        
        self.play_btn = QPushButton("▶️")
        self.play_btn.setFixedSize(50, 40)
        self.play_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.METER_GREEN};
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: #45a049;
            }}
        """)
        self.play_btn.clicked.connect(self._toggle_play)
        preview_layout.addWidget(self.play_btn)
        
        self.stop_btn = QPushButton("⏹️")
        self.stop_btn.setFixedSize(50, 40)
        self.stop_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.METER_RED};
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 16px;
            }}
            QPushButton:hover {{
                background: #cc3333;
            }}
        """)
        self.stop_btn.clicked.connect(self._stop_preview)
        preview_layout.addWidget(self.stop_btn)
        
        # Time display
        self.time_label = QLabel("0:00")
        self.time_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; font-family: 'Menlo', 'Courier New'; min-width: 45px;")
        preview_layout.addWidget(self.time_label)
        
        # Video Seek slider
        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setMinimum(0)
        self.seek_slider.setMaximum(1000)
        self.seek_slider.setValue(0)
        self.seek_slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                background: {Colors.BG_TERTIARY};
                height: 8px;
                border-radius: 4px;
            }}
            QSlider::handle:horizontal {{
                background: {Colors.VIDEO_COLOR};
                width: 16px;
                height: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }}
            QSlider::sub-page:horizontal {{
                background: {Colors.VIDEO_COLOR};
                border-radius: 4px;
            }}
        """)
        self.seek_slider.sliderPressed.connect(self._on_seek_pressed)
        self.seek_slider.sliderReleased.connect(self._on_seek_released)
        self.seek_slider.sliderMoved.connect(self._on_seek_moved)
        preview_layout.addWidget(self.seek_slider, 1)
        
        # Duration display
        self.duration_label = QLabel("0:00")
        self.duration_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-family: 'Menlo', 'Courier New'; min-width: 45px;")
        preview_layout.addWidget(self.duration_label)
        
        layout.addLayout(preview_layout)
        
        # Now playing label
        self.now_playing_label = QLabel("Double-click video or select & press ▶️")
        self.now_playing_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-style: italic; font-size: 11px;")
        self.now_playing_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.now_playing_label)
        
        # Setup video player for preview
        self.preview_player = QMediaPlayer()
        self.preview_audio = QAudioOutput()
        self.preview_player.setAudioOutput(self.preview_audio)
        self.preview_player.setVideoOutput(self.video_widget)  # Connect to video widget!
        self.preview_player.positionChanged.connect(self._on_position_changed)
        self.preview_player.durationChanged.connect(self._on_duration_changed)
        
        # Seek state
        self.is_seeking = False
        self.is_playing = False
        
        # Video list
        list_header = QLabel("📋 Video Order (drag to reorder, double-click to play):")
        list_header.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 12px;")
        layout.addWidget(list_header)
        
        self.track_list = DraggableTrackListWidget()
        self.track_list.setMinimumHeight(300)
        self.track_list.orderChanged.connect(self._on_order_changed)
        self.track_list.playRequested.connect(self._play_video_at_index)
        layout.addWidget(self.track_list)
        
        # Bottom buttons
        bottom_layout = QHBoxLayout()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 8px;
                padding: 12px 30px;
                font-size: 14px;
            }}
            QPushButton:hover {{
                background: {Colors.BORDER};
            }}
        """)
        cancel_btn.clicked.connect(self.reject)
        bottom_layout.addWidget(cancel_btn)
        
        # Clear All button
        clear_btn = QPushButton("🗑️ Clear All")
        clear_btn.setStyleSheet(f"""
            QPushButton {{
                background: #FF4444;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: #FF6666;
            }}
        """)
        clear_btn.clicked.connect(self._clear_all_videos)
        bottom_layout.addWidget(clear_btn)
        
        bottom_layout.addStretch()
        
        # Rename button
        rename_btn = QPushButton("📝 Rename Files (01, 02...)")
        rename_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.VIDEO_COLOR};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: #e07b00;
            }}
        """)
        rename_btn.clicked.connect(self._rename_files)
        bottom_layout.addWidget(rename_btn)
        
        apply_btn = QPushButton("✅ Apply Order")
        apply_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.METER_GREEN};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 30px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: #45a049;
            }}
        """)
        apply_btn.clicked.connect(self._apply_order)
        bottom_layout.addWidget(apply_btn)
        
        layout.addLayout(bottom_layout)
        
    def _analyze_videos(self):
        """Analyze all videos"""
        if not self.video_files:
            return
            
        self.track_data = []
        total_duration = 0
        
        for vf in self.video_files:
            duration = vf.duration if hasattr(vf, 'duration') else 5.0
            total_duration += duration
            
            self.track_data.append({
                'file_path': vf.path,
                'name': os.path.splitext(os.path.basename(vf.path))[0],
                'duration_sec': duration,
                'bpm': 0,  # Not applicable for video
                'key': 'N/A',
                'energy': 50,  # Default
                'intro_score': 50,
            })
            
        # Set initial order
        self.current_order = [t['file_path'] for t in self.track_data]
        self.track_list.set_tracks(self.track_data)
        
        # Update stats
        mins = int(total_duration) // 60
        secs = int(total_duration) % 60
        self.total_duration_label.setText(f"⏱️ Total: {mins}:{secs:02d}")
        self.video_count_label.setText(f"📹 Videos: {len(self.video_files)}")
        
    def _get_strategy(self) -> str:
        """Get selected strategy"""
        idx = self.strategy_combo.currentIndex()
        strategies = ["duration_asc", "duration_desc", "name_asc", "random"]
        return strategies[idx]
        
    def _ai_suggest(self):
        """Run AI suggestion based on strategy"""
        strategy = self._get_strategy()
        
        if strategy == "duration_asc":
            # Sort by duration (short to long)
            sorted_data = sorted(self.track_data, key=lambda x: x['duration_sec'])
            new_order = [t['file_path'] for t in sorted_data]
        elif strategy == "duration_desc":
            # Sort by duration (long to short)
            sorted_data = sorted(self.track_data, key=lambda x: x['duration_sec'], reverse=True)
            new_order = [t['file_path'] for t in sorted_data]
        elif strategy == "name_asc":
            # Sort by name
            sorted_data = sorted(self.track_data, key=lambda x: x['name'].lower())
            new_order = [t['file_path'] for t in sorted_data]
        else:
            # Random
            import random
            new_order = self.current_order.copy()
            random.shuffle(new_order)
            
        self._apply_new_order(new_order)
        self._add_to_history(new_order)
        
    def _shuffle_again(self):
        """Generate another shuffle"""
        import random
        new_order = self.current_order.copy()
        random.shuffle(new_order)
        self._apply_new_order(new_order)
        self._add_to_history(new_order)
        
    def _pure_random(self):
        """Pure random shuffle"""
        import random
        new_order = self.current_order.copy()
        random.shuffle(new_order)
        self._apply_new_order(new_order)
        self._add_to_history(new_order)
        
    def _add_to_history(self, order: list):
        """Add order to shuffle history"""
        # Truncate future history if we're not at the end
        if self.current_shuffle_index < len(self.shuffle_history) - 1:
            self.shuffle_history = self.shuffle_history[:self.current_shuffle_index + 1]
        
        self.shuffle_history.append(order.copy())
        self.current_shuffle_index = len(self.shuffle_history) - 1
        
        self.prev_btn.setEnabled(self.current_shuffle_index > 0)
        self.next_btn.setEnabled(False)
        
    def _previous_shuffle(self):
        """Go to previous shuffle"""
        if self.current_shuffle_index > 0:
            self.current_shuffle_index -= 1
            self._apply_new_order(self.shuffle_history[self.current_shuffle_index])
            self.prev_btn.setEnabled(self.current_shuffle_index > 0)
            self.next_btn.setEnabled(True)
            
    def _next_shuffle(self):
        """Go to next shuffle"""
        if self.current_shuffle_index < len(self.shuffle_history) - 1:
            self.current_shuffle_index += 1
            self._apply_new_order(self.shuffle_history[self.current_shuffle_index])
            self.next_btn.setEnabled(self.current_shuffle_index < len(self.shuffle_history) - 1)
            self.prev_btn.setEnabled(True)
            
    def _apply_new_order(self, new_order: list):
        """Apply a new track order"""
        self.current_order = new_order
        
        # Rebuild track_data in new order
        path_to_data = {t['file_path']: t for t in self.track_data}
        reordered_data = [path_to_data[p] for p in new_order if p in path_to_data]
        
        self.track_list.set_tracks(reordered_data)
        
    def _on_order_changed(self, new_paths: list):
        """Handle manual reorder via drag & drop"""
        self.current_order = new_paths
        
    # ==================== Video Preview Methods ====================
    def _toggle_play(self):
        """Toggle play/pause for video preview"""
        if self.is_playing:
            self.preview_player.pause()
            self.play_btn.setText("▶️")
            self.is_playing = False
        else:
            # If no video loaded, try to play selected
            if self.preview_player.source().isEmpty():
                selected = self.track_list.currentRow()
                if selected >= 0 and selected < len(self.current_order):
                    self._play_video_at_index(selected)
                    return
            self.preview_player.play()
            self.play_btn.setText("⏸️")
            self.is_playing = True
            
    def _stop_preview(self):
        """Stop video preview"""
        self.preview_player.stop()
        self.play_btn.setText("▶️")
        self.is_playing = False
        self.seek_slider.setValue(0)
        self.time_label.setText("0:00")
        self.now_playing_label.setText("Double-click video or select & press ▶️")
        
    def _play_video_at_index(self, index: int):
        """Play video at specific index"""
        if index < 0 or index >= len(self.current_order):
            return
            
        video_path = self.current_order[index]
        video_name = os.path.basename(video_path)
        
        self.preview_player.setSource(QUrl.fromLocalFile(video_path))
        self.preview_player.play()
        self.play_btn.setText("⏸️")
        self.is_playing = True
        self.now_playing_label.setText(f"🎬 Playing: {video_name}")
        
    def _on_position_changed(self, position: int):
        """Handle playback position change"""
        if not self.is_seeking:
            duration = self.preview_player.duration()
            if duration > 0:
                slider_pos = int((position / duration) * 1000)
                self.seek_slider.setValue(slider_pos)
                
        # Update time label
        pos_sec = position // 1000
        pos_min = pos_sec // 60
        pos_sec = pos_sec % 60
        self.time_label.setText(f"{pos_min}:{pos_sec:02d}")
        
    def _on_duration_changed(self, duration: int):
        """Handle duration change"""
        dur_sec = duration // 1000
        dur_min = dur_sec // 60
        dur_sec = dur_sec % 60
        self.duration_label.setText(f"{dur_min}:{dur_sec:02d}")
        
    def _on_seek_pressed(self):
        """Handle seek slider press"""
        self.is_seeking = True
        
    def _on_seek_released(self):
        """Handle seek slider release"""
        self.is_seeking = False
        duration = self.preview_player.duration()
        if duration > 0:
            position = int((self.seek_slider.value() / 1000) * duration)
            self.preview_player.setPosition(position)
            
    def _on_seek_moved(self, value: int):
        """Handle seek slider move"""
        duration = self.preview_player.duration()
        if duration > 0:
            position = int((value / 1000) * duration)
            pos_sec = position // 1000
            pos_min = pos_sec // 60
            pos_sec = pos_sec % 60
            self.time_label.setText(f"{pos_min}:{pos_sec:02d}")
    
    def _apply_order(self):
        """Apply and close"""
        # Stop any playing video first
        self.preview_player.stop()
        self.orderApplied.emit(self.current_order)
        self.accept()
    
    def _clear_all_videos(self):
        """Clear all videos and close dialog"""
        if not self.current_order:
            return
            
        reply = QMessageBox.question(
            self,
            "Clear All Videos?",
            f"ลบวิดีโอทั้งหมด {len(self.current_order)} ไฟล์ออกจากโปรเจค?\n\n"
            "⚠️ ไฟล์จะไม่ถูกลบ แค่จะเอาออกจาก playlist",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.preview_player.stop()
            self.current_order.clear()
            self.track_data.clear()
            self.track_list.clear()
            self.orderApplied.emit([])  # Emit empty list to clear in main window
            self.accept()
        
    def _rename_files(self):
        """Rename video files with sequential numbering - replaces old numbering"""
        if not self.current_order:
            return
            
        # Confirm with user
        reply = QMessageBox.question(
            self,
            "Rename Video Files?",
            f"This will rename {len(self.current_order)} video files with sequential numbering:\n\n"
            "Example:\n"
            "  01.Video Name.mp4\n"
            "  02.Another Video.mp4\n"
            "  ...\n\n"
            "⚠️ This will REPLACE existing numbering!\n\n"
            "Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
            
        import re
        import shutil
        import uuid
        
        renamed_count = 0
        new_order = []
        errors = []
        
        # Phase 1: Rename all files to temporary unique names first
        temp_paths = []
        for file_path in self.current_order:
            try:
                if not os.path.exists(file_path):
                    errors.append(f"{os.path.basename(file_path)}: File not found")
                    continue
                    
                directory = os.path.dirname(file_path)
                old_name = os.path.basename(file_path)
                ext = os.path.splitext(old_name)[1]
                
                # Create unique temp name in same directory
                temp_name = f"_temp_{uuid.uuid4().hex}{ext}"
                temp_path = os.path.join(directory, temp_name)
                
                # Use shutil.move for cross-filesystem support
                shutil.move(file_path, temp_path)
                temp_paths.append((temp_path, old_name, directory))
            except Exception as e:
                errors.append(f"{os.path.basename(file_path)}: {str(e)}")
                temp_paths.append((file_path, os.path.basename(file_path), os.path.dirname(file_path)))
        
        # Phase 2: Rename from temp to final numbered names
        for idx, (temp_path, old_name, directory) in enumerate(temp_paths, 1):
            try:
                # Remove existing numbering prefix (handles "01.Song", "01 Song", "01. Song", "01_Song")
                clean_name = re.sub(r'^\d+[.\s_-]+', '', old_name)
                # Also remove leading dots/spaces/underscores if any remain
                clean_name = clean_name.lstrip('. _-')
                # If nothing left, use original name
                if not clean_name:
                    clean_name = old_name
                
                # Add new numbering
                new_name = f"{idx:02d}.{clean_name}"
                new_path = os.path.join(directory, new_name)
                
                shutil.move(temp_path, new_path)
                new_order.append(new_path)
                renamed_count += 1
                
            except Exception as e:
                errors.append(f"{old_name}: {str(e)}")
                new_order.append(temp_path)
                
        # Update current order
        self.current_order = new_order
        
        # Update track_data
        for i, track in enumerate(self.track_data):
            if i < len(new_order):
                track['file_path'] = new_order[i]
                track['name'] = os.path.splitext(os.path.basename(new_order[i]))[0]
        
        # Refresh display
        self._apply_new_order(self.current_order)
        
        # Show result
        if errors:
            QMessageBox.warning(
                self,
                "Rename Completed with Errors",
                f"✅ Renamed: {renamed_count} files\n"
                f"❌ Errors: {len(errors)}\n\n"
                + "\n".join(errors[:5])
            )
        else:
            QMessageBox.information(
                self,
                "Rename Complete",
                f"✅ Successfully renamed {renamed_count} video files!"
            )

