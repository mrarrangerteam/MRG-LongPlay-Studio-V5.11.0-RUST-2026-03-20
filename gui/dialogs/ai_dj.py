"""
AI DJ dialog — smart playlist ordering with BPM/Key/Energy analysis.

Classes:
    AIDJDialog — Dialog for AI DJ features
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
    QMediaPlayer, QAudioOutput, QUrl, pyqtSignal, QApplication,
)
from gui.styles import Colors
from gui.timeline.track_list import DraggableTrackListWidget


class AIDJDialog(QDialog):
    """Dialog for AI DJ features - smart playlist ordering"""
    orderApplied = pyqtSignal(list)  # emits new order of file paths
    
    def __init__(self, audio_files: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🎧 AI DJ - Smart Playlist")
        self.setMinimumSize(900, 700)
        self.setStyleSheet(f"""
            QDialog {{
                background: {Colors.BG_PRIMARY};
            }}
        """)
        
        self.audio_files = audio_files  # List of MediaFile objects
        self.ai_dj = None
        self.current_order = []
        self.track_data = []
        
        # Import AI DJ
        try:
            from ai_dj import AIDJ, AudioAnalysis
            self.ai_dj = AIDJ()
        except ImportError:
            pass
            
        self._setup_ui()
        self._analyze_tracks()
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Header
        header_layout = QHBoxLayout()
        
        header = QLabel("🎧 AI DJ - Smart Playlist Ordering")
        header.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; font-size: 18px; font-weight: bold;")
        header_layout.addWidget(header)
        
        header_layout.addStretch()
        
        # Strategy selector
        strategy_label = QLabel("Strategy:")
        strategy_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        header_layout.addWidget(strategy_label)
        
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems([
            "🎯 Smooth Flow (Key + Energy)",
            "📈 Energy Up (Low → High)",
            "📉 Energy Down (High → Low)",
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
        
        self.best_opener_label = QLabel("🏆 Best #1: Analyzing...")
        self.best_opener_label.setStyleSheet(f"color: {Colors.ACCENT}; font-weight: bold;")
        stats_layout.addWidget(self.best_opener_label)
        
        stats_layout.addStretch()
        
        self.flow_label = QLabel("📊 Flow: --")
        self.flow_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        stats_layout.addWidget(self.flow_label)
        
        self.energy_label = QLabel("⚡ Energy: --")
        self.energy_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        stats_layout.addWidget(self.energy_label)
        
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
        
        # Navigation buttons for shuffle history
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
        
        # Preview controls row
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
        
        # Seek slider
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
                background: {Colors.ACCENT};
                width: 16px;
                height: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }}
            QSlider::sub-page:horizontal {{
                background: {Colors.ACCENT};
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
        self.now_playing_label = QLabel("Double-click track or select & press ▶️")
        self.now_playing_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-style: italic; font-size: 11px;")
        self.now_playing_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.now_playing_label)
        
        # Track list
        list_header = QLabel("📋 Track Order (drag to reorder, double-click to play):")
        list_header.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 12px;")
        layout.addWidget(list_header)
        
        self.track_list = DraggableTrackListWidget()
        self.track_list.setMinimumHeight(350)
        self.track_list.orderChanged.connect(self._on_order_changed)
        self.track_list.playRequested.connect(self._play_track_at_index)
        layout.addWidget(self.track_list)
        
        # Setup audio player for preview
        self.preview_player = QMediaPlayer()
        self.preview_audio = QAudioOutput()
        self.preview_player.setAudioOutput(self.preview_audio)
        self.preview_player.positionChanged.connect(self._on_position_changed)
        self.preview_player.durationChanged.connect(self._on_duration_changed)
        
        # Seek state
        self.is_seeking = False
        self.is_playing = False
        
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
        clear_btn.clicked.connect(self._clear_all_tracks)
        bottom_layout.addWidget(clear_btn)
        
        bottom_layout.addStretch()
        
        # Rename button - rename files with numbering
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
        
    def _analyze_tracks(self):
        """Analyze all tracks"""
        if not self.ai_dj or not self.audio_files:
            return
            
        self.track_data = []
        for af in self.audio_files:
            analysis = self.ai_dj.analyze_track(af.path)
            self.track_data.append({
                'file_path': af.path,
                'name': os.path.splitext(os.path.basename(af.path))[0],
                'duration_sec': analysis.duration_sec,
                'bpm': analysis.bpm,
                'key': analysis.key,
                'energy': analysis.energy,
                'intro_score': analysis.intro_score,
            })
            
        # Set initial order
        self.current_order = [t['file_path'] for t in self.track_data]
        self.track_list.set_tracks(self.track_data)
        
        # Update stats
        self._update_stats()
        
    def _update_stats(self):
        """Update statistics display"""
        if not self.ai_dj or not self.current_order:
            return
            
        # Find best opener
        best = self.ai_dj.get_best_opener(self.current_order, top_n=1)
        if best:
            best_name = os.path.splitext(os.path.basename(best[0][0]))[0]
            self.best_opener_label.setText(f"🏆 Best #1: {best_name} (Score: {best[0][1]:.0f})")
            
        # Get stats
        stats = self.ai_dj.get_playlist_stats(self.current_order)
        if stats:
            self.flow_label.setText(f"📊 Flow: {stats['smoothness']:.0f}% Smooth")
            self.energy_label.setText(f"⚡ Balance: {stats['energy_balance']:.0f}%")
            
    def _get_strategy(self) -> str:
        """Get selected strategy"""
        idx = self.strategy_combo.currentIndex()
        strategies = ["smooth", "energy_up", "energy_down", "random_smart"]
        return strategies[idx]
        
    def _ai_suggest(self):
        """Run AI suggestion"""
        if not self.ai_dj:
            return
            
        strategy = self._get_strategy()
        new_order = self.ai_dj.suggest_order(self.current_order, strategy)
        self._apply_new_order(new_order)
        
    def _shuffle_again(self):
        """Generate another shuffle"""
        if not self.ai_dj:
            return
            
        new_order = self.ai_dj.shuffle_again(self.current_order)
        self._apply_new_order(new_order)
        
        # Update navigation buttons
        self.prev_btn.setEnabled(self.ai_dj.current_shuffle_index > 0)
        self.next_btn.setEnabled(False)
        
    def _pure_random(self):
        """Pure random shuffle"""
        import random
        new_order = self.current_order.copy()
        random.shuffle(new_order)
        self._apply_new_order(new_order)
        
    def _previous_shuffle(self):
        """Go to previous shuffle in history"""
        if self.ai_dj:
            prev_order = self.ai_dj.get_previous_shuffle()
            if prev_order:
                self._apply_new_order(prev_order)
                self.prev_btn.setEnabled(self.ai_dj.current_shuffle_index > 0)
                self.next_btn.setEnabled(True)
                
    def _next_shuffle(self):
        """Go to next shuffle in history"""
        if self.ai_dj:
            next_order = self.ai_dj.get_next_shuffle()
            if next_order:
                self._apply_new_order(next_order)
                self.next_btn.setEnabled(self.ai_dj.current_shuffle_index < len(self.ai_dj.shuffle_history) - 1)
                self.prev_btn.setEnabled(True)
                
    def _apply_new_order(self, new_order: list):
        """Apply a new track order"""
        self.current_order = new_order
        
        # Rebuild track_data in new order
        path_to_data = {t['file_path']: t for t in self.track_data}
        reordered_data = [path_to_data[p] for p in new_order if p in path_to_data]
        
        self.track_list.set_tracks(reordered_data)
        self._update_stats()
        
    def _on_order_changed(self, new_paths: list):
        """Handle manual reorder via drag & drop"""
        self.current_order = new_paths
        self._update_stats()
        
    def _toggle_play(self):
        """Toggle play/pause"""
        if self.is_playing:
            self.preview_player.pause()
            self.play_btn.setText("▶️")
            self.is_playing = False
        else:
            if self.preview_player.source().isEmpty():
                # No track loaded, play selected or first
                current_row = self.track_list.currentRow()
                if current_row >= 0:
                    self._play_track_at_index(current_row)
                elif self.track_list.count() > 0:
                    self._play_track_at_index(0)
            else:
                self.preview_player.play()
                self.play_btn.setText("⏸️")
                self.is_playing = True
            
    def _play_track_at_index(self, index: int):
        """Play track at specific index"""
        if index < 0 or index >= len(self.current_order):
            return
            
        file_path = self.current_order[index]
        track_name = os.path.basename(file_path)
        
        self.preview_player.stop()
        self.preview_player.setSource(QUrl.fromLocalFile(file_path))
        self.preview_player.play()
        
        self.play_btn.setText("⏸️")
        self.is_playing = True
        
        self.now_playing_label.setText(f"🎵 {track_name}")
        self.now_playing_label.setStyleSheet(f"color: {Colors.METER_GREEN}; font-weight: bold; font-size: 11px;")
        
        # Select the track in list
        self.track_list.setCurrentRow(index)
        
    def _stop_preview(self):
        """Stop preview playback"""
        self.preview_player.stop()
        self.play_btn.setText("▶️")
        self.is_playing = False
        self.seek_slider.setValue(0)
        self.time_label.setText("0:00")
        self.now_playing_label.setText("Playback stopped")
        self.now_playing_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-style: italic; font-size: 11px;")
        
    def _on_position_changed(self, position: int):
        """Handle playback position change"""
        if not self.is_seeking:
            duration = self.preview_player.duration()
            if duration > 0:
                slider_pos = int((position / duration) * 1000)
                self.seek_slider.setValue(slider_pos)
            
            # Update time label
            secs = position // 1000
            mins = secs // 60
            secs = secs % 60
            self.time_label.setText(f"{mins}:{secs:02d}")
            
    def _on_duration_changed(self, duration: int):
        """Handle duration change when new track loads"""
        secs = duration // 1000
        mins = secs // 60
        secs = secs % 60
        self.duration_label.setText(f"{mins}:{secs:02d}")
        
    def _on_seek_pressed(self):
        """User started dragging seek slider"""
        self.is_seeking = True
        
    def _on_seek_released(self):
        """User released seek slider"""
        self.is_seeking = False
        duration = self.preview_player.duration()
        if duration > 0:
            position = int((self.seek_slider.value() / 1000) * duration)
            self.preview_player.setPosition(position)
            
    def _on_seek_moved(self, value: int):
        """User is dragging seek slider"""
        duration = self.preview_player.duration()
        if duration > 0:
            position = int((value / 1000) * duration)
            secs = position // 1000
            mins = secs // 60
            secs = secs % 60
            self.time_label.setText(f"{mins}:{secs:02d}")
        
    def _apply_order(self):
        """Apply and close"""
        self._stop_preview()  # Stop playback before closing
        self.orderApplied.emit(self.current_order)
        self.accept()
    
    def _clear_all_tracks(self):
        """Clear all tracks and close dialog"""
        if not self.current_order:
            return
            
        reply = QMessageBox.question(
            self,
            "Clear All Tracks?",
            f"ลบเพลงทั้งหมด {len(self.current_order)} เพลงออกจากโปรเจค?\n\n"
            "⚠️ ไฟล์จะไม่ถูกลบ แค่จะเอาออกจาก playlist",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self._stop_preview()
            self.current_order.clear()
            self.track_data.clear()
            self.track_list.clear()
            self.orderApplied.emit([])  # Emit empty list to clear in main window
            self.accept()
        
    def _rename_files(self):
        """Rename files with sequential numbering - replaces old numbering"""
        if not self.current_order:
            return
            
        # Confirm with user
        reply = QMessageBox.question(
            self,
            "Rename Files?",
            f"This will rename {len(self.current_order)} files with sequential numbering:\n\n"
            "Example:\n"
            "  01.Song Name.wav\n"
            "  02.Another Song.wav\n"
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
                # Keep original path for retry
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
                
        # Update current order with new paths
        self.current_order = new_order
        
        # Update track_data with new paths
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
                f"✅ Successfully renamed {renamed_count} files!"
            )

