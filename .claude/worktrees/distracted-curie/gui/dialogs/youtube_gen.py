"""
YouTube metadata generator dialog — titles, descriptions, timestamps, tags.

Classes:
    YouTubeGeneratorDialog — Dialog for generating YouTube metadata
"""

import os
from typing import List

from gui.utils.compat import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QComboBox, QSpinBox, QLineEdit, QTextEdit,
    QMessageBox, QFileDialog, QApplication, Qt, pyqtSignal,
)
from gui.styles import Colors


class YouTubeGeneratorDialog(QDialog):
    """Dialog for generating YouTube metadata with Channel Presets"""
    
    # Channel Presets with full templates
    CHANNEL_PRESETS = {
        "Chillin' Vibes Official": {
            "url": "https://www.youtube.com/@ChillinVibesOfficial",
            "style": "chill_music",
            "tags": "chill music, relaxing music, study music, work music, เพลงชิลๆ, เพลงฟังสบาย, เพลงทำงาน, เพลงอ่านหนังสือ, lofi, cafe music, ambient music, peaceful music, เพลงคลายเครียด, เพลงพักผ่อน, ChillinVibes, RoadTripVibes, DrivingMusic, TravelPlaylist"
        },
        "Custom Channel": {
            "url": "",
            "style": "custom",
            "tags": ""
        }
    }
    
    # SEO Keywords Database (from research)
    SEO_KEYWORDS_DB = {
        "High Volume (34M+)": "chill music, study music, sleep music, relaxing music, lofi hip hop, jazz music, piano music, meditation music, background music, calm music",
        "Relax/Chill": "เพลงเพราะๆ ฟังสบาย, chill vibes, peaceful music, calm music, soft music, gentle music, soothing music, tranquil music, เพลงชิลๆ, เพลงผ่อนคลาย",
        "Study/Work": "focus music, productivity music, concentration music, background music for work, music for studying, deep focus, work music, office music, เพลงทำงาน, เพลงอ่านหนังสือ",
        "Sleep": "deep sleep music, เพลงก่อนนอน, sleep meditation, insomnia music, relaxing sleep music, bedtime music, night music, เพลงนอนหลับ, เพลงกล่อมนอน",
        "Cafe/Jazz": "cafe music, coffee shop music, bossa nova, jazz cafe, morning coffee music, cafe ambience, jazz lounge, เพลงร้านกาแฟ, เพลงคาเฟ่",
        "Piano": "relaxing piano, soft piano music, piano instrumental, peaceful piano, calm piano, piano for relaxation, เพลงเปียโน, เปียโนบรรเลง",
        "Thai Keywords": "เพลงชิลๆ, เพลงฟังสบาย, เพลงทำงาน, เพลงอ่านหนังสือ, เพลงคลายเครียด, เพลงพักผ่อน, เพลงนอนหลับ, เพลงร้านกาแฟ, เพลงเปียโน, เพลงบรรเลง"
    }
    
    def __init__(self, audio_files: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("📺 YouTube Generator")
        self.setMinimumSize(800, 800)
        self.setStyleSheet(f"""
            QDialog {{
                background: {Colors.BG_PRIMARY};
            }}
        """)
        
        self.audio_files = audio_files
        self.yt_gen = None
        
        # Import YouTube Generator
        try:
            from ai_dj import YouTubeGenerator
            self.yt_gen = YouTubeGenerator()
        except ImportError:
            pass
            
        self._setup_ui()
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Header
        header = QLabel("📺 YouTube Metadata Generator")
        header.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; font-size: 18px; font-weight: bold;")
        layout.addWidget(header)
        
        # Settings row
        settings_layout = QHBoxLayout()
        
        # Channel Preset
        channel_label = QLabel("📺 Channel:")
        channel_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        settings_layout.addWidget(channel_label)
        
        self.channel_combo = QComboBox()
        self.channel_combo.addItems(list(self.CHANNEL_PRESETS.keys()))
        self.channel_combo.setStyleSheet(f"""
            QComboBox {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 6px;
                padding: 8px;
                min-width: 150px;
            }}
        """)
        settings_layout.addWidget(self.channel_combo)
        
        settings_layout.addSpacing(15)
        
        # Volume
        vol_label = QLabel("Volume:")
        vol_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        settings_layout.addWidget(vol_label)
        
        self.volume_spin = QSpinBox()
        self.volume_spin.setRange(1, 999)
        self.volume_spin.setValue(1)
        self.volume_spin.setPrefix("Vol. ")
        self.volume_spin.setStyleSheet(f"""
            QSpinBox {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 6px;
                padding: 8px;
                min-width: 100px;
            }}
        """)
        settings_layout.addWidget(self.volume_spin)
        
        settings_layout.addSpacing(20)
        
        # Theme
        theme_label = QLabel("Theme:")
        theme_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        settings_layout.addWidget(theme_label)
        
        self.theme_combo = QComboBox()
        self.theme_combo.addItems([
            "☕ Cafe & Coffee",
            "🚗 Driving & Travel",
            "🌙 Sleep & Relax",
            "💪 Workout & Energy",
            "🎯 Focus & Study",
            "🌴 Chill Vibes"
        ])
        self.theme_combo.setStyleSheet(f"""
            QComboBox {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 6px;
                padding: 8px;
                min-width: 180px;
            }}
        """)
        settings_layout.addWidget(self.theme_combo)
        
        settings_layout.addSpacing(20)
        
        # SEO Keywords Preset
        seo_label = QLabel("SEO:")
        seo_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        settings_layout.addWidget(seo_label)
        
        self.seo_combo = QComboBox()
        self.seo_combo.addItems([
            "🎯 Auto (Theme-based)",
            "📈 High Volume Keywords",
            "🎵 Relax/Chill Keywords",
            "📚 Study/Work Keywords",
            "😴 Sleep Keywords",
            "☕ Cafe/Jazz Keywords",
            "🎹 Piano/Instrumental",
            "🇹🇭 Thai Keywords"
        ])
        self.seo_combo.setStyleSheet(self.theme_combo.styleSheet())
        settings_layout.addWidget(self.seo_combo)
        
        # Add SEO button
        add_seo_btn = QPushButton("+ Add")
        add_seo_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.SUCCESS};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 12px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: #2ECC71;
            }}
        """)
        add_seo_btn.clicked.connect(self._add_seo_keywords_to_tags)
        settings_layout.addWidget(add_seo_btn)
        
        settings_layout.addStretch()
        
        # Generate button
        generate_btn = QPushButton("🚀 Generate All")
        generate_btn.setStyleSheet(f"""
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
        generate_btn.clicked.connect(self._generate_all)
        settings_layout.addWidget(generate_btn)
        
        layout.addLayout(settings_layout)
        
        # Title section
        title_header_layout = QHBoxLayout()
        title_header = QLabel("📝 Title:")
        title_header.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 12px;")
        title_header_layout.addWidget(title_header)
        title_header_layout.addStretch()
        
        copy_title_btn = QPushButton("📋 Copy")
        copy_title_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_PRIMARY};
                border: none;
                border-radius: 4px;
                padding: 4px 12px;
                font-size: 11px;
            }}
            QPushButton:hover {{
                background: {Colors.BORDER};
            }}
        """)
        copy_title_btn.clicked.connect(lambda: self._copy_text(self.title_edit.text()))
        title_header_layout.addWidget(copy_title_btn)
        layout.addLayout(title_header_layout)
        
        self.title_edit = QLineEdit()
        self.title_edit.setPlaceholderText("Click 'Generate All' to create title...")
        self.title_edit.setStyleSheet(f"""
            QLineEdit {{
                background: {Colors.BG_SECONDARY};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 8px;
                padding: 12px;
                font-size: 14px;
            }}
        """)
        layout.addWidget(self.title_edit)
        
        # Description section
        desc_header_layout = QHBoxLayout()
        desc_header = QLabel("📄 Description (with Timestamps):")
        desc_header.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 12px;")
        desc_header_layout.addWidget(desc_header)
        desc_header_layout.addStretch()
        
        copy_desc_btn = QPushButton("📋 Copy")
        copy_desc_btn.setStyleSheet(copy_title_btn.styleSheet())
        copy_desc_btn.clicked.connect(lambda: self._copy_text(self.desc_edit.toPlainText()))
        desc_header_layout.addWidget(copy_desc_btn)
        layout.addLayout(desc_header_layout)
        
        self.desc_edit = QTextEdit()
        self.desc_edit.setPlaceholderText("Description with timestamps will appear here...")
        self.desc_edit.setStyleSheet(f"""
            QTextEdit {{
                background: {Colors.BG_SECONDARY};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 8px;
                padding: 12px;
                font-family: 'Menlo', 'Courier New';
                font-size: 12px;
            }}
        """)
        self.desc_edit.setMinimumHeight(250)
        layout.addWidget(self.desc_edit)
        
        # Tags section
        tags_header_layout = QHBoxLayout()
        self.tags_header = QLabel("🏷️ Tags (0/500):")
        self.tags_header.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 12px;")
        tags_header_layout.addWidget(self.tags_header)
        tags_header_layout.addStretch()
        
        copy_tags_btn = QPushButton("📋 Copy")
        copy_tags_btn.setStyleSheet(copy_title_btn.styleSheet())
        copy_tags_btn.clicked.connect(lambda: self._copy_text(self.tags_edit.toPlainText()))
        tags_header_layout.addWidget(copy_tags_btn)
        layout.addLayout(tags_header_layout)
        
        self.tags_edit = QTextEdit()
        self.tags_edit.setPlaceholderText("Tags will appear here (max 500 characters)...")
        self.tags_edit.setStyleSheet(self.desc_edit.styleSheet())
        self.tags_edit.setMaximumHeight(100)
        self.tags_edit.textChanged.connect(self._update_tags_count)
        layout.addWidget(self.tags_edit)
        
        # Bottom buttons
        bottom_layout = QHBoxLayout()
        
        export_btn = QPushButton("📤 Export to .txt")
        export_btn.setStyleSheet(f"""
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
        export_btn.clicked.connect(self._export_txt)
        bottom_layout.addWidget(export_btn)
        
        bottom_layout.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet(f"""
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
        close_btn.clicked.connect(self.accept)
        bottom_layout.addWidget(close_btn)
        
        layout.addLayout(bottom_layout)
        
    def _get_theme_key(self) -> str:
        """Get theme key from combo selection"""
        idx = self.theme_combo.currentIndex()
        themes = ["cafe", "driving", "sleep", "workout", "focus", "chill"]
        return themes[idx]
        
    def _get_seo_type(self) -> str:
        """Get SEO type from combo selection"""
        idx = self.seo_combo.currentIndex()
        seo_types = ["auto", "high_volume", "relax_chill", "study_work", "sleep", "cafe_jazz", "piano_instrumental"]
        return seo_types[idx]
        
    def _on_channel_preset_changed(self, channel_name: str):
        """Handle channel preset change"""
        if channel_name in self.CHANNEL_PRESETS:
            preset = self.CHANNEL_PRESETS[channel_name]
            # Show preset info
            if preset.get('tags'):
                QMessageBox.information(
                    self, "📺 Channel Preset",
                    f"Selected: {channel_name}\n\n"
                    f"Click 'Generate All' to apply this channel's style."
                )
    
    def _add_seo_keywords_to_tags(self):
        """Add SEO keywords from selected preset to tags"""
        seo_text = self.seo_combo.currentText()
        
        # Map combo text to database key
        seo_map = {
            "🎯 Auto (Theme-based)": "auto",
            "📈 High Volume Keywords": "High Volume (34M+)",
            "🎵 Relax/Chill Keywords": "Relax/Chill",
            "📚 Study/Work Keywords": "Study/Work",
            "😴 Sleep Keywords": "Sleep",
            "☕ Cafe/Jazz Keywords": "Cafe/Jazz",
            "🎹 Piano/Instrumental": "Piano",
            "🇹🇭 Thai Keywords": "Thai Keywords"
        }
        
        seo_key = seo_map.get(seo_text, "auto")
        
        if seo_key == "auto":
            # Add top keywords from each category
            keywords_to_add = []
            for key, keywords in self.SEO_KEYWORDS_DB.items():
                kw_list = [k.strip() for k in keywords.split(",")]
                keywords_to_add.extend(kw_list[:2])  # Top 2 from each
            new_keywords = ", ".join(keywords_to_add[:20])  # Max 20
        elif seo_key in self.SEO_KEYWORDS_DB:
            new_keywords = self.SEO_KEYWORDS_DB[seo_key]
        else:
            return
        
        current_tags = self.tags_edit.toPlainText()
        
        if current_tags:
            # Avoid duplicates
            existing = set(t.strip().lower() for t in current_tags.split(","))
            new_kw_list = [k.strip() for k in new_keywords.split(",")]
            unique_new = [k for k in new_kw_list if k.lower() not in existing]
            if unique_new:
                self.tags_edit.setPlainText(f"{current_tags}, {', '.join(unique_new)}")
                QMessageBox.information(self, "✅ Keywords Added", 
                    f"Added {len(unique_new)} SEO keywords!")
            else:
                QMessageBox.information(self, "ℹ️ Info", "All keywords already exist in tags.")
        else:
            self.tags_edit.setPlainText(new_keywords)
            count = len([k for k in new_keywords.split(",") if k.strip()])
            QMessageBox.information(self, "✅ Keywords Added", 
                f"Added {count} SEO keywords!")
    
    def _generate_all(self):
        """Generate all YouTube metadata"""
        try:
            self._do_generate_all()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Generation failed: {str(e)}")
            
    def _do_generate_all(self):
        """Internal generate method"""
        if not self.yt_gen:
            QMessageBox.warning(self, "Error", "YouTube Generator module not available")
            return
            
        volume = self.volume_spin.value()
        theme = self._get_theme_key()
        
        # Build tracks list
        tracks = []
        total_duration = 0
        for af in self.audio_files:
            name = os.path.splitext(os.path.basename(af.path))[0]
            duration_sec = af.duration if af.duration and af.duration > 0 else 180
            tracks.append({
                'name': name,
                'duration_sec': duration_sec
            })
            total_duration += duration_sec
            
        # Generate timestamps
        timestamped = self.yt_gen.generate_timestamps(tracks)
        
        # Format duration
        duration_str = self.yt_gen.format_duration(total_duration)
        
        # Generate title
        title = self.yt_gen.generate_title(volume, theme, duration_str)
        self.title_edit.setText(title)
        
        # Generate description
        desc = self.yt_gen.generate_description(volume, theme, timestamped, duration_str)
        self.desc_edit.setPlainText(desc)
        
        # Generate tags with SEO keywords and channel preset
        seo_type = self._get_seo_type()
        tags = self.yt_gen.generate_tags(theme, seo_type)
        
        # Add channel preset tags if available
        channel_name = self.channel_combo.currentText()
        if channel_name in self.CHANNEL_PRESETS:
            preset = self.CHANNEL_PRESETS[channel_name]
            if preset.get('tags'):
                preset_tags = preset['tags']
                # Combine with generated tags, avoiding duplicates
                existing = set(t.strip().lower() for t in tags.split(","))
                new_tags = [t.strip() for t in preset_tags.split(",") if t.strip().lower() not in existing]
                if new_tags:
                    tags = f"{tags}, {', '.join(new_tags)}"
        
        self.tags_edit.setPlainText(tags)
        
        self._update_tags_count()
        
    def _update_tags_count(self):
        """Update tags character count"""
        text = self.tags_edit.toPlainText()
        count = len(text)
        color = Colors.METER_GREEN if count <= 500 else Colors.METER_RED
        self.tags_header.setText(f"🏷️ Tags ({count}/500):")
        self.tags_header.setStyleSheet(f"color: {color}; font-size: 12px;")
        
    def _copy_text(self, text: str):
        """Copy text to clipboard"""
        clipboard = QApplication.clipboard()
        clipboard.setText(text)
        QMessageBox.information(self, "Copied", "Copied to clipboard!")
        
    def _export_txt(self):
        """Export all metadata to .txt file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export YouTube Metadata", 
            f"youtube_vol{self.volume_spin.value()}.txt",
            "Text Files (*.txt)"
        )
        
        if file_path:
            content = f"""=== YOUTUBE METADATA ===
Generated by LongPlay Studio V4.31

=== TITLE ===
{self.title_edit.text()}

=== DESCRIPTION ===
{self.desc_edit.toPlainText()}

=== TAGS ===
{self.tags_edit.toPlainText()}
"""
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
            QMessageBox.information(self, "Exported", f"Saved to: {file_path}")
