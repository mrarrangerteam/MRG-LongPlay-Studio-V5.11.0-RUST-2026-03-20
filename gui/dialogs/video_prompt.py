"""
Video prompt generator dialog — Midjourney-style prompts for meta.ai.

Classes:
    VideoPromptDialog — Dialog for generating video prompts
"""

import os
from typing import List

from gui.utils.compat import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QLineEdit, QTextEdit,
    QMessageBox, QFileDialog, QApplication, Qt,
)
from gui.styles import Colors

# Import Video Prompt Generator
try:
    from video_prompt_generator import VideoPromptGenerator, MIDJOURNEY_STYLES
except ImportError:
    VideoPromptGenerator = None
    MIDJOURNEY_STYLES = {}


class VideoPromptDialog(QDialog):
    """Dialog for generating Midjourney-style video prompts for meta.ai"""
    
    def __init__(self, video_files: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🎬 Video Prompt Generator - Midjourney Style")
        self.setMinimumSize(900, 700)
        self.setStyleSheet(f"""
            QDialog {{
                background: {Colors.BG_PRIMARY};
            }}
        """)
        
        self.video_files = video_files
        self.generator = VideoPromptGenerator() if VideoPromptGenerator else None
        self.current_analysis = None
        
        self._setup_ui()
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Header
        header = QLabel("🎬 Video Prompt Generator - Midjourney Style for meta.ai")
        header.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; font-size: 18px; font-weight: bold;")
        layout.addWidget(header)
        
        # Description
        desc = QLabel("Generate AI video prompts from your video files. Supports meta.ai and Midjourney styles.")
        desc.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 12px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        # Video selection
        video_layout = QHBoxLayout()
        video_label = QLabel("📹 Select Video:")
        video_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        video_layout.addWidget(video_label)
        
        self.video_combo = QComboBox()
        for vf in self.video_files:
            self.video_combo.addItem(vf.name, vf.path)
        self.video_combo.setStyleSheet(f"""
            QComboBox {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 6px;
                padding: 8px;
                min-width: 300px;
            }}
        """)
        self.video_combo.currentIndexChanged.connect(self._on_video_changed)
        video_layout.addWidget(self.video_combo, 1)
        
        analyze_btn = QPushButton("🔍 Analyze")
        analyze_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.ACCENT};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 20px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: {Colors.ACCENT_DIM};
            }}
        """)
        analyze_btn.clicked.connect(self._analyze_video)
        video_layout.addWidget(analyze_btn)
        layout.addLayout(video_layout)
        
        # Analysis info
        self.analysis_info = QLabel("Select a video and click Analyze to begin")
        self.analysis_info.setStyleSheet(f"""
            QLabel {{
                color: {Colors.TEXT_SECONDARY};
                background: {Colors.BG_SECONDARY};
                border-radius: 8px;
                padding: 10px;
            }}
        """)
        layout.addWidget(self.analysis_info)
        
        # Style selection
        style_layout = QHBoxLayout()
        style_label = QLabel("🎨 Prompt Style:")
        style_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        style_layout.addWidget(style_label)
        
        self.style_combo = QComboBox()
        styles = list(MIDJOURNEY_STYLES.keys()) if MIDJOURNEY_STYLES else [
            "cinematic", "anime", "documentary", "music_video", 
            "abstract", "lofi", "vaporwave", "nature"
        ]
        self.style_combo.addItems([s.replace("_", " ").title() for s in styles])
        self.style_combo.setStyleSheet(self.video_combo.styleSheet())
        self.style_combo.currentIndexChanged.connect(self._regenerate_prompt)
        style_layout.addWidget(self.style_combo)
        
        style_layout.addStretch()
        
        # Custom subject
        subject_label = QLabel("📝 Custom Subject:")
        subject_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        style_layout.addWidget(subject_label)
        
        self.subject_edit = QLineEdit()
        self.subject_edit.setPlaceholderText("e.g., 'a lone figure walking through neon-lit streets'")
        self.subject_edit.setStyleSheet(f"""
            QLineEdit {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 6px;
                padding: 8px;
                min-width: 250px;
            }}
        """)
        self.subject_edit.textChanged.connect(self._regenerate_prompt)
        style_layout.addWidget(self.subject_edit)
        layout.addLayout(style_layout)
        
        # Generated prompt
        prompt_header = QLabel("✨ Generated Prompt (Midjourney Style):")
        prompt_header.setStyleSheet(f"color: {Colors.ACCENT}; font-size: 14px; font-weight: bold;")
        layout.addWidget(prompt_header)
        
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setPlaceholderText("Generated prompt will appear here...")
        self.prompt_edit.setStyleSheet(f"""
            QTextEdit {{
                background: {Colors.BG_SECONDARY};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 8px;
                padding: 15px;
                font-size: 13px;
            }}
        """)
        self.prompt_edit.setMinimumHeight(100)
        layout.addWidget(self.prompt_edit)
        
        # Meta.ai prompt
        meta_header = QLabel("🤖 meta.ai Video Prompt:")
        meta_header.setStyleSheet(f"color: {Colors.VIDEO_COLOR}; font-size: 14px; font-weight: bold;")
        layout.addWidget(meta_header)
        
        self.meta_prompt_edit = QTextEdit()
        self.meta_prompt_edit.setPlaceholderText("meta.ai optimized prompt will appear here...")
        self.meta_prompt_edit.setStyleSheet(self.prompt_edit.styleSheet())
        self.meta_prompt_edit.setMinimumHeight(80)
        layout.addWidget(self.meta_prompt_edit)
        
        # All styles preview
        all_styles_header = QLabel("📋 All Style Variations:")
        all_styles_header.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 12px;")
        layout.addWidget(all_styles_header)
        
        self.all_styles_edit = QTextEdit()
        self.all_styles_edit.setReadOnly(True)
        self.all_styles_edit.setStyleSheet(f"""
            QTextEdit {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_SECONDARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 8px;
                padding: 10px;
                font-size: 11px;
            }}
        """)
        self.all_styles_edit.setMaximumHeight(150)
        layout.addWidget(self.all_styles_edit)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        copy_mj_btn = QPushButton("📋 Copy Midjourney Prompt")
        copy_mj_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.ACCENT};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 25px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: {Colors.ACCENT_DIM};
            }}
        """)
        copy_mj_btn.clicked.connect(self._copy_mj_prompt)
        btn_layout.addWidget(copy_mj_btn)
        
        copy_meta_btn = QPushButton("📋 Copy meta.ai Prompt")
        copy_meta_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.VIDEO_COLOR};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 25px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: #3A80C9;
            }}
        """)
        copy_meta_btn.clicked.connect(self._copy_meta_prompt)
        btn_layout.addWidget(copy_meta_btn)
        
        export_btn = QPushButton("📤 Export All")
        export_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.METER_GREEN};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 25px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: #3D8B40;
            }}
        """)
        export_btn.clicked.connect(self._export_all)
        btn_layout.addWidget(export_btn)
        
        btn_layout.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 8px;
                padding: 12px 25px;
            }}
            QPushButton:hover {{
                background: {Colors.BORDER};
            }}
        """)
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)
        
        layout.addLayout(btn_layout)
        
        # Auto-analyze first video if available
        if self.video_files:
            self._analyze_video()
    
    def _on_video_changed(self, index: int):
        """Handle video selection change"""
        self.current_analysis = None
        self.analysis_info.setText("Click Analyze to process the selected video")
        
    def _analyze_video(self):
        """Analyze the selected video"""
        if not self.generator:
            QMessageBox.warning(self, "Error", "Video Prompt Generator not available")
            return
            
        if self.video_combo.count() == 0:
            QMessageBox.warning(self, "No Video", "Please add video files first")
            return
            
        video_path = self.video_combo.currentData()
        if not video_path:
            return
            
        self.analysis_info.setText("🔄 Analyzing video...")
        QApplication.processEvents()
        
        try:
            self.current_analysis = self.generator.analyze_video(video_path)
            
            # Update analysis info
            info_text = f"""
📹 <b>{self.current_analysis.filename}</b><br>
⏱️ Duration: {self.current_analysis.duration_sec:.1f}s | 
📐 Resolution: {self.current_analysis.width}x{self.current_analysis.height} | 
🎬 FPS: {self.current_analysis.fps:.1f}<br>
💡 Brightness: {self.current_analysis.brightness} | 
🎭 Motion: {self.current_analysis.motion_level} | 
🌆 Scene: {self.current_analysis.scene_type}<br>
🎨 Colors: {', '.join(self.current_analysis.dominant_colors)}
            """
            self.analysis_info.setText(info_text.strip())
            
            # Generate prompts
            self._regenerate_prompt()
            
        except Exception as e:
            self.analysis_info.setText(f"❌ Analysis failed: {str(e)}")
    
    def _regenerate_prompt(self):
        """Regenerate prompt with current settings"""
        if not self.current_analysis or not self.generator:
            return
            
        # Get selected style
        style_index = self.style_combo.currentIndex()
        styles = list(MIDJOURNEY_STYLES.keys()) if MIDJOURNEY_STYLES else [
            "cinematic", "anime", "documentary", "music_video", 
            "abstract", "lofi", "vaporwave", "nature"
        ]
        style = styles[style_index] if style_index < len(styles) else "cinematic"
        
        # Get custom subject
        custom_subject = self.subject_edit.text().strip()
        
        # Generate Midjourney prompt
        mj_prompt = self.generator.generate_prompt(
            self.current_analysis, style, custom_subject
        )
        self.prompt_edit.setPlainText(mj_prompt)
        
        # Generate meta.ai prompt
        meta_prompt = self.generator.generate_meta_ai_prompt(
            self.current_analysis, 
            duration_sec=5,
            custom_description=custom_subject
        )
        self.meta_prompt_edit.setPlainText(meta_prompt)
        
        # Generate all styles
        all_prompts = self.generator.generate_all_styles(
            self.current_analysis, custom_subject
        )
        all_text = ""
        for style_name, prompt in all_prompts.items():
            all_text += f"[{style_name.upper()}]\n{prompt}\n\n"
        self.all_styles_edit.setPlainText(all_text)
    
    def _copy_mj_prompt(self):
        """Copy Midjourney prompt to clipboard"""
        clipboard = QApplication.clipboard()
        clipboard.setText(self.prompt_edit.toPlainText())
        QMessageBox.information(self, "Copied", "Midjourney prompt copied to clipboard!")
    
    def _copy_meta_prompt(self):
        """Copy meta.ai prompt to clipboard"""
        clipboard = QApplication.clipboard()
        clipboard.setText(self.meta_prompt_edit.toPlainText())
        QMessageBox.information(self, "Copied", "meta.ai prompt copied to clipboard!")
    
    def _export_all(self):
        """Export all prompts to file"""
        if not self.current_analysis:
            QMessageBox.warning(self, "No Analysis", "Please analyze a video first")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Video Prompts",
            f"{self.current_analysis.filename}_prompts.txt",
            "Text Files (*.txt);;Markdown Files (*.md)"
        )
        
        if file_path:
            content = f"""# Video Prompt Generator - Midjourney Style
# Generated by LongPlay Studio V4.31

## Video: {self.current_analysis.filename}
- Duration: {self.current_analysis.duration_sec:.1f}s
- Resolution: {self.current_analysis.width}x{self.current_analysis.height}
- FPS: {self.current_analysis.fps:.1f}
- Brightness: {self.current_analysis.brightness}
- Motion: {self.current_analysis.motion_level}
- Scene: {self.current_analysis.scene_type}
- Colors: {', '.join(self.current_analysis.dominant_colors)}

## Midjourney Prompt
```
{self.prompt_edit.toPlainText()}
```

## meta.ai Prompt
```
{self.meta_prompt_edit.toPlainText()}
```

## All Style Variations
{self.all_styles_edit.toPlainText()}
"""
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            QMessageBox.information(self, "Exported", f"Saved to: {file_path}")

