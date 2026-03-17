"""
Hook extractor dialog — extract hook sections from audio using waveform analysis.

Classes:
    HookExtractorDialog — Dialog for extracting hooks from audio files
"""

import os
from typing import List

from gui.utils.compat import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QProgressBar, QListWidget, QListWidgetItem,
    QMessageBox, QFileDialog, QApplication, Qt,
)
from gui.styles import Colors

# Import Hook Extractor
try:
    from hook_extractor import HookExtractor, HookResult
except ImportError:
    HookExtractor = None
    HookResult = None


class HookExtractorDialog(QDialog):
    """Dialog for extracting hook sections from audio files using waveform analysis"""
    
    def __init__(self, audio_files: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🎵 Hook Extractor - Audio Waveform Analysis")
        self.setMinimumSize(900, 700)
        self.setStyleSheet(f"""
            QDialog {{
                background: {Colors.BG_PRIMARY};
            }}
        """)
        
        self.audio_files = audio_files[:20]  # Limit to 20 files
        self.extractor = HookExtractor(hook_duration=30.0) if HookExtractor else None
        self.results: List = []
        
        self._setup_ui()
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Header
        header = QLabel("🎵 Hook Extractor - Audio Waveform Analysis")
        header.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; font-size: 18px; font-weight: bold;")
        layout.addWidget(header)
        
        # Description
        desc = QLabel(f"Extract hook sections from up to 20 audio files using energy analysis and peak detection. ({len(self.audio_files)} files loaded)")
        desc.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 12px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        # Settings row
        settings_layout = QHBoxLayout()
        
        # Hook duration
        duration_label = QLabel("⏱️ Hook Duration:")
        duration_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        settings_layout.addWidget(duration_label)
        
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(10, 60)
        self.duration_spin.setValue(30)
        self.duration_spin.setSuffix(" sec")
        self.duration_spin.setStyleSheet(f"""
            QSpinBox {{
                background: {Colors.BG_TERTIARY};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 6px;
                padding: 8px;
                min-width: 80px;
            }}
        """)
        settings_layout.addWidget(self.duration_spin)
        
        settings_layout.addStretch()
        
        # Analyze button
        analyze_btn = QPushButton("🔍 Analyze All")
        analyze_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.ACCENT};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 25px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: {Colors.ACCENT_DIM};
            }}
        """)
        analyze_btn.clicked.connect(self._analyze_all)
        settings_layout.addWidget(analyze_btn)
        
        # Extract button
        extract_btn = QPushButton("✂️ Extract Hooks")
        extract_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.METER_GREEN};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 25px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: #3D8B40;
            }}
        """)
        extract_btn.clicked.connect(self._extract_hooks)
        settings_layout.addWidget(extract_btn)
        
        layout.addLayout(settings_layout)
        
        # Progress bar
        self.progress = QProgressBar()
        self.progress.setStyleSheet(f"""
            QProgressBar {{
                background: {Colors.BG_TERTIARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 6px;
                height: 20px;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background: {Colors.ACCENT};
                border-radius: 5px;
            }}
        """)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        
        # Results table header
        results_header = QLabel("📊 Analysis Results:")
        results_header.setStyleSheet(f"color: {Colors.ACCENT}; font-size: 14px; font-weight: bold;")
        layout.addWidget(results_header)
        
        # Results list
        self.results_list = QListWidget()
        self.results_list.setStyleSheet(f"""
            QListWidget {{
                background: {Colors.BG_SECONDARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 8px;
                padding: 10px;
            }}
            QListWidget::item {{
                background: {Colors.BG_TERTIARY};
                border-radius: 6px;
                padding: 10px;
                margin: 3px 0;
                color: {Colors.TEXT_PRIMARY};
            }}
            QListWidget::item:selected {{
                background: {Colors.ACCENT};
            }}
        """)
        layout.addWidget(self.results_list, 1)
        
        # Summary
        self.summary_label = QLabel("Click 'Analyze All' to start hook detection")
        self.summary_label.setStyleSheet(f"""
            QLabel {{
                color: {Colors.TEXT_SECONDARY};
                background: {Colors.BG_TERTIARY};
                border-radius: 8px;
                padding: 15px;
            }}
        """)
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        export_btn = QPushButton("📤 Export Report")
        export_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Colors.VIDEO_COLOR};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 25px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: #e07b00;
            }}
        """)
        export_btn.clicked.connect(self._export_report)
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
    
    def _analyze_all(self):
        """Analyze all audio files for hooks"""
        if not self.extractor:
            QMessageBox.warning(self, "Error", "Hook Extractor not available")
            return
            
        if not self.audio_files:
            QMessageBox.warning(self, "No Audio", "Please add audio files first")
            return
        
        # Update hook duration
        self.extractor.hook_duration = self.duration_spin.value()
        
        # Show progress
        self.progress.setVisible(True)
        self.progress.setMaximum(len(self.audio_files))
        self.progress.setValue(0)
        self.results_list.clear()
        self.results = []
        
        # Analyze each file
        for i, af in enumerate(self.audio_files):
            self.progress.setValue(i)
            QApplication.processEvents()
            
            try:
                result = self.extractor.analyze_audio(af.path)
                self.results.append(result)
                
                # Add to list
                confidence_emoji = "🟢" if result.hook_confidence >= 0.7 else ("🟡" if result.hook_confidence >= 0.5 else "🔴")
                item_text = f"{confidence_emoji} {result.filename}\n"
                item_text += f"   ⏱️ Duration: {result.duration_sec:.1f}s | "
                item_text += f"🎵 Hook: {result.hook_time_str} | "
                item_text += f"🎯 Confidence: {result.hook_confidence:.0%}"
                
                item = QListWidgetItem(item_text)
                self.results_list.addItem(item)
                
            except Exception as e:
                item = QListWidgetItem(f"❌ {af.name} - Error: {str(e)}")
                self.results_list.addItem(item)
        
        self.progress.setValue(len(self.audio_files))
        self.progress.setVisible(False)
        
        # Update summary
        if self.results:
            avg_confidence = sum(r.hook_confidence for r in self.results) / len(self.results)
            high_conf = sum(1 for r in self.results if r.hook_confidence >= 0.7)
            
            summary = f"""
<b>✅ Analysis Complete!</b><br>
📊 Total files: {len(self.results)}<br>
🎯 Average confidence: {avg_confidence:.0%}<br>
🟢 High confidence (>70%): {high_conf} files<br>
<br>
Click 'Extract Hooks' to save hook sections as separate files.
            """
            self.summary_label.setText(summary.strip())
    
    def _extract_hooks(self):
        """Extract hook sections from analyzed files"""
        if not self.results:
            QMessageBox.warning(self, "No Analysis", "Please analyze files first")
            return
        
        # Ask for output directory
        output_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Directory for Hooks"
        )
        
        if not output_dir:
            return
        
        # Show progress
        self.progress.setVisible(True)
        self.progress.setMaximum(len(self.results))
        self.progress.setValue(0)
        
        extracted = 0
        for i, result in enumerate(self.results):
            self.progress.setValue(i)
            QApplication.processEvents()
            
            try:
                hook_path = self.extractor.extract_hook(result.file_path, output_dir)
                if hook_path:
                    extracted += 1
            except Exception as e:
                print(f"Extract error for {result.filename}: {e}")
        
        self.progress.setValue(len(self.results))
        self.progress.setVisible(False)
        
        QMessageBox.information(
            self, "Extraction Complete",
            f"✅ Extracted {extracted}/{len(self.results)} hooks to:\n{output_dir}"
        )
    
    def _export_report(self):
        """Export analysis report"""
        if not self.results:
            QMessageBox.warning(self, "No Results", "Please analyze files first")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Hook Analysis Report",
            "hook_analysis_report.md",
            "Markdown Files (*.md);;Text Files (*.txt)"
        )
        
        if file_path:
            content = "# Hook Extractor Report\n"
            content += "# Generated by LongPlay Studio V4.31\n\n"
            content += f"Total files analyzed: {len(self.results)}\n"
            content += f"Hook duration setting: {self.duration_spin.value()}s\n\n"
            
            for result in self.results:
                confidence_emoji = "🟢" if result.hook_confidence >= 0.7 else ("🟡" if result.hook_confidence >= 0.5 else "🔴")
                content += f"## {confidence_emoji} {result.filename}\n"
                content += f"- Full Duration: {result.duration_sec:.1f}s\n"
                content += f"- Hook Time: {result.hook_time_str}\n"
                content += f"- Hook Duration: {result.hook_duration_sec:.1f}s\n"
                content += f"- Confidence: {result.hook_confidence:.0%}\n"
                if result.hook_file_path:
                    content += f"- Extracted File: {result.hook_file_path}\n"
                content += "\n"
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            QMessageBox.information(self, "Exported", f"Report saved to: {file_path}")
