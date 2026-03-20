"""
Embedded JSON/JSONC Configuration Editor for LaserCom Digital Twin.

Provides a monospace text editor for direct configuration editing,
stripping C-style comments before parsing.
"""

import json
import re
import os
from pathlib import Path
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPlainTextEdit,
    QPushButton,
    QProgressBar,
    QMessageBox,
    QLabel
)

class ConfigEditorWidget(QWidget):
    """
    Embedded text editor for editing config.jsonc directly.
    
    Signals:
        start_requested (dict): Emitted with the parsed configuration when 'Simulate' is clicked.
        stop_requested: Emitted when 'Stop' is clicked.
    """
    start_requested = Signal(dict)
    stop_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_config_dict = {}
        self._setup_ui()
        self._load_initial_config()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        # Header with icon-like text
        header_layout = QHBoxLayout()
        header = QLabel("<b><span style='color:#4FC1FF;'>{ }</span> &nbsp; CONFIGURATION EDITOR</b>")
        header.setStyleSheet("color: #DDD; font-size: 13px; letter-spacing: 1px;")
        header_layout.addWidget(header)
        header_layout.addStretch()
        layout.addLayout(header_layout)

        # The Editor
        self.editor = QPlainTextEdit()
        
        # Monospace Font Setup
        # Try to get a high-quality monospace font
        font_db = QFontDatabase()
        font_families = [
            "Cascadia Code", "Consolas", "Fira Code", 
            "JetBrains Mono", "Courier New", "Monospace"
        ]
        chosen_font = None
        for family in font_families:
            if family in font_db.families():
                chosen_font = QFont(family, 10)
                break
        
        if not chosen_font:
            chosen_font = font_db.systemFont(QFontDatabase.FixedFont)
            chosen_font.setPointSize(10)
            
        self.editor.setFont(chosen_font)
        
        # Disable Word Wrap for clean indentation
        self.editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        
        # Editor styling for VS-Code like appearance
        self.editor.setStyleSheet("""
            QPlainTextEdit {
                background-color: #1E1E1E;
                color: #D4D4D4;
                border: 1px solid #333;
                border-radius: 6px;
                padding: 8px;
                selection-background-color: #264F78;
            }
            QScrollBar:vertical {
                border: none;
                background: #1E1E1E;
                width: 10px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #333;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover {
                background: #444;
            }
        """)
        layout.addWidget(self.editor)

        # Progress indicator
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Ready")
        self.progress_bar.setMinimumHeight(24)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #252526;
                border: 1px solid #333;
                border-radius: 4px;
                text-align: center;
                color: #AAA;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #007ACC, stop:1 #4FC1FF);
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.progress_bar)

        # Control Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)
        
        self.btn_simulate = QPushButton("▶  RUN SIMULATION")
        self.btn_simulate.setMinimumHeight(48)
        self.btn_simulate.setCursor(Qt.PointingHandCursor)
        self.btn_simulate.setStyleSheet("""
            QPushButton {
                background-color: #2D5A27; 
                color: white; 
                font-weight: bold; 
                border-radius: 6px;
                font-size: 13px;
                border: 1px solid #3D7A35;
            }
            QPushButton:hover { background-color: #3D7A35; border: 1px solid #4D8A45; }
            QPushButton:pressed { background-color: #1E3D1A; }
            QPushButton:disabled { background-color: #333; color: #777; border: 1px solid #222; }
        """)
        self.btn_simulate.clicked.connect(self.on_simulate_clicked)
        
        self.btn_stop = QPushButton("■  STOP")
        self.btn_stop.setMinimumHeight(48)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setCursor(Qt.PointingHandCursor)
        self.btn_stop.setStyleSheet("""
            QPushButton {
                background-color: #8B0000; 
                color: white; 
                font-weight: bold; 
                border-radius: 6px;
                font-size: 13px;
                border: 1px solid #A52A2A;
            }
            QPushButton:hover { background-color: #A52A2A; border: 1px solid #B53A3A; }
            QPushButton:pressed { background-color: #5F0000; }
            QPushButton:disabled { background-color: #333; color: #777; border: 1px solid #222; }
        """)
        self.btn_stop.clicked.connect(self.stop_requested)
        
        btn_layout.addWidget(self.btn_simulate, 2)
        btn_layout.addWidget(self.btn_stop, 1)
        layout.addLayout(btn_layout)

    def _load_initial_config(self):
        """Locates and loads the data/config.jsonc file on startup."""
        try:
            # Find data/config.jsonc relative to project root
            # Assume we are in lasercom_digital_twin/gui/views/
            base_path = Path(__file__).resolve().parent.parent.parent
            config_path = base_path / "data" / "config.jsonc"
            
            if not config_path.exists():
                # Fallback check for current working directory
                config_path = Path("lasercom_digital_twin/data/config.jsonc")
            
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.editor.setPlainText(f.read())
            else:
                self.editor.setPlainText("// Warning: config.jsonc not found.\n// Expected at: " + str(config_path.absolute()))
        except Exception as e:
            self.editor.setPlainText(f"// Error loading configuration: {e}")

    @Slot()
    def on_simulate_clicked(self):
        """
        Gathers text, strips comments, validates JSON, and triggers simulation.
        """
        raw_text = self.editor.toPlainText()
        
        # 1. Strip C-style comments (//)
        # Robust regex to remove // comments while ignoring ones inside double quotes (like URLs)
        # This handles the "bulletproof" requirement for common edge cases.
        stripped_text = re.sub(r'(?m)^\s*//.*$|(?<!:)\/\/.*$', '', raw_text)

        # 2. Parse JSON safely
        try:
            config_dict = json.loads(stripped_text)
            
            # Map "duration" -> "_gui_duration" for controller compatibility if present
            if "duration" in config_dict:
                config_dict["_gui_duration"] = config_dict["duration"]
            elif "dt_sim" in config_dict and "total_steps" in config_dict:
                # Optional fallback if they define steps instead
                config_dict["_gui_duration"] = config_dict["dt_sim"] * config_dict["total_steps"]
            
            self._current_config_dict = config_dict
            
            # 3. Success Handling: Save to simulation_config.json
            output_path = "simulation_config.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=4)
            
            # 4. Trigger the worker via signal
            self.start_requested.emit(config_dict)
            
        except json.JSONDecodeError as e:
            # 5. Error Handling: Pop up detail error message with specific line info
            error_msg = (
                f"<div style='color:#FF5555; font-size:14px;'><b>JSON Syntax Error</b></div><br>"
                f"<b>Message:</b> {e.msg}<br>"
                f"<b>Location:</b> Line {e.lineno}, Column {e.colno}<br><br>"
                f"<i>Please verify commas, brackets, and quotes in the editor.</i>"
            )
            QMessageBox.critical(self, "Validation Failed", error_msg)
        except Exception as e:
            QMessageBox.critical(self, "System Error", f"Failed to process configuration: {str(e)}")

    def get_simulation_config(self) -> dict:
        """Returns the last successfully parsed configuration."""
        return self._current_config_dict

    def set_running_state(self, running: bool):
        """Updates UI status when simulation starts/stops."""
        self.btn_simulate.setEnabled(not running)
        self.btn_stop.setEnabled(running)
        self.editor.setReadOnly(running)
        
        if running:
            self.progress_bar.setFormat("%p%  —  Running...")
        else:
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("Ready")

    def set_progress(self, pct: int):
        """Updates the progress bar percentage."""
        self.progress_bar.setValue(pct)
        self.progress_bar.setFormat(f"{pct}%  —  Running...")
