"""
Dual-Mode Configuration Panel for LaserCom Digital Twin.
Provides a "Visual Editor" (Form-based) and an "Advanced (JSON)" tab.
"""

import json
import re
import os
from pathlib import Path
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPlainTextEdit, 
    QPushButton, QProgressBar, QMessageBox, QLabel,
    QTabWidget, QScrollArea, QGroupBox, QFormLayout, QLineEdit
)

class DualConfigWidget(QWidget):
    """
    Refactored configuration panel with Visual and JSON editing modes.
    """
    start_requested = Signal(dict)
    stop_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._field_map = {}  # Maps nested keys (tuple) to QLineEdit widgets
        self._setup_ui()
        self._load_and_fill_all()

    def _setup_ui(self):
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(5, 5, 5, 5)
        self.main_layout.setSpacing(10)

        # Tab Widget
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #333; top: -1px; background: #1E1E1E; }
            QTabBar::tab { background: #2D2D2D; color: #AAA; padding: 10px 20px; border: 1px solid #333; border-bottom: none; }
            QTabBar::tab:selected { background: #1E1E1E; color: #FFF; border-bottom: 2px solid #007ACC; }
        """)

        # --- Tab 1: Visual Editor ---
        self.visual_tab = QWidget()
        visual_layout = QVBoxLayout(self.visual_tab)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setStyleSheet("background: transparent;")
        
        self.form_container = QWidget()
        self.form_layout = QVBoxLayout(self.form_container)
        self.form_layout.setSpacing(15)
        
        # Create sections exactly as requested
        self.sections = {
            "Timing & Execution": ["dt_sim", "dt_fine", "dt_coarse", "dt_encoder", "dt_gyro", "dt_qpd", "log_period", "enable_visualization", "enable_plotting", "real_time_factor", "viewer_fps", "seed"],
            "Target Trajectory": ["target_az", "target_el", "target_enabled", "target_type", "target_amplitude", "target_period", "target_reachangle"],
            "Control Strategy": ["use_feedback_linearization", "use_direct_state_feedback"],
            "Dynamics & Plant": ["dynamics_config", "motor_config", "fsm_config"],
            "Sensors": ["encoder_config", "gyro_config", "qpd_config"],
            "Estimator": ["estimator_config"],
            "Controllers": ["coarse_controller_config", "feedback_linearization_config", "fsm_controller_config", "ndob_config"],
            "Optics & Environment": ["optics_config", "frame_config", "vibration_enabled", "vibration_config", "environmental_disturbance_enabled", "environmental_disturbance_config"]
        }
        
        self.group_boxes = {}
        for section_name in self.sections.keys():
            group = QGroupBox(section_name)
            group.setStyleSheet("""
                QGroupBox { font-weight: bold; color: #4FC1FF; border: 1px solid #333; border-radius: 4px; margin-top: 15px; padding-top: 10px; }
                QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            """)
            group_layout = QFormLayout(group)
            group_layout.setLabelAlignment(Qt.AlignRight)
            group_layout.setHorizontalSpacing(10)
            self.group_boxes[section_name] = group_layout
            self.form_layout.addWidget(group)
            
        scroll.setWidget(self.form_container)
        visual_layout.addWidget(scroll)
        
        # --- Tab 2: Advanced (JSON) ---
        self.json_tab = QWidget()
        json_layout = QVBoxLayout(self.json_tab)
        self.editor = QPlainTextEdit()
        font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        font.setPointSize(10)
        self.editor.setFont(font)
        self.editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.editor.setStyleSheet("""
            QPlainTextEdit { background-color: #1E1E1E; color: #D4D4D4; border: none; padding: 5px; }
        """)
        json_layout.addWidget(self.editor)

        self.tabs.addTab(self.visual_tab, "Visual Editor")
        self.tabs.addTab(self.json_tab, "Advanced (JSON)")
        self.main_layout.addWidget(self.tabs)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setFormat("Ready")
        self.main_layout.addWidget(self.progress_bar)

        # Control Buttons
        btn_layout = QHBoxLayout()
        self.btn_simulate = QPushButton("▶  RUN SIMULATION")
        self.btn_simulate.setMinimumHeight(45)
        self.btn_simulate.clicked.connect(self.on_simulate_clicked)
        
        self.btn_stop = QPushButton("■  STOP")
        self.btn_stop.setMinimumHeight(45)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_requested)
        
        btn_layout.addWidget(self.btn_simulate, 2)
        btn_layout.addWidget(self.btn_stop, 1)
        self.main_layout.addLayout(btn_layout)

    def _load_and_fill_all(self):
        """Loads config.jsonc and pre-fills both tabs."""
        try:
            base_dir = Path(__file__).resolve().parent.parent.parent
            path = base_dir / "data" / "config.jsonc"
            
            if not path.exists():
                path = Path("lasercom_digital_twin/data/config.jsonc")

            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    raw_text = f.read()
                
                # Fill JSON tab
                self.editor.setPlainText(raw_text)
                
                # Parse for Visual tab
                stripped = re.sub(r'//.*', '', raw_text)
                config_data = json.loads(stripped)
                self._fill_visual_form(config_data)
        except Exception as e:
            QMessageBox.warning(self, "Load Error", f"Could not load config: {e}")

    def _fill_visual_form(self, data):
        """Iterates through config sections and populates QLineEdits."""
        for section_name, top_keys in self.sections.items():
            group_layout = self.group_boxes[section_name]
            for key in top_keys:
                if key in data:
                    self._add_field_or_subgroup(group_layout, key, data[key], (key,))

    def _add_field_or_subgroup(self, layout, key, value, path_tuple):
        """Recursively adds fields or nested sections to the form."""
        if isinstance(value, dict):
            # Create a sub-layout or just add rows with indented labels
            for subkey, subval in value.items():
                new_path = path_tuple + (subkey,)
                # Use a separator for the label to show hierarchy
                label_text = "  " * (len(path_tuple) - 1) + f"↳ {subkey}" if len(path_tuple) > 1 else subkey
                self._add_field_or_subgroup(layout, label_text, subval, new_path)
        else:
            le = QLineEdit()
            # Use json.dumps to get the string representation (true, [400, 800], etc.)
            val_str = json.dumps(value)
            le.setText(val_str)
            le.setStyleSheet("""
                QLineEdit { 
                    background: #2D2D2D; 
                    color: #EEE; 
                    border: 1px solid #444; 
                    padding: 4px; 
                    border-radius: 2px;
                    font-family: 'Consolas', 'Courier New', monospace;
                }
                QLineEdit:focus { border: 1px solid #007ACC; background: #333; }
            """)
            layout.addRow(key, le)
            self._field_map[path_tuple] = le

    def _cast_value(self, text):
        """Helper to cast QLineEdit strings back to Python types matching requested logic."""
        text = text.strip()
        if not text: return None
        
        # We use json.loads for booleans, lists, and numbers as requested.
        # json.loads is case-sensitive for true/false, so we lowercase them if they match.
        lower_text = text.lower()
        if lower_text in ("true", "false", "null"):
            return json.loads(lower_text)
            
        try:
            # This handles numbers and lists/arrays like [400.0, 800.0]
            return json.loads(text)
        except json.JSONDecodeError:
            # Fallback to string if it's not a valid JSON primitive (like "constant")
            # If it's a string literal in the editor, users might type: "constant"
            # If they type just: constant, json.loads fails and we return "constant"
            return text

    def get_simulation_config(self) -> dict:
        """
        Scrapes the current UI state (from active tab) and returns the config dict.
        Allows menu shortcuts (F5) to work.
        """
        if self.tabs.currentIndex() == 1: # Advanced (JSON)
            raw_text = self.editor.toPlainText()
            stripped = re.sub(r'//.*', '', raw_text)
            try:
                config_dict = json.loads(stripped)
            except json.JSONDecodeError as e:
                # We can't easily return a partial dict, so we raise or return {}
                # For safety with shortcuts, we'll try to show the error
                error_msg = f"JSON Error: {e.msg} at line {e.lineno}"
                QMessageBox.critical(self, "Validation Failed", error_msg)
                return {}
        else: # Visual Editor
            config_dict = {}
            for path, line_edit in self._field_map.items():
                val_str = line_edit.text()
                try:
                    val = self._cast_value(val_str)
                    
                    # Nesting logic
                    d = config_dict
                    for part in path[:-1]:
                        if part not in d: d[part] = {}
                        d = d[part]
                    d[path[-1]] = val
                except Exception as e:
                    QMessageBox.critical(self, "Type Error", f"Field '{'.'.join(path)}' has invalid value: {val_str}")
                    return {}
        
        # Inject _gui_duration if missing (SimulationController handles it)
        # We manually check the 'dt_sim' or similar if they are editing them
        if "duration" in config_dict:
            config_dict["_gui_duration"] = config_dict["duration"]
        
        return config_dict

    @Slot()
    def on_simulate_clicked(self):
        """Validates and emits the current configuration."""
        config_dict = self.get_simulation_config()
        if not config_dict:
            return  # Error already shown in get_simulation_config

        # Success: Save and Emit
        try:
            with open("simulation_config.json", 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=4)
            self.start_requested.emit(config_dict)
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save simulation_config.json: {e}")

    def set_running_state(self, running):
        self.btn_simulate.setEnabled(not running)
        self.btn_stop.setEnabled(running)
        self.tabs.setEnabled(not running)

    def set_progress(self, pct):
        self.progress_bar.setValue(pct)
        self.progress_bar.setFormat(f"Running: {pct}%")
