"""
Configuration Editor — left-dock configuration panel for LaserCom Digital Twin.

Replaces the hardcoded layout with a deeply nested dictionary mapped layout.
Automatically loads `data/config.jsonc`, generates a tabbed UI,
and saves changes to `simulation_config.json`.
"""

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QWidget,
    QScrollArea,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QGroupBox,
    QLabel,
    QComboBox,
    QCheckBox,
    QDoubleSpinBox,
    QSpinBox,
    QPushButton,
    QProgressBar,
    QSizePolicy,
    QTabWidget,
    QLineEdit
)
import json
import re
import os

def load_jsonc(filepath: str) -> dict:
    """Loads a JSON file, stripping C-style (//) comments first."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Regular expression to remove // comments, ignoring // inside strings
    comment_re = re.compile(
        r'(^)?[^\S\n]*/(?:\*(.*?)\*/[^\S\n]*|/[^\n]*)($)?',
        re.DOTALL | re.MULTILINE
    )
    content_no_comments = comment_re.sub('', content)
    return json.loads(content_no_comments)


class ControlPanelWidget(QWidget):
    """
    Professional Configuration Panel (ConfigurationPanelWidget) for the GUI.
    """
    start_requested = Signal(dict)
    stop_requested  = Signal()
    get_config = None

    def __init__(self, parent=None):
        super().__init__(parent)
        self.widget_registry = []
        
        # Load master config
        self._load_master_config()

        self._build_ui()
        self._connect_signals()
        self.get_config = self.get_simulation_config

    def _load_master_config(self):
        # Look for the config file relative to this script
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        config_path = os.path.join(base_dir, 'data', 'config.jsonc')
        
        if not os.path.exists(config_path):
            # Fallback path if running from project root
            config_path = os.path.join('lasercom_digital_twin', 'data', 'config.jsonc')
            
        try:
            self.master_config = load_jsonc(config_path)
            print(f"Loaded configuration from: {config_path}")
        except Exception as e:
            print(f"Error loading {config_path}: {e}")
            self.master_config = {}

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Tabs for grouping
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # Build individual tabs
        self.tabs.addTab(self._create_setup_tab(), "1. Setup")
        self.tabs.addTab(self._create_tab_from_groups(["dynamics_config", "motor_config", "fsm_config"], "Plant & Actuators"), "2. Plant")
        self.tabs.addTab(self._create_tab_from_groups(["qpd_config", "encoder_config", "gyro_config", "estimator_config"], "Sensors & EKF"), "3. Sensors")
        self.tabs.addTab(self._create_tab_from_groups(["coarse_controller_config", "feedback_linearization_config", "fsm_controller_config", "ndob_config"], "Controllers"), "4. Control")
        self.tabs.addTab(self._create_tab_from_groups(["environmental_disturbance_config"], "Disturbances"), "5. Disturbances")
        
        # Execution group at the bottom
        main_layout.addWidget(self._build_execution_group())

    def _create_setup_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        layout = QVBoxLayout(container)
        
        grp = QGroupBox("Setup & Targets")
        form = QFormLayout(grp)
        form.setLabelAlignment(Qt.AlignLeft)
        
        # Special ComboBox for target_type
        combo = QComboBox()
        combo.addItems(["constant", "square", "sine", "cosine", "hybridsig"])
        curr_val = self.master_config.get("target_type", "constant")
        combo.setCurrentText(curr_val)
        self.widget_registry.append({'path': ['target_type'], 'widget': combo, 'type': 'combo'})
        form.addRow("Target Type", combo)
        
        keys = ['dt_sim', 'target_az', 'target_el', 'target_amplitude', 'target_period', 'target_reachangle']
        self._build_form_for_keys(form, self.master_config, [], keys)
        
        layout.addWidget(grp)
        
        # Adding controller toggles to setup tab for quick access
        ctrl_grp = QGroupBox("Control Strategy")
        ctrl_form = QFormLayout(ctrl_grp)
        self._build_form_for_keys(ctrl_form, self.master_config, [], ["use_feedback_linearization", "use_direct_state_feedback"])
        layout.addWidget(ctrl_grp)
        
        layout.addStretch()
        scroll.setWidget(container)
        return scroll

    def _create_tab_from_groups(self, group_keys, title=""):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        layout = QVBoxLayout(container)
        
        for gkey in group_keys:
            if gkey in self.master_config:
                grp = QGroupBox(gkey.replace('_config', '').replace('_', ' ').title())
                form = QFormLayout(grp)
                sub_dict = self.master_config[gkey]
                if isinstance(sub_dict, dict):
                    self._build_dict_recursively(form, sub_dict, [gkey])
                else:
                    self._build_form_for_keys(form, self.master_config, [], [gkey])
                layout.addWidget(grp)
                
        layout.addStretch()
        scroll.setWidget(container)
        return scroll

    def _build_dict_recursively(self, form_layout, dict_ref, path_prefix):
        for k, val in dict_ref.items():
            if isinstance(val, dict):
                lbl = QLabel(f"<b style='color:#555;'>--- {k.replace('_', ' ').title()} ---</b>")
                form_layout.addRow(lbl)
                self._build_dict_recursively(form_layout, val, path_prefix + [k])
            else:
                self._build_form_for_keys(form_layout, dict_ref, path_prefix, [k])

    def _build_form_for_keys(self, form_layout, dict_ref, path_prefix, keys_to_include):
        for k in keys_to_include:
            if k not in dict_ref:
                continue
            val = dict_ref[k]
            label = k.replace('_', ' ').title()
            path = path_prefix + [k]
            
            if isinstance(val, bool):
                cb = QCheckBox()
                cb.setChecked(val)
                self.widget_registry.append({'path': path, 'widget': cb, 'type': bool})
                form_layout.addRow(label, cb)
            elif isinstance(val, int) and type(val) is not bool:
                sb = QSpinBox()
                sb.setRange(-999999, 999999)
                sb.setValue(val)
                self.widget_registry.append({'path': path, 'widget': sb, 'type': int})
                form_layout.addRow(label, sb)
            elif isinstance(val, float):
                sb = QDoubleSpinBox()
                sb.setRange(-1e9, 1e9)
                if abs(val) < 0.01 and val != 0.0:
                    sb.setDecimals(6)
                    sb.setSingleStep(1e-5)
                else:
                    sb.setDecimals(4)
                    sb.setSingleStep(0.1)
                sb.setValue(val)
                self.widget_registry.append({'path': path, 'widget': sb, 'type': float})
                form_layout.addRow(label, sb)
            elif isinstance(val, str):
                le = QLineEdit(val)
                self.widget_registry.append({'path': path, 'widget': le, 'type': str})
                form_layout.addRow(label, le)
            elif isinstance(val, list):
                # Assumes list of numbers
                str_val = ", ".join(str(x) for x in val)
                le = QLineEdit(str_val)
                self.widget_registry.append({'path': path, 'widget': le, 'type': list})
                form_layout.addRow(label, le)

    def _build_execution_group(self) -> QGroupBox:
        grp = QGroupBox("▶ Execution")
        layout = QVBoxLayout(grp)
        layout.setSpacing(10)
        
        # Duration
        dur_row = QHBoxLayout()
        dur_row.addWidget(QLabel("Duration [s]"))
        self.spin_duration = QDoubleSpinBox()
        self.spin_duration.setRange(0.1, 600.0)
        self.spin_duration.setValue(5.0)
        self.spin_duration.setToolTip("Total simulation wall-clock time")
        dur_row.addWidget(self.spin_duration)
        layout.addLayout(dur_row)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Ready")
        self.progress_bar.setMinimumHeight(22)
        layout.addWidget(self.progress_bar)

        # Buttons
        btn_row = QHBoxLayout()
        self.btn_run = QPushButton("▶ RUN SIMULATION")
        self.btn_run.setMinimumHeight(42)
        self.btn_run.setObjectName("btn_run")
        self.btn_stop = QPushButton("■ STOP")
        self.btn_stop.setMinimumHeight(42)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setObjectName("btn_stop")
        
        btn_row.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_stop)
        layout.addLayout(btn_row)
        
        return grp

    def _connect_signals(self):
        self.btn_run.clicked.connect(self.on_simulate_clicked)
        self.btn_stop.clicked.connect(self.stop_requested)

    @Slot()
    def on_simulate_clicked(self):
        """
        Scrapes UI state, updates master config, saves via json.dump,
        and emits start_requested.
        """
        # 1. Scrape UI
        for item in self.widget_registry:
            path = item['path']
            widget = item['widget']
            val_type = item['type']
            
            try:
                if val_type == float or val_type == int:
                    val = widget.value()
                elif val_type == bool:
                    val = widget.isChecked()
                elif val_type == str:
                    val = widget.text()
                elif val_type == 'combo':
                    val = widget.currentText()
                elif val_type == list:
                    val = [float(x.strip()) for x in widget.text().split(',')]
                
                # Update underlying tree
                ref = self.master_config
                for p in path[:-1]:
                    if p not in ref:
                        ref[p] = {}
                    ref = ref[p]
                ref[path[-1]] = val
            except Exception as e:
                print(f"Error reading field {path}: {e}")
        
        # Insert UI duration for runner backward compatibility 
        self.master_config['_gui_duration'] = self.spin_duration.value()
        
        # 2. Save directly to root directory
        out_path = os.path.join(os.getcwd(), 'simulation_config.json')
        try:
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(self.master_config, f, indent=4)
            print(f"Saved simulation_config.json to {out_path}")
        except Exception as e:
            print(f"Failed to write simulation_config.json: {e}")

        # 3. Emit run command
        self.start_requested.emit(self.master_config)

    def get_simulation_config(self) -> dict:
        return self.master_config

    def set_running_state(self, running: bool) -> None:
        """Disable/enable all inputs while simulation is active."""
        self.btn_run.setEnabled(not running)
        self.btn_stop.setEnabled(running)
        self.tabs.setEnabled(not running)
        self.spin_duration.setEnabled(not running)

        if running:
            self.progress_bar.setFormat("%p%  —  Running…")
        else:
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("Ready")

    def set_progress(self, pct: int) -> None:
        self.progress_bar.setValue(pct)
        self.progress_bar.setFormat(f"{pct}%  —  Running…")

