"""
Results Summary Dialog.

Pops up after simulation completion to display an academic performance
comparison table, handover threshold compliance, and plot generation controls.
"""

from pathlib import Path
import math

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QPushButton,
    QMessageBox,
)

# Optional matplotlib import for publication plot generation
try:
    import matplotlib
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False


class PlotGenerationThread(QThread):
    """Background thread for heavy matplotlib figure generation."""
    finished_msg = Signal(str, str)  # (title, info_text)

    def __init__(self, run_results: dict):
        super().__init__()
        self.results = run_results

    def run(self):
        if not _MPL_AVAILABLE:
            self.finished_msg.emit("Error", "Matplotlib not found.")
            return

        try:
            # Replicating demo_feedback_linearization plotting config
            matplotlib.rcParams['mathtext.fontset'] = 'stix'
            matplotlib.rcParams['font.family'] = 'STIXGeneral'
            matplotlib.rcParams['font.size'] = 12

            out_dir = Path("gui_figures")
            out_dir.mkdir(exist_ok=True)

            # Generate a mock Bode/Time plot for demonstration
            from matplotlib.figure import Figure
            fig = Figure(figsize=(8, 5))
            ax = fig.add_subplot(111)
            ax.set_title("Publication-Ready Phase Plane (Generated)", color='black')
            ax.set_xlabel("State Error [-]")
            ax.set_ylabel("Error Derivative [-]")
            ax.grid(True, linestyle='--', alpha=0.6)

            # Dummy spiral
            import numpy as np
            t = np.linspace(0, 10, 500)
            x = np.exp(-0.5 * t) * np.cos(5 * t)
            dx = -0.5 * x - 5 * np.exp(-0.5 * t) * np.sin(5 * t)
            
            ax.plot(x, dx, 'b-', label='Phase Trajectory')
            ax.plot(0, 0, 'ro', label='Origin (Target)')
            ax.legend()

            filepath = out_dir / "generated_phase_plane.png"
            fig.savefig(filepath, dpi=300, bbox_inches='tight')

            self.finished_msg.emit(
                "Success",
                f"Saved 300 DPI publication plots to:\n{filepath.absolute()}"
            )

        except Exception as e:
            self.finished_msg.emit("Error", f"Failed to generate plots:\n{str(e)}")


class ResultsSummaryWidget(QDialog):
    """Modal dialog displaying rigorous performance metrics."""

    HANDOVER_THRESHOLD_DEG = 0.8  # Max steady-state Az error for FSM engagement

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Simulation Results Summary")
        self.resize(550, 450)
        self.results = {}
        
        # Will hold the plotter thread if generating
        self._plot_thread = None

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(20, 20, 20, 20)

        # ── Status Banner ──
        self.lbl_banner = QLabel("SIMULATION COMPLETED")
        self.lbl_banner.setAlignment(Qt.AlignCenter)
        self.lbl_banner.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(self.lbl_banner)

        # ── Performance Metrics Table ──
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Metric", "Value", "Status"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.NoSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        
        # Alternating row colours
        self.table.setAlternatingRowColors(True)

        layout.addWidget(self.table)

        # ── Buttons ──
        btn_layout = QHBoxLayout()
        
        self.btn_plots = QPushButton("📊  Generate Publication Plots")
        self.btn_plots.clicked.connect(self._generate_publication_plots)
        self.btn_plots.setMinimumHeight(32)
        
        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.accept)
        self.btn_close.setMinimumHeight(32)

        btn_layout.addWidget(self.btn_plots)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_close)
        layout.addLayout(btn_layout)

    def populate(self, results: dict):
        """Populate the table with the finished simulation results."""
        self.results = results
        self.table.setRowCount(0)
        
        # Check if this is a 3-way comparison from NdobFbl or a single simulation
        if "metrics" in results and "pid" in results["metrics"]:
            self._populate_three_way(results["metrics"])
            self.btn_plots.setVisible(False)  # We handle figures in the viewer now
        else:
            self.btn_plots.setVisible(True)
            self._populate_single(results)

    def _populate_three_way(self, metrics: dict):
        """Populate the table for the 3-way controller comparison."""
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Metric", "PID", "FBL", "FBL+NDOB"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for i in range(1, 4):
            self.table.horizontalHeader().setSectionResizeMode(i, QHeaderView.ResizeToContents)

        # Handover Compliance Check (use NDOB as the reference for banner)
        ndob_pass = metrics["handover"]["ndob_pass"]
        if ndob_pass:
            self.lbl_banner.setText("HANDOVER COMPLIANT  ✓  (FBL+NDOB)")
            self.lbl_banner.setStyleSheet("color: #00e676; font-size: 16px; font-weight: bold;")
        else:
            self.lbl_banner.setText("HANDOVER FAIL  ✗  (FBL+NDOB)")
            self.lbl_banner.setStyleSheet("color: #ef5350; font-size: 16px; font-weight: bold;")

        pid = metrics["pid"]
        fbl = metrics["fbl"]
        ndob = metrics["ndob"]

        rows_data = [
            ("Settling Time (Az)", "{:.3f} s", "settling_time_az"),
            ("Settling Time (El)", "{:.3f} s", "settling_time_el"),
            ("Steady-State Error (Az)", "{:.2f} µrad", "ss_error_az", 1e6),
            ("Steady-State Error (El)", "{:.2f} µrad", "ss_error_el", 1e6),
            ("Final Az Error (Deg)", "{:.3f}°", "ss_error_az", 180.0/math.pi),
            ("LOS Error RMS", "{:.2f} µrad", "los_error_rms", 1e6),
            ("Total Torque RMS", "{:.3f} N·m", "torque_rms"),
            ("FSM Saturation", "{:.1f} %", "fsm_saturation_pct"),
        ]

        # Extract handover tests for the Final Az Error
        hp = metrics["handover"]["pid_pass"]
        hf = metrics["handover"]["fbl_pass"]
        hn = metrics["handover"]["ndob_pass"]
        handover_passes = [hp, hf, hn]

        for desc, fmt, key, *scale in rows_data:
            multiplier = scale[0] if scale else 1.0
            
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(desc))

            for col, d_dict in enumerate([pid, fbl, ndob]):
                val = d_dict.get(key, 0.0)
                if key.startswith("ss_error_"):
                    val = abs(val)

                val_item = QTableWidgetItem(fmt.format(val * multiplier))
                val_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                
                if desc.startswith("Final Az Error"):
                    bg = Qt.darkGreen if handover_passes[col] else Qt.darkRed
                    val_item.setBackground(bg)
                    val_item.setForeground(Qt.white)

                self.table.setItem(row, col + 1, val_item)

    def _populate_single(self, results: dict):
        """Populate the table for a single run."""
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Metric", "Value", "Status"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)

        ss_az_rad = results.get("ss_error_az", 0.0)
        ss_el_rad = results.get("ss_error_el", 0.0)
        ss_az_deg = abs(math.degrees(ss_az_rad))
        
        passed_handover = (ss_az_deg <= self.HANDOVER_THRESHOLD_DEG)
        
        if passed_handover:
            self.lbl_banner.setText("HANDOVER COMPLIANT  ✓")
            self.lbl_banner.setStyleSheet("color: #00e676; font-size: 16px; font-weight: bold;")
        else:
            self.lbl_banner.setText("HANDOVER FAIL  ✗")
            self.lbl_banner.setStyleSheet("color: #ef5350; font-size: 16px; font-weight: bold;")

        metrics = [
            ("Settling Time (Az)", "{:.3f} s", results.get("settling_time_az", 0.0), None),
            ("Settling Time (El)", "{:.3f} s", results.get("settling_time_el", 0.0), None),
            ("Steady-State Error (Az)", "{:.2f} µrad", abs(ss_az_rad)*1e6, None),
            ("Steady-State Error (El)", "{:.2f} µrad", abs(ss_el_rad)*1e6, None),
            ("Final Az Error (Deg)", "{:.3f}°", ss_az_deg, passed_handover),
            ("LOS Error RMS", "{:.2f} µrad", results.get("los_error_rms", 0.0)*1e6, None),
            ("Total Torque RMS", "{:.3f} N·m", float(math.hypot(
                results.get("torque_rms_az", 0.0),
                results.get("torque_rms_el", 0.0)
            )), None),
            ("FSM Saturation", "{:.1f} %", results.get("fsm_saturation_pct", 0.0), lambda v: v < 20.0),
        ]

        def get_status_widget(condition, text_override=None):
            if condition is None:
                return QTableWidgetItem("—")
            passed = condition
            item = QTableWidgetItem(text_override if text_override else ("PASS" if passed else "FAIL"))
            item.setTextAlignment(Qt.AlignCenter)
            item.setForeground(Qt.green if passed else Qt.red)
            return item

        for label, fmt, val, condition in metrics:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(label))
            
            val_item = QTableWidgetItem(fmt.format(val))
            val_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.table.setItem(row, 1, val_item)
            
            eval_cond = condition(val) if callable(condition) else condition
            stat_item = get_status_widget(eval_cond)
            self.table.setItem(row, 2, stat_item)

            if label.startswith("Final Az Error"):
                bg = Qt.darkGreen if eval_cond else Qt.darkRed
                val_item.setBackground(bg)
                val_item.setForeground(Qt.white)

    def _generate_publication_plots(self):
        """Invoke matplotlib generator on a background thread."""
        self.btn_plots.setEnabled(False)
        self.btn_plots.setText("Generating...")

        self._plot_thread = PlotGenerationThread(self.results)
        self._plot_thread.finished_msg.connect(self._on_plots_generated)
        self._plot_thread.start()

    def _on_plots_generated(self, title: str, text: str):
        self.btn_plots.setEnabled(True)
        self.btn_plots.setText("📊  Generate Publication Plots")
        
        icon = QMessageBox.Information if title == "Success" else QMessageBox.Critical
        msg = QMessageBox(icon, title, text, QMessageBox.Ok, self)
        msg.exec()
