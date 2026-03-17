"""
ControlPanelWidget — left-dock configuration panel.

Maps directly to ``SimulationConfig`` dataclass fields.  Four collapsible
``QGroupBox`` sections:

1. Target Trajectory  — signal type, Az/El setpoints, amplitude, period
2. Controller Selection — radio buttons: PID / FBL / FBL + NDOB
3. Disturbance Injection — checkable group; wind, vibration, noise sub-controls
4. Execution                — Run / Stop buttons, duration, progress bar

Emits ``start_requested(dict)`` where the dict maps 1-to-1 onto
``SimulationConfig`` constructor kwargs (so the caller can do
``SimulationConfig(**panel.get_simulation_config())``).
"""

from PySide6.QtCore import Qt, Signal
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
    QRadioButton,
    QButtonGroup,
    QSizePolicy,
    QFrame,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: compact horizontal separator
# ─────────────────────────────────────────────────────────────────────────────

def _h_sep() -> QFrame:
    sep = QFrame()
    sep.setFrameShape(QFrame.HLine)
    sep.setFrameShadow(QFrame.Sunken)
    sep.setStyleSheet("color: #2a2a4a;")
    return sep


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: labelled spinbox row for QFormLayout
# ─────────────────────────────────────────────────────────────────────────────

def _dspin(lo: float, hi: float, val: float, step: float = 1.0,
           decimals: int = 3, suffix: str = "", tip: str = "") -> QDoubleSpinBox:
    sb = QDoubleSpinBox()
    sb.setRange(lo, hi)
    sb.setValue(val)
    sb.setSingleStep(step)
    sb.setDecimals(decimals)
    if suffix:
        sb.setSuffix(f"  {suffix}")
    if tip:
        sb.setToolTip(tip)
    sb.setMinimumWidth(110)
    return sb


# ─────────────────────────────────────────────────────────────────────────────
#  Main widget
# ─────────────────────────────────────────────────────────────────────────────

class ControlPanelWidget(QWidget):
    """
    Professional configuration panel for the LaserCom Digital Twin.

    Signals
    -------
    start_requested : Signal(dict)
        Emitted on Run; dict is a flat ``SimulationConfig`` kwarg mapping.
    stop_requested : Signal()
        Emitted when Stop is clicked.
    """

    start_requested = Signal(dict)
    stop_requested  = Signal()

    # Convenience alias so ``main_window.py`` can reference either name
    get_config = None  # assigned to get_simulation_config at end of __init__

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self._build_ui()
        self._connect_signals()
        self.get_config = self.get_simulation_config   # backward compat alias

    # ═════════════════════════════════════════════════════════════════════════
    #  UI construction
    # ═════════════════════════════════════════════════════════════════════════

    def _build_ui(self) -> None:
        # Outer layout wraps a scroll area so the panel stays usable at any height
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; }")

        container = QWidget()
        self._inner_layout = QVBoxLayout(container)
        self._inner_layout.setContentsMargins(10, 10, 10, 10)
        self._inner_layout.setSpacing(14)

        self._inner_layout.addWidget(self._build_trajectory_group())
        self._inner_layout.addWidget(self._build_controller_group())
        self._inner_layout.addWidget(self._build_disturbance_group())
        self._inner_layout.addWidget(_h_sep())
        self._inner_layout.addWidget(self._build_execution_group())
        self._inner_layout.addStretch(1)

        scroll.setWidget(container)
        outer.addWidget(scroll)

    # ─────────────────────────────────────────────────────────────────────────
    #  Group 1 — Target Trajectory
    # ─────────────────────────────────────────────────────────────────────────

    def _build_trajectory_group(self) -> QGroupBox:
        grp = QGroupBox("🎯  Target Trajectory")
        form = QFormLayout(grp)
        form.setRowWrapPolicy(QFormLayout.WrapAllRows)
        form.setLabelAlignment(Qt.AlignLeft)
        form.setSpacing(8)

        # Signal type
        self.combo_signal = QComboBox()
        self.combo_signal.addItems(["Constant", "Sine", "Cosine", "Square", "Hybridsig"])
        self.combo_signal.setToolTip(
            "Waveform type for the target angle trajectory.\n"
            "'Hybridsig' performs a slew to reach-angle then holds."
        )
        form.addRow("Signal Type", self.combo_signal)

        # Az / El setpoints
        self.spin_target_az = _dspin(-180.0, 180.0, 0.0, 1.0, 2, "deg",
                                     "Target azimuth angle (Constant / Hybridsig)")
        self.spin_target_el = _dspin(-90.0,  90.0,  28.6479, 1.0, 2, "deg",
                                     "Target elevation angle (≈0.5 rad default)")
        form.addRow("Target Az", self.spin_target_az)
        form.addRow("Target El", self.spin_target_el)

        # Time-varying parameters (hidden for Constant)
        self.spin_amplitude = _dspin(0.0, 90.0, 1.0, 0.5, 2, "deg",
                                     "Peak amplitude for Sine/Square/Hybridsig")
        self.spin_period = _dspin(0.1, 60.0, 2.0, 0.5, 2, "s",
                                  "Waveform period / slew duration")
        self.spin_reach_angle = _dspin(0.0, 90.0, 45.0, 5.0, 1, "deg",
                                       "Hold angle after reaching target (Hybridsig)")

        form.addRow("Amplitude",    self.spin_amplitude)
        form.addRow("Period",       self.spin_period)
        form.addRow("Reach Angle",  self.spin_reach_angle)

        # Show/hide dynamic controls based on signal type
        self._traj_dynamic_widgets = [
            self.spin_amplitude, self.spin_period, self.spin_reach_angle,
        ]
        self._traj_labels = [
            form.labelForField(self.spin_amplitude),
            form.labelForField(self.spin_period),
            form.labelForField(self.spin_reach_angle),
        ]
        self.combo_signal.currentTextChanged.connect(self._on_signal_type_changed)
        self._on_signal_type_changed(self.combo_signal.currentText())

        return grp

    def _on_signal_type_changed(self, text: str) -> None:
        show_dynamic = text != "Constant"
        show_reach   = text == "Hybridsig"
        for w in self._traj_dynamic_widgets[:2]:
            w.setVisible(show_dynamic)
        for lbl in self._traj_labels[:2]:
            if lbl:
                lbl.setVisible(show_dynamic)
        self.spin_reach_angle.setVisible(show_reach)
        if self._traj_labels[2]:
            self._traj_labels[2].setVisible(show_reach)

    # ─────────────────────────────────────────────────────────────────────────
    #  Group 2 — Controller Selection
    # ─────────────────────────────────────────────────────────────────────────

    def _build_controller_group(self) -> QGroupBox:
        grp = QGroupBox("🕹  Controller Selection")
        layout = QVBoxLayout(grp)
        layout.setSpacing(10)

        self._ctrl_btn_group = QButtonGroup(self)
        self._ctrl_btn_group.setExclusive(True)

        options = [
            ("Baseline PID",
             "Standard PID on both coarse gimbal axes.\n"
             "Fast, robust, no model knowledge required."),
            ("Feedback Linearization (FBL)",
             "Exact linearisation using the nonlinear EOM.\n"
             "Cancels gravity/Coriolis; near-linear closed-loop."),
            ("FBL + NDOB  (Nonlinear Disturbance Observer)",
             "FBL + online disturbance estimation.\n"
             "Eliminates SSE from friction, wind, unmodelled dynamics."),
        ]

        for i, (label, tip) in enumerate(options):
            rb = QRadioButton(label)
            rb.setToolTip(tip)
            if i == 0:
                rb.setChecked(True)
            self._ctrl_btn_group.addButton(rb, i)
            layout.addWidget(rb)

            if i == 1:
                # NDOB bandwidth sub-controls (only visible when FBL+NDOB selected)
                self._ndob_frame = QFrame()
                fl = QFormLayout(self._ndob_frame)
                fl.setContentsMargins(18, 0, 0, 0)
                fl.setSpacing(4)
                self.spin_lambda_az = _dspin(1.0, 500.0, 40.0, 5.0, 1, "rad/s",
                                             "NDOB bandwidth — azimuth axis")
                self.spin_lambda_el = _dspin(1.0, 500.0, 40.0, 5.0, 1, "rad/s",
                                             "NDOB bandwidth — elevation axis")
                fl.addRow("λ Az", self.spin_lambda_az)
                fl.addRow("λ El", self.spin_lambda_el)
                self._ndob_frame.setVisible(False)
                layout.addWidget(self._ndob_frame)

        self._ctrl_btn_group.idClicked.connect(self._on_controller_changed)

        # Direct state feedback debug toggle
        layout.addWidget(_h_sep())
        self.chk_direct_feedback = QCheckBox("Direct State Feedback  (bypass EKF)")
        self.chk_direct_feedback.setToolTip(
            "Use true simulation state instead of EKF estimates.\n"
            "Useful for isolating control law performance from estimator noise."
        )
        layout.addWidget(self.chk_direct_feedback)

        return grp

    def _on_controller_changed(self, idx: int) -> None:
        self._ndob_frame.setVisible(idx == 2)

    # ─────────────────────────────────────────────────────────────────────────
    #  Group 3 — Disturbance Injection (checkable = collapsible)
    # ─────────────────────────────────────────────────────────────────────────

    def _build_disturbance_group(self) -> QGroupBox:
        grp = QGroupBox("💨  Disturbance Injection")
        grp.setCheckable(True)
        grp.setChecked(False)
        grp.setToolTip("Enable environmental disturbances injected into the PLANT only.\n"
                       "The controller model does NOT see them — NDOB must estimate them.")
        self._dist_group = grp

        layout = QVBoxLayout(grp)
        layout.setSpacing(10)

        # ── Wind ──────────────────────────────────────────────────────────
        wind_grp = QGroupBox("Dryden Wind Turbulence")
        wind_grp.setCheckable(True)
        wind_grp.setChecked(False)
        self._wind_grp = wind_grp
        wf = QFormLayout(wind_grp)
        wf.setSpacing(6)

        self.spin_wind_velocity = _dspin(0.0, 50.0, 5.0, 0.5, 1, "m/s",
                                         "Mean wind speed used in Dryden model")
        self.spin_wind_turbulence = _dspin(0.0, 0.5, 0.15, 0.01, 2, "",
                                           "Turbulence intensity σ/V  (0.1=light, 0.2=moderate)")
        self.spin_wind_start = _dspin(0.0, 60.0, 2.0, 0.5, 1, "s",
                                      "Time at which wind starts")
        wf.addRow("Mean Velocity",   self.spin_wind_velocity)
        wf.addRow("Turbulence σ/V",  self.spin_wind_turbulence)
        wf.addRow("Start Time",      self.spin_wind_start)
        layout.addWidget(wind_grp)

        # ── Structural Vibration ──────────────────────────────────────────
        vib_grp = QGroupBox("Structural Vibration")
        vib_grp.setCheckable(True)
        vib_grp.setChecked(False)
        self._vib_grp = vib_grp
        vf = QFormLayout(vib_grp)
        vf.setSpacing(6)

        self.spin_vib_start = _dspin(0.0, 60.0, 0.0, 0.5, 1, "s",
                                     "Vibration injection start time")
        self.spin_vib_coupling = _dspin(0.0, 1.0, 0.1, 0.01, 3, "N·m/(m/s²)",
                                        "Inertia coupling coefficient")
        vf.addRow("Start Time",       self.spin_vib_start)
        vf.addRow("Inertia Coupling", self.spin_vib_coupling)
        layout.addWidget(vib_grp)

        # ── Structural Noise ──────────────────────────────────────────────
        noise_grp = QGroupBox("High-Freq Structural Noise")
        noise_grp.setCheckable(True)
        noise_grp.setChecked(False)
        self._noise_grp = noise_grp
        nf = QFormLayout(noise_grp)
        nf.setSpacing(6)

        self.spin_noise_std = _dspin(0.0, 0.1, 0.005, 0.001, 4, "N·m",
                                     "Noise RMS torque magnitude")
        nf.addRow("RMS Torque", self.spin_noise_std)
        layout.addWidget(noise_grp)

        return grp

    # ─────────────────────────────────────────────────────────────────────────
    #  Group 4 — Execution
    # ─────────────────────────────────────────────────────────────────────────

    def _build_execution_group(self) -> QGroupBox:
        grp = QGroupBox("▶  Execution")
        layout = QVBoxLayout(grp)
        layout.setSpacing(10)

        # Duration
        dur_row = QHBoxLayout()
        dur_row.addWidget(QLabel("Duration"))
        self.spin_duration = _dspin(1.0, 600.0, 5.0, 1.0, 1, "s",
                                    "Total simulation wall-clock time")
        dur_row.addWidget(self.spin_duration)
        layout.addLayout(dur_row)

        # Seed
        seed_row = QHBoxLayout()
        seed_row.addWidget(QLabel("Random Seed"))
        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 99999)
        self.spin_seed.setValue(42)
        self.spin_seed.setToolTip("Deterministic RNG seed for reproducible runs")
        self.spin_seed.setMinimumWidth(90)
        seed_row.addWidget(self.spin_seed)
        layout.addLayout(seed_row)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Ready")
        self.progress_bar.setMinimumHeight(22)
        layout.addWidget(self.progress_bar)

        # Run / Stop buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self.btn_run = QPushButton("▶  RUN SIMULATION")
        self.btn_run.setObjectName("btn_run")
        self.btn_run.setMinimumHeight(42)
        self.btn_run.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_run.setToolTip("Launch the simulation on a background thread (F5)")
        self.btn_run.setShortcut("F5")

        self.btn_stop = QPushButton("■  STOP")
        self.btn_stop.setObjectName("btn_stop")
        self.btn_stop.setMinimumHeight(42)
        self.btn_stop.setFixedWidth(90)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setToolTip("Gracefully cancel the running simulation (Shift+F5)")
        self.btn_stop.setShortcut("Shift+F5")

        btn_row.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_stop)
        layout.addLayout(btn_row)

        return grp

    # ═════════════════════════════════════════════════════════════════════════
    #  Signal wiring
    # ═════════════════════════════════════════════════════════════════════════

    def _connect_signals(self) -> None:
        self.btn_run.clicked.connect(
            lambda: self.start_requested.emit(self.get_simulation_config())
        )
        self.btn_stop.clicked.connect(self.stop_requested)

    # ═════════════════════════════════════════════════════════════════════════
    #  Public API — SimulationConfig construction
    # ═════════════════════════════════════════════════════════════════════════

    def get_simulation_config(self) -> dict:
        """
        Build a flat dict of kwargs that maps 1-to-1 onto ``SimulationConfig``.

        The caller can do::

            from lasercom_digital_twin.core.simulation.simulation_runner import SimulationConfig
            cfg = SimulationConfig(**panel.get_simulation_config())
        """
        import math

        ctrl_id = self._ctrl_btn_group.checkedId()  # 0=PID, 1=FBL, 2=FBL+NDOB

        signal_map = {
            "Constant":  "constant",
            "Sine":      "sine",
            "Cosine":    "cosine",
            "Square":    "square",
            "Hybridsig": "hybridsig",
        }
        signal_type = signal_map.get(self.combo_signal.currentText(), "constant")

        # Angle conversions deg → rad
        az_rad = math.radians(self.spin_target_az.value())
        el_rad = math.radians(self.spin_target_el.value())

        dist_enabled = self._dist_group.isChecked()
        wind_on      = dist_enabled and self._wind_grp.isChecked()
        vib_on       = dist_enabled and self._vib_grp.isChecked()
        noise_on     = dist_enabled and self._noise_grp.isChecked()

        env_cfg = {
            "seed": self.spin_seed.value(),
            "wind": {
                "enabled":               wind_on,
                "start_time":            self.spin_wind_start.value(),
                "mean_velocity":         self.spin_wind_velocity.value(),
                "turbulence_intensity":  self.spin_wind_turbulence.value(),
                "scale_length":          200.0,
                "direction_deg":         45.0,
                "gimbal_area":           0.02,
                "gimbal_arm":            0.15,
                "drag_coefficient":      1.2,
            },
            "vibration": {
                "enabled":              vib_on,
                "start_time":           self.spin_vib_start.value(),
                "modal_frequencies":    [15.0, 45.0, 80.0],
                "modal_dampings":       [0.02, 0.015, 0.01],
                "modal_amplitudes":     [1e-3, 5e-4, 2e-4],
                "inertia_coupling":     self.spin_vib_coupling.value(),
                "noise_floor_psd":      1e-6,
            },
            "structural_noise": {
                "enabled":  noise_on,
                "std":      self.spin_noise_std.value(),
                "freq_low":  100.0,
                "freq_high": 500.0,
            },
        }

        return {
            # ── Timing (defaults — GUI exposes duration, not dt) ──────────
            "dt_sim":    0.001,
            "dt_fine":   0.0001,
            "dt_coarse": 0.010,
            "log_period": 0.001,
            # ── Execution ────────────────────────────────────────────────
            "seed":               self.spin_seed.value(),
            "enable_plotting":    False,   # GUI handles its own plots
            # ── Target trajectory ────────────────────────────────────────
            "target_az":          az_rad,
            "target_el":          el_rad,
            "target_enabled":     True,
            "target_type":        signal_type,
            "target_amplitude":   self.spin_amplitude.value(),
            "target_period":      self.spin_period.value(),
            "target_reachangle":  self.spin_reach_angle.value(),
            # ── Controller ───────────────────────────────────────────────
            "use_feedback_linearization": ctrl_id >= 1,
            "use_direct_state_feedback":  self.chk_direct_feedback.isChecked(),
            "ndob_config": {
                "enable":    ctrl_id == 2,
                "lambda_az": self.spin_lambda_az.value(),
                "lambda_el": self.spin_lambda_el.value(),
                "d_max":     5.0,
            },
            # ── Disturbances ─────────────────────────────────────────────
            "environmental_disturbance_enabled": dist_enabled,
            "environmental_disturbance_config":  env_cfg,
            # duration is passed separately to run_simulation(), not SimulationConfig
            "_gui_duration": self.spin_duration.value(),
        }

    # ═════════════════════════════════════════════════════════════════════════
    #  State management — called by SimulationController
    # ═════════════════════════════════════════════════════════════════════════

    def set_running_state(self, running: bool) -> None:
        """Disable/enable all inputs while simulation is active."""
        self.btn_run.setEnabled(not running)
        self.btn_stop.setEnabled(running)

        for w in [
            self.combo_signal, self.spin_target_az, self.spin_target_el,
            self.spin_amplitude, self.spin_period, self.spin_reach_angle,
            self.spin_duration, self.spin_seed,
            self.spin_wind_velocity, self.spin_wind_turbulence, self.spin_wind_start,
            self.spin_vib_start, self.spin_vib_coupling, self.spin_noise_std,
            self.spin_lambda_az, self.spin_lambda_el,
            self.chk_direct_feedback, self._dist_group, self._wind_grp,
            self._vib_grp, self._noise_grp,
        ]:
            w.setEnabled(not running)

        for btn in self._ctrl_btn_group.buttons():
            btn.setEnabled(not running)

        if running:
            self.progress_bar.setFormat("%p%  —  Running…")
        else:
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("Ready")

    def set_progress(self, pct: int) -> None:
        """Update progress bar — called from SimulationController."""
        self.progress_bar.setValue(pct)
        self.progress_bar.setFormat(f"{pct}%  —  Running…")
