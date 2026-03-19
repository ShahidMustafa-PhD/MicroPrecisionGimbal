"""
Simulation controller — MVC glue between views and the worker thread.

Responsibilities
----------------
1. Own the ``QThread`` + ``SimulationWorker`` lifecycle.
2. Connect config-panel signals (start/stop) → worker.
3. Connect worker signals (progress, finished) → console.
4. Ensure clean thread shutdown on application exit.
"""

from PySide6.QtCore import QObject, Slot

from lasercom_digital_twin.gui.models.simulation_worker import (
    SimulationWorker,
    create_worker_thread,
)
from lasercom_digital_twin.gui.models.analysis_worker import (
    AnalysisWorker,
    create_analysis_worker,
)
from lasercom_digital_twin.gui.views.main_window import DigitalTwinMainWindow
from lasercom_digital_twin.gui.views.results_summary import ResultsSummaryWidget


class SimulationController(QObject):
    """
    Orchestrates simulation lifecycle and data routing.

    Parameters
    ----------
    main_window : DigitalTwinMainWindow
        The top-level window whose child widgets we wire up.
    """

    def __init__(self, main_window: DigitalTwinMainWindow, parent: QObject = None):
        super().__init__(parent)
        self._window = main_window
        self._thread = None
        self._worker = None

        self._connect_view_signals()

    # ------------------------------------------------------------------ #
    #  Signal wiring                                                      #
    # ------------------------------------------------------------------ #

    def _connect_view_signals(self) -> None:
        w = self._window

        # Control panel buttons
        w.control_panel.start_requested.connect(self.start_simulation)
        w.control_panel.stop_requested.connect(self.stop_simulation)

        # Menu bar actions
        w.action_sim_start.triggered.connect(
            lambda: self.start_simulation(w.control_panel.get_simulation_config())
        )
        w.action_sim_stop.triggered.connect(self.stop_simulation)
        if hasattr(w, "action_run_3way"):
            w.action_run_3way.triggered.connect(
                lambda: self.start_analysis(w.control_panel.get_simulation_config())
            )

    # ------------------------------------------------------------------ #
    #  Simulation lifecycle                                                #
    # ------------------------------------------------------------------ #

    @Slot(dict)
    def start_simulation(self, config: dict) -> None:
        """Launch the simulation on a background thread."""
        if self._thread is not None and self._thread.isRunning():
            self._window.console_logger.log(
                "Simulation already running — stop it first.", "WARN"
            )
            return

        # UI state
        self._window.control_panel.set_running_state(True)
        self._window.action_sim_start.setEnabled(False)
        self._window.action_sim_stop.setEnabled(True)
        self._window.set_status("Simulation running…")

        # Log
        ctrl = config.get("controller_type", "")
        use_fbl  = config.get("use_feedback_linearization", False)
        use_ndob = config.get("ndob_config", {}).get("enable", False)
        ctrl_label = ("FBL + NDOB" if use_ndob else "FBL") if use_fbl else "Baseline PID"
        dur   = config.get("_gui_duration", 5.0)
        dist  = config.get("environmental_disturbance_enabled", False)
        self._window.console_logger.log(
            f"Starting simulation: controller={ctrl_label}, duration={dur:.1f}s, "
            f"disturbances={'ON' if dist else 'OFF'}",
            "INFO",
        )

        # Create fresh thread + worker
        self._thread, self._worker = create_worker_thread()
        # Pass duration separately; SimulationConfig doesn't have a duration field
        worker_config = dict(config)
        worker_config["duration"] = config.get("_gui_duration", 5.0)
        self._worker.configure(worker_config)

        # Worker → UI connections
        self._worker.progress_update.connect(self._on_progress)
        self._worker.simulation_finished.connect(self._on_finished)
        self._worker.error_occurred.connect(self._on_error)

        # Start
        self._thread.started.connect(self._worker.run)
        self._thread.start()

    @Slot()
    def stop_simulation(self) -> None:
        """Request graceful cancellation of the running simulation."""
        if self._worker is not None:
            self._window.console_logger.log("Stop requested — cancelling…", "WARN")
            self._worker.request_stop()

    # ------------------------------------------------------------------ #
    #  Worker signal handlers                                              #
    # ------------------------------------------------------------------ #


    @Slot(int)
    def _on_progress(self, pct: int) -> None:
        self._window.control_panel.set_progress(pct)

    @Slot(dict)
    def _on_finished(self, results: dict) -> None:
        status = results.get("status", "unknown")
        rms = results.get("los_error_rms", 0.0) * 1e6
        sat = results.get("fsm_saturation_pct", 0.0)

        level = "OK" if status == "completed" else "WARN"
        self._window.console_logger.log(
            f"Simulation {status}  |  LOS RMS: {rms:.2f} µrad  |  "
            f"FSM saturation: {sat:.1f}%",
            level,
        )

        self._cleanup_thread()

        # Only show the rigorous academic view if it completed normally.
        if status == "completed":
            figures = results.get("figures", {})
            if figures:
                self._window.figure_viewer.populate(figures)

            dialog = ResultsSummaryWidget(self._window)
            dialog.populate(results)
            dialog.exec()

    @Slot(str)
    def _on_error(self, message: str) -> None:
        self._window.console_logger.log(message, "ERROR")
        self._cleanup_thread()

    # ------------------------------------------------------------------ #
    #  Analysis lifecycle                                                 #
    # ------------------------------------------------------------------ #

    @Slot(dict)
    def start_analysis(self, config: dict) -> None:
        """Launch the 3-way controller analysis on a background thread."""
        if self._thread is not None and self._thread.isRunning():
            self._window.console_logger.log(
                "Simulation or analysis already running — stop it first.", "WARN"
            )
            return

        self._window.control_panel.set_running_state(True)
        self._window.action_sim_start.setEnabled(False)
        self._window.action_sim_stop.setEnabled(False) # Not cancellable yet
        if hasattr(self._window, "action_run_3way"):
            self._window.action_run_3way.setEnabled(False)
        self._window.set_status("3-Way Controller Analysis running…")
        self._window.console_logger.log("Starting 3-way MVC Feedback Linearization Comparison...", "INFO")

        self._thread, self._worker = create_analysis_worker(config)

        self._worker.progress.connect(lambda msg: self._window.console_logger.log(msg, "INFO"))
        self._worker.finished.connect(self._on_analysis_finished)
        self._worker.error.connect(self._on_error)

        self._thread.started.connect(self._worker.run)
        self._thread.start()

    @Slot(dict)
    def _on_analysis_finished(self, results: dict) -> None:
        """Handle the returned metrics and figures from the NdobFbl model."""
        self._window.console_logger.log("Analysis Output Processed Successfully", "OK")
        self._cleanup_thread()

        figures = results.get("figures", {})
        if figures:
            self._window.figure_viewer.populate(figures)

        dialog = ResultsSummaryWidget(self._window)
        dialog.populate(results)
        dialog.exec()

    # ------------------------------------------------------------------ #
    #  Thread cleanup                                                      #
    # ------------------------------------------------------------------ #

    def _cleanup_thread(self) -> None:
        """Restore UI state and tear down the worker thread."""
        self._window.control_panel.set_running_state(False)
        self._window.action_sim_start.setEnabled(True)
        self._window.action_sim_stop.setEnabled(False)
        if hasattr(self._window, "action_run_3way"):
            self._window.action_run_3way.setEnabled(True)
        self._window.set_status("Ready  •  No simulation running")

        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(3000)  # 3 s timeout
            self._thread.deleteLater()
            self._thread = None

        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None

    def shutdown(self) -> None:
        """Called on application exit — ensures thread is stopped."""
        if self._worker is not None:
            self._worker.request_stop()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(5000)
