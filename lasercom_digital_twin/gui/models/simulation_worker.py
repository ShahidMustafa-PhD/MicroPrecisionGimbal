"""
QThread + QObject worker for non-blocking simulation execution.

Architecture
------------
- ``SimulationWorker(QObject)`` lives on a ``QThread``.
- The controller calls ``worker.start_simulation(config)`` via a
  queued-connection signal; the worker's ``run()`` slot executes on the
  worker thread, keeping the GUI event loop free.
- Three outbound signals push data back to the main thread:
    * ``progress_update(int)``     — 0–100 %
    * ``simulation_finished(dict)``— final summary when done

Thread-safety
-------------
Cancellation uses ``threading.Event`` (GIL-safe, wait-free check).
All cross-thread communication uses Qt's queued signal/slot mechanism.
"""

import threading

from PySide6.QtCore import QObject, QThread, Signal, Slot

from lasercom_digital_twin.gui.models.mock_backend import NdobaFbl


class SimulationWorker(QObject):
    """
    Worker that runs the simulation backend on a dedicated QThread.

    Signals
    -------
    progress_update : Signal(int)
        Emitted with integer percentage 0–100.
    simulation_finished : Signal(dict)
        Emitted once when the simulation completes or is cancelled.
    error_occurred : Signal(str)
        Emitted if an unhandled exception occurs in the worker.
    """

    progress_update = Signal(int)
    simulation_finished = Signal(dict)
    error_occurred = Signal(str)

    def __init__(self, parent: QObject = None):
        super().__init__(parent)
        self._stop_event = threading.Event()
        self._config: dict = {}

    # ------------------------------------------------------------------ #
    #  Configuration                                                      #
    # ------------------------------------------------------------------ #

    def configure(self, config: dict) -> None:
        """Set simulation parameters before starting.

        Parameters
        ----------
        config : dict
            Keys: ``controller_type``, ``disturbance_enabled``,
            ``duration``, ``dt``, ``seed``, …
        """
        self._config = dict(config)

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    @Slot()
    def run(self) -> None:
        """Execute the simulation — runs on the worker QThread."""
        self._stop_event.clear()

        duration = self._config.get("duration", 5.0)
        dt = self._config.get("dt", 0.1)

        try:
            backend = NdobaFbl(config=self._config)

            def _on_step(step_data: dict) -> None:
                """Callback invoked by the backend each step."""
                pct = step_data.get("progress_pct", 0)
                self.progress_update.emit(pct)

            results = backend.run_simulation(
                duration=duration,
                dt=dt,
                callback=_on_step,
                stop_flag=self._stop_event,
            )

            self.simulation_finished.emit(results)

        except Exception as exc:
            self.error_occurred.emit(f"Simulation error: {exc}")

    def request_stop(self) -> None:
        """Thread-safe request to cancel the running simulation."""
        self._stop_event.set()


def create_worker_thread() -> tuple[QThread, SimulationWorker]:
    """
    Factory that creates a properly-wired QThread + SimulationWorker pair.

    Returns
    -------
    thread : QThread
        The host thread (not yet started).
    worker : SimulationWorker
        The worker object, already moved to *thread*.

    Usage
    -----
    >>> thread, worker = create_worker_thread()
    >>> thread.started.connect(worker.run)
    >>> thread.start()
    """
    thread = QThread()
    worker = SimulationWorker()
    worker.moveToThread(thread)

    # Clean up when the worker signals completion
    worker.simulation_finished.connect(thread.quit)
    worker.error_occurred.connect(thread.quit)

    return thread, worker
