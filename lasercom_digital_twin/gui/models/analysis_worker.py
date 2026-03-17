"""
QThread worker for the 3-Way Controller Analysis.
Runs NdobFbl.run_analysis without blocking the main event loop.
"""

from PySide6.QtCore import QObject, QThread, Signal, Slot
from lasercom_digital_twin.analysis.feedback_linearization import NdobFbl


class AnalysisWorker(QObject):
    """
    Executes the NdobFbl 3-way analysis in a separate thread.
    """
    progress = Signal(str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, config: dict):
        super().__init__()
        self._config = config

    @Slot()
    def run(self):
        try:
            model = NdobFbl()

            def progress_callback(msg: str):
                self.progress.emit(msg)

            # Execution block is CPU heavy
            results = model.run_analysis(self._config, progress_callback)
            self.finished.emit(results)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


def create_analysis_worker(config: dict) -> tuple[QThread, AnalysisWorker]:
    """
    Factory to spin up an AnalysisWorker on a dedicated QThread.

    Returns
        (thread, worker)
    """
    thread = QThread()
    worker = AnalysisWorker(config)
    worker.moveToThread(thread)

    worker.finished.connect(thread.quit)
    worker.error.connect(thread.quit)

    return thread, worker
