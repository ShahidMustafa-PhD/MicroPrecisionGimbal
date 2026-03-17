"""
Console logger — bottom dock widget.

Provides a read-only, terminal-styled ``QTextEdit`` for structured log output.
Supports severity levels with colour coding and optional timestamps.

Thread-safe: use ``log()`` from any thread — it internally marshals
the call to the GUI thread via a queued signal.
"""

from datetime import datetime

from PySide6.QtCore import Signal, Slot, Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QHBoxLayout, QPushButton


# Colour map for log levels
_LEVEL_COLOURS = {
    "INFO":    "#58a6ff",
    "OK":      "#00e676",
    "WARN":    "#ffc107",
    "ERROR":   "#ef5350",
    "DEBUG":   "#a0a0b0",
    "TELEM":   "#bb86fc",
}


class ConsoleLogger(QWidget):
    """
    Bottom-dock console widget with coloured, timestamped log output.

    Signals
    -------
    _append_signal : Signal(str)
        Internal signal for marshalling log calls to the GUI thread.
    """

    _append_signal = Signal(str)

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self._build_ui()
        # Connect internal signal to slot (queued when cross-thread)
        self._append_signal.connect(self._do_append, Qt.QueuedConnection)

    # ------------------------------------------------------------------ #
    #  UI                                                                 #
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # Console text area
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setLineWrapMode(QTextEdit.NoWrap)
        self.text_edit.setPlaceholderText("Console output will appear here…")
        layout.addWidget(self.text_edit)

        # Bottom toolbar
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)

        self.btn_clear = QPushButton("Clear")
        self.btn_clear.setFixedWidth(80)
        self.btn_clear.clicked.connect(self.text_edit.clear)
        toolbar.addStretch()
        toolbar.addWidget(self.btn_clear)

        layout.addLayout(toolbar)

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def log(self, message: str, level: str = "INFO") -> None:
        """
        Append a log line.  Safe to call from **any** thread.

        Parameters
        ----------
        message : str
            Log text.
        level : str
            One of ``INFO``, ``OK``, ``WARN``, ``ERROR``, ``DEBUG``, ``TELEM``.
        """
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        colour = _LEVEL_COLOURS.get(level.upper(), "#eaeaea")
        html = (
            f'<span style="color:#666680;">[{ts}]</span> '
            f'<span style="color:{colour}; font-weight:600;">'
            f'[{level.upper():>5s}]</span> '
            f'<span style="color:#c9d1d9;">{message}</span>'
        )
        # Emit via queued signal — safe from worker threads
        self._append_signal.emit(html)

    @Slot(str)
    def _do_append(self, html: str) -> None:
        """Slot that actually mutates the QTextEdit (runs on GUI thread)."""
        self.text_edit.append(html)
        # Auto-scroll to bottom
        sb = self.text_edit.verticalScrollBar()
        sb.setValue(sb.maximum())
