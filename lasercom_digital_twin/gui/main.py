#!/usr/bin/env python3
"""
LaserCom Digital Twin — PySide6 GUI Entry Point.

Launches the dark-themed main window and wires the MVC simulation controller.

Usage
-----
    python -m lasercom_digital_twin.gui.main

Requirements
------------
    pip install PySide6
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path for module resolution
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import matplotlib
matplotlib.use("Agg")  # CRITICAL: Force non-interactive backend for thread safety

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from lasercom_digital_twin.gui.resources.styles import DARK_THEME_QSS
from lasercom_digital_twin.gui.views.main_window import DigitalTwinMainWindow
from lasercom_digital_twin.gui.controllers.simulation_controller import (
    SimulationController,
)


def main() -> None:
    """Application entry point."""

    # High-DPI scaling (PySide6 enables this by default, but be explicit)
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("LaserCom Digital Twin")
    app.setOrganizationName("MicroPrecisionGimbal")

    # Apply dark theme
    app.setStyleSheet(DARK_THEME_QSS)

    # Create main window
    window = DigitalTwinMainWindow()

    # Wire MVC controller
    controller = SimulationController(window)

    # Ensure clean shutdown
    app.aboutToQuit.connect(controller.shutdown)

    window.show()

    # Log startup
    window.console_logger.log(
        "LaserCom Digital Twin GUI initialized — ready.", "OK"
    )
    window.console_logger.log(
        "Select a controller, configure disturbances, and press Start.", "INFO"
    )

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
