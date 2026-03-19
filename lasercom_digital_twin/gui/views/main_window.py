"""
Main application window with dockable panel layout.

Layout
------
- Left dock   : ``ConfigPanel``   — controller & disturbance settings
- Center      : ``InteractivePlotViewer`` — plot placeholders
- Bottom dock : ``ConsoleLogger`` — timestamped status log

All three panels are ``QDockWidget`` instances so the user can rearrange,
float, or close them at will.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMainWindow, QDockWidget, QMenuBar, QMenu, QStatusBar

from lasercom_digital_twin.gui.views.config_panel import ControlPanelWidget
from lasercom_digital_twin.gui.views.console_logger import ConsoleLogger
from lasercom_digital_twin.gui.views.figure_viewer import InteractivePlotViewer


class DigitalTwinMainWindow(QMainWindow):
    """
    Top-level ``QMainWindow`` for the LaserCom Digital Twin interface.

    Provides:
    - Menu bar (File, Simulation, View)
    - Three dockable panels
    - Status bar
    """

    WINDOW_TITLE = "LaserCom Digital Twin — Gimbal + FSM Interface"
    DEFAULT_SIZE = (1400, 850)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle(self.WINDOW_TITLE)
        self.resize(*self.DEFAULT_SIZE)
        self.setDockNestingEnabled(True)

        # ---- Create child widgets ----
        self.control_panel = ControlPanelWidget()
        self.console_logger = ConsoleLogger()
        self.figure_viewer = InteractivePlotViewer()

        # ---- Build layout ----
        self._create_docks()
        self._create_menu_bar()
        self._create_status_bar()

    # ------------------------------------------------------------------ #
    #  Dock widgets                                                       #
    # ------------------------------------------------------------------ #

    def _create_docks(self) -> None:
        # --- Central widget (interactive plot viewer) ---
        self.setCentralWidget(self.figure_viewer)

        # --- Left dock: Configuration ---
        self.dock_config = QDockWidget("⚙  Configuration", self)
        self.dock_config.setObjectName("dock_config")
        self.dock_config.setWidget(self.control_panel)
        # Give the left side 50% of the Default Size (1400 / 2 = 700)
        self.dock_config.setMinimumWidth(320)
        self.dock_config.setFeatures(
            QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
            | QDockWidget.DockWidgetClosable
        )
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_config)
        self.resizeDocks([self.dock_config], [700], Qt.Horizontal)

        # --- Bottom dock: Console ---
        self.dock_console = QDockWidget("📋  Console Output", self)
        self.dock_console.setObjectName("dock_console")
        self.dock_console.setWidget(self.console_logger)
        self.dock_console.setMinimumHeight(140)
        self.dock_console.setFeatures(
            QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
            | QDockWidget.DockWidgetClosable
        )
        self.addDockWidget(Qt.BottomDockWidgetArea, self.dock_console)



    # ------------------------------------------------------------------ #
    #  Menu bar                                                           #
    # ------------------------------------------------------------------ #

    def _create_menu_bar(self) -> None:
        menu_bar = self.menuBar()

        # ---- File ----
        file_menu = menu_bar.addMenu("&File")
        self.action_exit = file_menu.addAction("Exit")
        self.action_exit.setShortcut("Ctrl+Q")
        self.action_exit.triggered.connect(self.close)

        # ---- Simulation ----
        sim_menu = menu_bar.addMenu("&Simulation")
        self.action_sim_start = sim_menu.addAction("▶  Start")
        self.action_sim_start.setShortcut("F5")
        self.action_sim_stop = sim_menu.addAction("■  Stop")
        self.action_sim_stop.setShortcut("Shift+F5")
        self.action_sim_stop.setEnabled(False)
        sim_menu.addSeparator()
        self.action_run_3way = sim_menu.addAction("📊  Run 3-Way Comparison")
        self.action_run_3way.setShortcut("Ctrl+M")

        # ---- View ----
        view_menu = menu_bar.addMenu("&View")
        view_menu.addAction(self.dock_config.toggleViewAction())
        view_menu.addAction(self.dock_console.toggleViewAction())

    # ------------------------------------------------------------------ #
    #  Status bar                                                         #
    # ------------------------------------------------------------------ #

    def _create_status_bar(self) -> None:
        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready  •  No simulation running")

    # ------------------------------------------------------------------ #
    #  Public helpers                                                      #
    # ------------------------------------------------------------------ #

    def set_status(self, message: str) -> None:
        """Update the status bar text."""
        self.status_bar.showMessage(message)
