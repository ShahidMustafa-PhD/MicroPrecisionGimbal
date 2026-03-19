"""
Dark-theme QSS stylesheet for the LaserCom Digital Twin GUI.

Colour palette
--------------
- Background    : #1a1a2e  (deep navy)
- Surface       : #16213e  (panel background)
- Surface Alt   : #0f3460  (elevated cards / active docks)
- Accent        : #e94560  (buttons, highlights)
- Accent Hover  : #ff6b6b  (hover state)
- Text Primary  : #eaeaea
- Text Secondary: #a0a0b0
- Success       : #00e676
- Warning       : #ffc107
- Border        : #2a2a4a
"""

DARK_THEME_QSS = """

/* ================================================================== */
/*  Global                                                             */
/* ================================================================== */

QMainWindow {
    background-color: #1a1a2e;
    color: #eaeaea;
}

QWidget {
    background-color: #1a1a2e;
    color: #eaeaea;
    font-family: "Segoe UI", "Inter", "Roboto", sans-serif;
    font-size: 13px;
}

/* ================================================================== */
/*  Menu Bar                                                           */
/* ================================================================== */

QMenuBar {
    background-color: #16213e;
    color: #eaeaea;
    border-bottom: 1px solid #2a2a4a;
    padding: 2px 0px;
}

QMenuBar::item {
    background-color: transparent;
    padding: 6px 14px;
    border-radius: 4px;
}

QMenuBar::item:selected {
    background-color: #0f3460;
}

QMenu {
    background-color: #16213e;
    color: #eaeaea;
    border: 1px solid #2a2a4a;
    border-radius: 6px;
    padding: 4px;
}

QMenu::item {
    padding: 6px 30px 6px 14px;
    border-radius: 4px;
}

QMenu::item:selected {
    background-color: #0f3460;
}

QMenu::separator {
    height: 1px;
    background: #2a2a4a;
    margin: 4px 8px;
}

/* ================================================================== */
/*  Dock Widgets                                                       */
/* ================================================================== */

QDockWidget {
    color: #eaeaea;
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
    font-weight: bold;
    font-size: 12px;
}

QDockWidget::title {
    background-color: #0f3460;
    padding: 8px 12px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    border: 1px solid #2a2a4a;
    border-bottom: none;
}

QDockWidget QWidget {
    background-color: #16213e;
}

/* ================================================================== */
/*  Group Boxes                                                        */
/* ================================================================== */

QGroupBox {
    border: 1px solid #2a2a4a;
    border-radius: 8px;
    margin-top: 14px;
    padding: 14px 10px 10px 10px;
    font-weight: bold;
    color: #e94560;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 2px 10px;
    color: #e94560;
}

/* ================================================================== */
/*  Buttons                                                            */
/* ================================================================== */

QPushButton {
    background-color: #0f3460;
    color: #eaeaea;
    border: 1px solid #2a2a4a;
    border-radius: 6px;
    padding: 8px 20px;
    font-weight: 600;
    min-height: 24px;
}

QPushButton:hover {
    background-color: #e94560;
    border-color: #e94560;
}

QPushButton:pressed {
    background-color: #c73e54;
}

QPushButton:disabled {
    background-color: #2a2a4a;
    color: #666680;
    border-color: #22223a;
}

QPushButton#btn_start {
    background-color: #00897b;
    border-color: #00897b;
}

QPushButton#btn_start:hover {
    background-color: #00bfa5;
}

QPushButton#btn_stop {
    background-color: #c62828;
    border-color: #c62828;
}

QPushButton#btn_stop:hover {
    background-color: #ef5350;
}

/* RUN SIMULATION — large, prominent green gradient */
QPushButton#btn_run {
    background: qlineargradient(
        x1:0, y1:0, x2:1, y2:0,
        stop:0 #00897b, stop:1 #00bfa5
    );
    color: #ffffff;
    border: none;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 0.5px;
    padding: 10px 20px;
}

QPushButton#btn_run:hover {
    background: qlineargradient(
        x1:0, y1:0, x2:1, y2:0,
        stop:0 #00bfa5, stop:1 #69f0ae
    );
}

QPushButton#btn_run:pressed {
    background-color: #005f56;
}

QPushButton#btn_run:disabled {
    background: #2a2a4a;
    color: #666680;
}

/* ================================================================== */
/*  Combo Box                                                          */
/* ================================================================== */

QComboBox {
    background-color: #0f3460;
    color: #eaeaea;
    border: 1px solid #2a2a4a;
    border-radius: 6px;
    padding: 6px 12px;
    min-width: 120px;
}

QComboBox:hover {
    border-color: #e94560;
}

QComboBox::drop-down {
    border: none;
    width: 30px;
}

QComboBox QAbstractItemView {
    background-color: #16213e;
    color: #eaeaea;
    border: 1px solid #2a2a4a;
    selection-background-color: #0f3460;
}

/* ================================================================== */
/*  Spin Box / Double Spin Box                                         */
/* ================================================================== */

QSpinBox, QDoubleSpinBox {
    background-color: #0f3460;
    color: #eaeaea;
    border: 1px solid #2a2a4a;
    border-radius: 6px;
    padding: 6px 10px;
}

QSpinBox:hover, QDoubleSpinBox:hover {
    border-color: #e94560;
}

/* ================================================================== */
/*  Check Box                                                          */
/* ================================================================== */

QCheckBox {
    spacing: 8px;
    color: #eaeaea;
}

QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 4px;
    border: 2px solid #2a2a4a;
    background-color: #0f3460;
}

QCheckBox::indicator:checked {
    background-color: #e94560;
    border-color: #e94560;
}

/* ================================================================== */
/*  Radio Buttons                                                      */
/* ================================================================== */

QRadioButton {
    spacing: 8px;
    color: #eaeaea;
    padding: 2px 0px;
}

QRadioButton::indicator {
    width: 16px;
    height: 16px;
    border-radius: 8px;
    border: 2px solid #2a2a4a;
    background-color: #0f3460;
}

QRadioButton::indicator:hover {
    border-color: #e94560;
}

QRadioButton::indicator:checked {
    background-color: #e94560;
    border-color: #e94560;
}

/* Checkable QGroupBox — custom indicator */
QGroupBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 2px solid #2a2a4a;
    background-color: #0f3460;
}

QGroupBox::indicator:checked {
    background-color: #e94560;
    border-color: #e94560;
}

QGroupBox::indicator:unchecked {
    background-color: #0f3460;
}

/* ================================================================== */
/*  Labels                                                             */
/* ================================================================== */

QLabel {
    color: #eaeaea;
    background-color: transparent;
}

QLabel#label_header {
    font-size: 15px;
    font-weight: bold;
    color: #e94560;
}

QLabel#label_value {
    font-size: 14px;
    font-family: "Consolas", "Fira Code", monospace;
    color: #00e676;
}

/* ================================================================== */
/*  Text Edit (Console)                                                */
/* ================================================================== */

QTextEdit {
    background-color: #0d1117;
    color: #c9d1d9;
    border: 1px solid #2a2a4a;
    border-radius: 6px;
    font-family: "Consolas", "Fira Code", "Courier New", monospace;
    font-size: 12px;
    padding: 8px;
    selection-background-color: #264f78;
}

/* ================================================================== */
/*  Progress Bar                                                       */
/* ================================================================== */

QProgressBar {
    border: 1px solid #2a2a4a;
    border-radius: 8px;
    background-color: #0f3460;
    text-align: center;
    color: #eaeaea;
    font-weight: bold;
    min-height: 22px;
}

QProgressBar::chunk {
    background: qlineargradient(
        x1:0, y1:0, x2:1, y2:0,
        stop:0 #e94560, stop:1 #ff6b6b
    );
    border-radius: 7px;
}

/* ================================================================== */
/*  Status Bar                                                         */
/* ================================================================== */

QStatusBar {
    background-color: #16213e;
    color: #a0a0b0;
    border-top: 1px solid #2a2a4a;
    font-size: 12px;
    padding: 2px 8px;
}

QStatusBar::item {
    border: none;
}

/* ================================================================== */
/*  Scroll Bars                                                        */
/* ================================================================== */

QScrollBar:vertical {
    background-color: #1a1a2e;
    width: 10px;
    border-radius: 5px;
}

QScrollBar::handle:vertical {
    background-color: #2a2a4a;
    min-height: 30px;
    border-radius: 5px;
}

QScrollBar::handle:vertical:hover {
    background-color: #e94560;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    background-color: #1a1a2e;
    height: 10px;
    border-radius: 5px;
}

QScrollBar::handle:horizontal {
    background-color: #2a2a4a;
    min-width: 30px;
    border-radius: 5px;
}

QScrollBar::handle:horizontal:hover {
    background-color: #e94560;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}

/* ================================================================== */
/*  Separator / Frame                                                  */
/* ================================================================== */

QFrame#plot_placeholder {
    background-color: #0d1117;
    border: 2px dashed #2a2a4a;
    border-radius: 8px;
    min-height: 150px;
}

/* ================================================================== */
/*  Tool Tips                                                          */
/* ================================================================== */

QToolTip {
    background-color: #16213e;
    color: #eaeaea;
    border: 1px solid #e94560;
    border-radius: 4px;
    padding: 6px;
    font-size: 12px;
}

/* ================================================================== */
/*  List Widget / Splitter (Interactive Plot Viewer)                   */
/* ================================================================== */

QListWidget {
    background-color: #0d1117;
    color: #c9d1d9;
    border: 1px solid #2a2a4a;
    border-radius: 6px;
    padding: 4px;
}

QListWidget::item {
    background-color: transparent;
    border-radius: 4px;
    padding: 6px;
    margin: 2px 0px;
}

QListWidget::item:hover {
    background-color: #16213e;
    color: #e94560;
}

QListWidget::item:selected {
    background-color: #0f3460;
    color: #ffffff;
    border: 1px solid #e94560;
}

QListWidget::indicator {
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 2px solid #2a2a4a;
    background-color: #0f3460;
}

QListWidget::indicator:checked {
    background-color: #e94560;
    border-color: #e94560;
}

QSplitter::handle {
    background-color: #1a1a2e;
}

QSplitter::handle:vertical {
    height: 2px;
}

QSplitter::handle:horizontal {
    width: 4px;
}
"""
