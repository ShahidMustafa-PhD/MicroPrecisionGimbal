"""
Interactive Plot Viewer Widget - Split panel with checkable list, matplotlib canvas, and PDF export.
"""

from pathlib import Path
import re

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSplitter,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QFileDialog,
    QMessageBox,
)
from PySide6.QtCore import Qt

import matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure


class InteractivePlotViewer(QWidget):
    """
    Split-panel widget for interacting with and saving Matplotlib figures.
    Left: Checkable QListWidget of plot names + Export buttons.
    Right: Live FigureCanvasQTAgg with NavigationToolbar2QT.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figures_map: dict[str, Figure] = {}
        
        self.canvas = None
        self.toolbar = None

        self._build_ui()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        self.splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(self.splitter)

        # ---------------------------------------------------------
        #  Left Panel: List + Actions
        # ---------------------------------------------------------
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 10, 0)

        lbl_header = QLabel("📊 Generated Plots")
        lbl_header.setStyleSheet("font-weight: bold; font-size: 14px;")
        left_layout.addWidget(lbl_header)

        self.list_widget = QListWidget()
        self.list_widget.itemClicked.connect(self.on_plot_selected)
        left_layout.addWidget(self.list_widget)

        # Buttons
        btn_layout = QVBoxLayout()
        btn_layout.setSpacing(8)
        
        self.btn_save_selected = QPushButton("💾 Save Selected to PDF")
        self.btn_save_selected.clicked.connect(self.on_save_selected_clicked)
        self.btn_save_selected.setMinimumHeight(35)
        btn_layout.addWidget(self.btn_save_selected)

        self.btn_save_all = QPushButton("📁 Save All to PDF")
        self.btn_save_all.clicked.connect(self.on_save_all_clicked)
        self.btn_save_all.setMinimumHeight(35)
        btn_layout.addWidget(self.btn_save_all)
        
        left_layout.addLayout(btn_layout)

        # ---------------------------------------------------------
        #  Right Panel: Active Canvas
        # ---------------------------------------------------------
        right_panel = QWidget()
        self.right_layout = QVBoxLayout(right_panel)
        self.right_layout.setContentsMargins(0, 0, 0, 0)
        self.right_layout.setSpacing(0)

        # Placeholder
        self.lbl_placeholder = QLabel("Run a simulation to view figures here.")
        self.lbl_placeholder.setAlignment(Qt.AlignCenter)
        self.lbl_placeholder.setStyleSheet("color: #7f8c8d; font-size: 16px; font-style: italic;")
        self.right_layout.addWidget(self.lbl_placeholder)

        # ---------------------------------------------------------
        #  Assemble
        # ---------------------------------------------------------
        self.splitter.addWidget(left_panel)
        self.splitter.addWidget(right_panel)

        # Dimensions: left ~280px, right takes rest
        self.splitter.setSizes([280, 800])

        self._set_ui_state(False)

    def _set_ui_state(self, has_data: bool):
        """Toggle placeholder visibility and button enablement."""
        if self.toolbar:
            self.toolbar.setVisible(has_data)
        if self.canvas:
            self.canvas.setVisible(has_data)
            
        self.btn_save_selected.setEnabled(has_data)
        self.btn_save_all.setEnabled(has_data)
        self.lbl_placeholder.setVisible(not has_data)

    def populate(self, figures: dict[str, Figure]):
        """Populate the list widget keys and display the first figure."""
        self.list_widget.blockSignals(True)
        self.list_widget.clear()

        self.figures_map = figures
        for fig_name in self.figures_map.keys():
            item = QListWidgetItem(fig_name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)  # Checked by default
            self.list_widget.addItem(item)

        self.list_widget.blockSignals(False)

        if self.list_widget.count() > 0:
            self.list_widget.setCurrentRow(0)
            first_item = self.list_widget.item(0)
            self.on_plot_selected(first_item)
            self._set_ui_state(True)
        else:
            self.clear_all()

    def clear_all(self):
        """Wipes figures safely and restores placeholder state."""
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        self.list_widget.blockSignals(False)
        self.figures_map.clear()
        
        if self.toolbar:
            self.right_layout.removeWidget(self.toolbar)
            self.toolbar.deleteLater()
            self.toolbar = None
            
        if self.canvas:
            self.right_layout.removeWidget(self.canvas)
            self.canvas.deleteLater()
            self.canvas = None
            
        self._set_ui_state(False)

    # ------------------------------------------------------------------ #
    #  Viewer Interactivity Slots                                        #
    # ------------------------------------------------------------------ #

    def on_plot_selected(self, item: QListWidgetItem):
        """Extract figure from dictionary and swap it onto the active canvas natively."""
        fig_key = item.text()
        if fig_key not in self.figures_map:
            return

        new_fig = self.figures_map[fig_key]

        # Properly replace the canvas and toolbar widgets instead of hacking the Figure reference.
        # This prevents Tkinter/Agg thread conflicts and avoids PyQt memory leaks.
        
        # 1. Clean up old widgets
        if self.toolbar:
            self.right_layout.removeWidget(self.toolbar)
            self.toolbar.deleteLater()
            self.toolbar = None
            
        if self.canvas:
            self.right_layout.removeWidget(self.canvas)
            self.canvas.deleteLater()
            self.canvas = None
            
        # 2. Rebuild with new figure
        self.canvas = FigureCanvasQTAgg(new_fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        
        # 3. Add to layout (Toolbar at top, Canvas below, Placeholder remains hidden at bottom)
        self.right_layout.insertWidget(0, self.toolbar)
        self.right_layout.insertWidget(1, self.canvas)
        
        # 4. Hide placeholder and force redraw
        self.lbl_placeholder.setVisible(False)
        self.canvas.draw_idle()

    # ------------------------------------------------------------------ #
    #  PDF Export Slots                                                  #
    # ------------------------------------------------------------------ #

    def _sanitize_filename(self, text: str) -> str:
        """Strip bad chars from plot name to make a safe filename."""
        # Replace spaces and special chars, keeping alphanumeric and underscores
        text = text.replace(' ', '_').replace('+', 'plus').replace('-', '_')
        safe = re.sub(r'[^A-Za-z0-9_]', '', text)
        # Collapse multiple underscores
        safe = re.sub(r'_+', '_', safe)
        return safe.strip('_')

    def on_save_selected_clicked(self):
        """Exports only the checked items in the QListWidget to PDF."""
        checked_keys = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.Checked:
                checked_keys.append(item.text())

        if not checked_keys:
            QMessageBox.warning(self, "No Selection", "Please check at least one plot in the list to save.")
            return

        self._export_figures(checked_keys)

    def on_save_all_clicked(self):
        """Exports every figure in the dictionary to PDF."""
        all_keys = list(self.figures_map.keys())
        if not all_keys:
            QMessageBox.information(self, "Empty", "No plots available to save.")
            return
            
        self._export_figures(all_keys)

    def _export_figures(self, keys_to_export: list[str]):
        """Runs the directory prompt and export loop strictly isolating export logic."""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Destination Directory for PDF Export",
            "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )

        if not dir_path:
            return  # User cancelled

        out_dir = Path(dir_path)
        saved_count = 0

        try:
            for key in keys_to_export:
                if key not in self.figures_map:
                    continue
                
                fig = self.figures_map[key]
                safe_name = self._sanitize_filename(key)
                filepath = out_dir / f"{safe_name}.pdf"
                
                # CRITICAL: bbox_inches='tight' forces matplotlib to evaluate the layout
                # before saving, ensuring axes and labels aren't cut off inside the PDF canvas
                fig.savefig(filepath, format='pdf', dpi=300, bbox_inches='tight')
                saved_count += 1

            QMessageBox.information(
                self,
                "Export Complete",
                f"Successfully saved {saved_count} publication-ready plot(s) to:\n\n{out_dir}"
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Export Failed",
                f"An error occurred while generating PDFs:\n\n{str(e)}"
            )
