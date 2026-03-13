"""
Research-Quality Comparative Plotter for Three-Way Controller Analysis.

This module provides publication-quality visualization for comparing
PID, Feedback Linearization (FBL), and FBL+NDOB controllers in the
lasercom gimbal pointing system.

Figures GeneratedF
-----------------
1. Angular Position Tracking (Az & El)
2. Tracking Error with Handover Thresholds (THE MONEY SHOT)
3. Control Effort (Torques) & Disturbance Estimation
4. Angular Velocities
5. Phase Plane (q vs qdot)
6. LOS Error Time History
7. Performance Summary Bar Charts
8. State Estimator (EKF) Performance
9. Fine Steering Mirror (FSM) Performance
10. Internal Control Signal & Disturbance Observer
11. EKF Diagnostics (Innovation & Covariance)
12. Environmental Disturbance Torques
13. Disturbance Statistics & PSD

Style Consistency
-----------------
All plots match the visual style from simulation_runner.py plot_results()
to ensure consistency across technical documentation.

Author: Dr. S. Shahid Mustafa
Version: 1.0.0
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
import time

from lasercom_digital_twin.core.plots.style_config import (
    PlotStyleConfig,
    ControllerColors,
    AxisColors,
    DisturbanceColors,
    configure_matplotlib_defaults
)
from lasercom_digital_twin.core.plots.metrics_utils import compute_tracking_metrics
from lasercom_digital_twin.core.plots.interactive_plotter import (
    InteractiveFigureManager,
    InteractiveStyleConfig,
    make_interactive
)
from lasercom_digital_twin.core.performance_metrics import StrokeMetrics, StrokeMetricsResult


class ResearchComparisonPlotter:
    """
    Publication-quality plotter for three-way controller comparison.
    
    This class encapsulates all plotting logic for comparing PID, FBL, and
    FBL+NDOB controllers, generating 13 research-grade figures suitable for
    IEEE/AIAA journal submission.
    
    Parameters
    ----------
    style : PlotStyleConfig, optional
        Style configuration. If None, uses defaults.
    save_figures : bool
        Whether to save figures to disk
    show_figures : bool
        Whether to display figures interactively
        
    Example
    -------
    ```python
    plotter = ResearchComparisonPlotter()
    plotter.plot_all(results_pid, results_fbl, results_ndob, 10.0, 15.0)
    ```
    """
    
    def __init__(
        self,
        style: Optional[PlotStyleConfig] = None,
        save_figures: bool = True,
        show_figures: bool = True,
        interactive: bool = True
    ):
        """Initialize plotter with style configuration.
        
        Parameters
        ----------
        interactive : bool
            If True, all figures will be interactive with zoom regions,
            annotations, and professional editing capabilities
        """
        self.style = style or PlotStyleConfig()
        self.save_figures = save_figures
        self.show_figures = show_figures
        self.interactive = interactive
        
        # Ensure output directory exists
        if self.save_figures:
            self.style.output_dir.mkdir(exist_ok=True)
        
        # Store generated figures for access
        self.figures: Dict[str, plt.Figure] = {}
        
        # Store interactive managers for each figure
        self.interactive_managers: Dict[str, InteractiveFigureManager] = {}
    
    def _get_layout_mode(self):
        """Determine layout mode based on interactive setting.
        
        Returns
        -------
        bool or None
            False if interactive (disable constrained_layout for toolbar),
            True if non-interactive (use matplotlib's automatic layout)
        """
        # Interactive mode requires manual layout for toolbar space
        # Non-interactive mode can use constrained_layout for better spacing
        return False if self.interactive else True
    
    def _make_figure_interactive(self, fig: plt.Figure, axes, fig_name: str) -> InteractiveFigureManager:
        """Make a figure interactive with enhanced controls.
        
        Uses the make_interactive() factory function following the standard pattern
        from test_zoom_deletion.py for consistent behavior across the project.
        
        CRITICAL: Figures must be created with constrained_layout=False BEFORE
        calling this method. The InteractiveFigureManager checks layout mode
        during __init__ and will disable toolbar if constrained_layout is active.
        
        Parameters
        ----------
        fig : plt.Figure
            The matplotlib figure to enhance (must have constrained_layout=False)
        axes : plt.Axes or list of plt.Axes
            The axes in the figure
        fig_name : str
            Name identifier for the figure
            
        Returns
        -------
        InteractiveFigureManager
            The interactive manager instance
        """
        if not self.interactive:
            return None
        
        # Use make_interactive() factory function (standard pattern from test_zoom_deletion.py)
        # This ensures consistent behavior across all interactive plots in the project
        # Pass style parameters as kwargs, not as a style object
        manager = make_interactive(
            fig=fig,
            axes=axes,
            save_dir=str(self.style.output_dir / 'interactive'),
            # Style parameters passed as kwargs
            vline_color='#e74c3c',         # Red vertical lines
            hline_color='#3498db',         # Blue horizontal lines
            zoom_rect_color='#2ecc71',     # Green zoom regions
            selection_color='#ff6600',     # Orange selection
            annotation_linewidth=1.5,
            selection_linewidth=3.0,
            zoom_rect_alpha=0.3,
            save_dpi=300,                  # Match research quality
        )
        
        return manager
    
    def plot_all(
        self,
        results_pid: Dict,
        results_fbl: Dict,
        results_ndob: Dict,
        target_az_deg: float,
        target_el_deg: float
    ) -> Dict[str, plt.Figure]:
        """
        Generate all 13 comparative figures.
        
        Parameters
        ----------
        results_pid : Dict
            PID controller simulation results
        results_fbl : Dict
            FBL controller simulation results
        results_ndob : Dict
            FBL+NDOB controller simulation results
        target_az_deg : float
            Target azimuth in degrees
        target_el_deg : float
            Target elevation in degrees
            
        Returns
        -------
        Dict[str, plt.Figure]
            Dictionary of generated figures by name
        """
        # Store inputs for individual plot methods
        self._results_pid = results_pid
        self._results_fbl = results_fbl
        self._results_ndob = results_ndob
        self._target_az_deg = target_az_deg
        self._target_el_deg = target_el_deg
        self._target_az_rad = np.deg2rad(target_az_deg)
        self._target_el_rad = np.deg2rad(target_el_deg)
        
        # Extract time vectors (used by most plots)
        self._t_pid = results_pid['log_arrays']['time']
        self._t_fbl = results_fbl['log_arrays']['time']
        self._t_ndob = results_ndob['log_arrays']['time']
        
        # Generate all figures
        self.figures['fig1_position'] = self._plot_position_tracking()
        self.figures['fig2_error'] = self._plot_tracking_error()
        self.figures['fig3_torque'] = self._plot_control_torques()
        self.figures['fig4_velocity'] = self._plot_velocities()
        self.figures['fig5_phase'] = self._plot_phase_plane()
        self.figures['fig6_los'] = self._plot_los_errors()
        self.figures['fig7_summary'] = self._plot_performance_summary()
        self.figures['fig8_ekf'] = self._plot_ekf_performance()
        self.figures['fig9_fsm'] = self._plot_fsm_performance()
        self.figures['fig10_internal'] = self._plot_internal_signals()
        self.figures['fig11_ekf_diag'] = self._plot_ekf_diagnostics()
        self.figures['fig12_disturbance'] = self._plot_disturbance_torques()
        self.figures['fig13_statistics'] = self._plot_disturbance_statistics()
        self.figures['fig14_stroke_metrics'] = self._plot_stroke_consumption()
        self.figures['fig15_stroke_summary'] = self._plot_stroke_margin_summary()
        self.figures['fig16_benchmark_table'] = self._plot_benchmark_table()
        
        # Save figures FIRST (before making interactive) to get clean PDFs without buttons
        if self.save_figures:
            self._save_all_figures()
        
        # Make figures interactive AFTER saving (so buttons don't appear in saved PDFs)
        if self.interactive:
            print("\n[OK] Enhancing figures with interactive capabilities...")
            print("     - Zoom regions (Z key)")
            print("     - Vertical/horizontal lines (V/H keys)")
            print("     - Mouse-based selection and deletion")
            print("     - Professional annotation tools")
            print("     - Press ? in any figure for full help")
            
            # Make each figure interactive with its axes
            fig_axes_map = {
                'fig1_position': self.figures['fig1_position'].get_axes(),
                'fig2_error': self.figures['fig2_error'].get_axes(),
                'fig3_torque': self.figures['fig3_torque'].get_axes(),
                'fig4_velocity': self.figures['fig4_velocity'].get_axes(),
                'fig5_phase': self.figures['fig5_phase'].get_axes(),
                'fig6_los': self.figures['fig6_los'].get_axes(),
                'fig7_summary': self.figures['fig7_summary'].get_axes(),
                'fig8_ekf': self.figures['fig8_ekf'].get_axes(),
                'fig9_fsm': self.figures['fig9_fsm'].get_axes(),
                'fig10_internal': self.figures['fig10_internal'].get_axes(),
                'fig11_ekf_diag': self.figures['fig11_ekf_diag'].get_axes(),
                'fig12_disturbance': self.figures['fig12_disturbance'].get_axes(),
                'fig13_statistics': self.figures['fig13_statistics'].get_axes(),
                'fig14_stroke_metrics': self.figures['fig14_stroke_metrics'].get_axes(),
                'fig15_stroke_summary': self.figures['fig15_stroke_summary'].get_axes(),
            }
            
            for fig_name, axes in fig_axes_map.items():
                if fig_name in self.figures:
                    manager = self._make_figure_interactive(
                        self.figures[fig_name],
                        axes,
                        fig_name
                    )
                    if manager:
                        self.interactive_managers[fig_name] = manager
            
            print(f"[OK] Made {len(self.interactive_managers)} figures interactive")
        
        # Show figures
        if self.show_figures:
            if self.interactive and len(self.interactive_managers) > 0:
                print("\n[OK] Displaying interactive figures...")
                print("     Press Ctrl+C in terminal to close all figures")
                print("     Interactive controls: Z(zoom), V(vline), H(hline), U(undo), S(save)")
                print("     Click green rectangles to select (orange), then press Delete")
                # Use first manager's show() to display all figures
                # (they all share the same matplotlib backend)
                first_manager = next(iter(self.interactive_managers.values()))
                first_manager.show()
            else:
                plt.show()
        
        return self.figures
    
    def _plot_position_tracking(self) -> plt.Figure:
        """Figure 1: Gimbal Position Tracking (Az & El with Commands)."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.style.get_figure_size('2x1'),
                                        sharex=True, constrained_layout=self._get_layout_mode())
        
        lw = self.style.linewidth_primary
        alpha = self.style.alpha_primary
        
        # Azimuth Position
        ax1.plot(self._t_pid, np.rad2deg(self._results_pid['log_arrays']['q_az']),
                 color=ControllerColors.PID, linewidth=lw, label='PID', alpha=alpha)
        ax1.plot(self._t_fbl, np.rad2deg(self._results_fbl['log_arrays']['q_az']),
                 color=ControllerColors.FBL, linewidth=lw, label='FBL', alpha=alpha)
        ax1.plot(self._t_ndob, np.rad2deg(self._results_ndob['log_arrays']['q_az']),
                 color=ControllerColors.FBL_NDOB, linewidth=lw, label='FBL+NDOB', alpha=alpha)
        
        # Target/Command
        if 'target_az' in self._results_pid['log_arrays']:
            ax1.plot(self._t_pid, np.rad2deg(self._results_pid['log_arrays']['target_az']),
                     color=ControllerColors.COMMAND, linewidth=lw, linestyle='--',
                     label='Command', alpha=0.7)
        else:
            ax1.plot(self._t_pid, np.full_like(self._t_pid, self._target_az_deg),
                     color=ControllerColors.COMMAND, linewidth=lw, linestyle='--',
                     label='Command', alpha=0.7)
        
        ax1.set_ylabel('Azimuth Angle [deg]', fontsize=self.style.axis_label_fontsize,
                       fontweight='bold')
        #ax1.set_title('Gimbal Azimuth Position', fontsize=self.style.title_fontsize,
        #              fontweight='bold')
        ax1.legend(loc='best', fontsize=self.style.legend_fontsize)
        ax1.grid(True, alpha=self.style.grid_alpha, linestyle=self.style.grid_linestyle)
        
        # Elevation Position
        ax2.plot(self._t_pid, np.rad2deg(self._results_pid['log_arrays']['q_el']),
                 color=ControllerColors.PID, linewidth=lw, label='PID', alpha=alpha)
        ax2.plot(self._t_fbl, np.rad2deg(self._results_fbl['log_arrays']['q_el']),
                 color=ControllerColors.FBL, linewidth=lw, label='FBL', alpha=alpha)
        ax2.plot(self._t_ndob, np.rad2deg(self._results_ndob['log_arrays']['q_el']),
                 color=ControllerColors.FBL_NDOB, linewidth=lw, label='FBL+NDOB', alpha=alpha)
        
        if 'target_el' in self._results_pid['log_arrays']:
            ax2.plot(self._t_pid, np.rad2deg(self._results_pid['log_arrays']['target_el']),
                     color=ControllerColors.COMMAND, linewidth=lw, linestyle='--',
                     label='Command', alpha=0.7)
        else:
            ax2.plot(self._t_pid, np.full_like(self._t_pid, self._target_el_deg),
                     color=ControllerColors.COMMAND, linewidth=lw, linestyle='--',
                     label='Command', alpha=0.7)
        
        ax2.set_ylabel('Elevation Angle [deg]', fontsize=self.style.axis_label_fontsize,
                       fontweight='bold')
        ax2.set_xlabel('Time [s]', fontsize=self.style.axis_label_fontsize, fontweight='bold')
        #ax2.set_title('Gimbal Elevation Position', fontsize=self.style.title_fontsize,
        #              fontweight='bold')
        ax2.legend(loc='best', fontsize=self.style.legend_fontsize)
        ax2.grid(True, alpha=self.style.grid_alpha, linestyle=self.style.grid_linestyle)
        
       # fig.suptitle('Gimbal Position Tracking', fontsize=self.style.suptitle_fontsize,
                 #    fontweight='bold')
        
        return fig
    
    def _plot_tracking_error(self) -> plt.Figure:
        """Figure 2: Tracking Error with Handover Thresholds (THE MONEY SHOT)."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.style.get_figure_size('2x1'),
                                        sharex=True, constrained_layout=self._get_layout_mode())
        
        lw = self.style.linewidth_primary
        alpha = self.style.alpha_primary
        
        # Compute errors
        if 'target_az' in self._results_pid['log_arrays']:
            error_az_pid = np.abs(self._results_pid['log_arrays']['q_az'] - 
                                  self._results_pid['log_arrays']['target_az'])
            error_az_fbl = np.abs(self._results_fbl['log_arrays']['q_az'] - 
                                  self._results_fbl['log_arrays']['target_az'])
            error_az_ndob = np.abs(self._results_ndob['log_arrays']['q_az'] - 
                                   self._results_ndob['log_arrays']['target_az'])
        else:
            error_az_pid = np.abs(self._results_pid['log_arrays']['q_az'] - self._target_az_rad)
            error_az_fbl = np.abs(self._results_fbl['log_arrays']['q_az'] - self._target_az_rad)
            error_az_ndob = np.abs(self._results_ndob['log_arrays']['q_az'] - self._target_az_rad)
        
        # Azimuth Error
        ax1.plot(self._t_pid, np.rad2deg(error_az_pid),
                 color=ControllerColors.PID, linewidth=lw, label='PID', alpha=alpha)
        ax1.plot(self._t_fbl, np.rad2deg(error_az_fbl),
                 color=ControllerColors.FBL, linewidth=lw, label='FBL', alpha=alpha)
        ax1.plot(self._t_ndob, np.rad2deg(error_az_ndob),
                 color=ControllerColors.FBL_NDOB, linewidth=lw, label='FBL+NDOB', alpha=alpha)
        
        # Threshold lines and shaded regions
        limit_rad_az = np.array(self._results_ndob['log_arrays'].get('fsm_stroke_limit_rad', np.full(len(self._t_ndob), 0.010)))
        limit_deg_az = np.rad2deg(limit_rad_az)
        
        ax1.plot(self._t_ndob, limit_deg_az, color=ControllerColors.HANDOVER, linewidth=lw*1.5, linestyle='--',
                 label='FSM Stroke / QPD Limit')
        
        ylim1 = ax1.get_ylim()
        ax1.fill_between(self._t_ndob, 1e-10, limit_deg_az, color='lightgreen', alpha=0.15, label='LOS Acquired (QPD Active)')
        ax1.fill_between(self._t_ndob, limit_deg_az, max(180.0, ylim1[1]), color='lightcoral', alpha=0.15, label='Connection Lost (Out of Reach)')
        ax1.set_ylim(bottom=max(1e-4, ylim1[0]), top=max(limit_deg_az[0]*2, ylim1[1]))
        
        ax1.set_ylabel('Azimuth Error [deg]', fontsize=self.style.axis_label_fontsize,
                       fontweight='bold')
       # ax1.set_title('Azimuth Tracking Error (with FSM Handover Threshold)',
                      #fontsize=self.style.title_fontsize, fontweight='bold')
        ax1.legend(loc='best', fontsize=self.style.legend_fontsize)
        ax1.grid(True, alpha=self.style.grid_alpha, linestyle=self.style.grid_linestyle)
        ax1.set_yscale('log')
        
        # Elevation Error
        if 'target_el' in self._results_pid['log_arrays']:
            error_el_pid = np.abs(self._results_pid['log_arrays']['q_el'] - 
                                  self._results_pid['log_arrays']['target_el'])
            error_el_fbl = np.abs(self._results_fbl['log_arrays']['q_el'] - 
                                  self._results_fbl['log_arrays']['target_el'])
            error_el_ndob = np.abs(self._results_ndob['log_arrays']['q_el'] - 
                                   self._results_ndob['log_arrays']['target_el'])
        else:
            error_el_pid = np.abs(self._results_pid['log_arrays']['q_el'] - self._target_el_rad)
            error_el_fbl = np.abs(self._results_fbl['log_arrays']['q_el'] - self._target_el_rad)
            error_el_ndob = np.abs(self._results_ndob['log_arrays']['q_el'] - self._target_el_rad)
        
        ax2.plot(self._t_pid, np.rad2deg(error_el_pid),
                 color=ControllerColors.PID, linewidth=lw, label='PID', alpha=alpha)
        ax2.plot(self._t_fbl, np.rad2deg(error_el_fbl),
                 color=ControllerColors.FBL, linewidth=lw, label='FBL', alpha=alpha)
        ax2.plot(self._t_ndob, np.rad2deg(error_el_ndob),
                 color=ControllerColors.FBL_NDOB, linewidth=lw, label='FBL+NDOB', alpha=alpha)
        
        # Threshold lines and shaded regions
        limit_rad_el = np.array(self._results_ndob['log_arrays'].get('fsm_stroke_limit_rad', np.full(len(self._t_ndob), 0.010)))
        limit_deg_el = np.rad2deg(limit_rad_el)
        
        ax2.plot(self._t_ndob, limit_deg_el, color=ControllerColors.HANDOVER, linewidth=lw*1.5, linestyle='--',
                 label='FSM Stroke / QPD Limit')
        
        ylim2 = ax2.get_ylim()
        ax2.fill_between(self._t_ndob, 1e-10, limit_deg_el, color='lightgreen', alpha=0.15, label='LOS Acquired (QPD Active)')
        ax2.fill_between(self._t_ndob, limit_deg_el, max(180.0, ylim2[1]), color='lightcoral', alpha=0.15, label='Connection Lost (Out of Reach)')
        ax2.set_ylim(bottom=max(1e-4, ylim2[0]), top=max(limit_deg_el[0]*2, ylim2[1]))
        
        ax2.set_ylabel('Elevation Error [deg]', fontsize=self.style.axis_label_fontsize,
                       fontweight='bold')
        ax2.set_xlabel('Time [s]', fontsize=self.style.axis_label_fontsize, fontweight='bold')
        #ax2.set_title('Elevation Tracking Error (with FSM Handover Threshold)',
                     # fontsize=self.style.title_fontsize, fontweight='bold')
        ax2.legend(loc='best', fontsize=self.style.legend_fontsize)
        ax2.grid(True, alpha=self.style.grid_alpha, linestyle=self.style.grid_linestyle)
        ax2.set_yscale('log')
        
        ##fig.suptitle('Tracking Error with Precision Thresholds',
                     #fontsize=self.style.suptitle_fontsize, fontweight='bold')
        
        return fig
    
    def _plot_control_torques(self) -> plt.Figure:
        """Figure 3: Control Torques & NDOB Disturbance Estimation."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.style.get_figure_size('2x2'),
                                                      constrained_layout=self._get_layout_mode())
        
        lw = self.style.linewidth_primary
        alpha = self.style.alpha_primary
        
        # Motor torque saturation limits
        tau_max = 0.0
        tau_min = -0.0
        
        # Azimuth Torque
        ax1.plot(self._t_pid, self._results_pid['log_arrays']['torque_az'],
                 color=ControllerColors.PID, linewidth=lw, label='PID', alpha=alpha)
        ax1.plot(self._t_fbl, self._results_fbl['log_arrays']['torque_az'],
                 color=ControllerColors.FBL, linewidth=lw, label='FBL', alpha=alpha)
        ax1.plot(self._t_ndob, self._results_ndob['log_arrays']['torque_az'],
                 color=ControllerColors.FBL_NDOB, linewidth=lw, label='FBL+NDOB', alpha=alpha)
        ax1.axhline(tau_max, color='red', linewidth=lw, linestyle=':',
                    alpha=self.style.alpha_threshold, label='Saturation Limit')
        ax1.axhline(tau_min, color='red', linewidth=lw, linestyle=':', alpha=self.style.alpha_threshold)
        ax1.axhline(0, color='black', linewidth=lw, linestyle='--', alpha=0.5)
        ax1.set_ylabel('Azimuth Torque [N·m]', fontsize=self.style.axis_label_fontsize, fontweight='bold')
        #ax1.set_title('Azimuth Motor Control Effort', fontsize=self.style.title_fontsize, fontweight='bold')
        ax1.legend(loc='best', fontsize=self.style.legend_fontsize)
        ax1.grid(True, alpha=self.style.grid_alpha, linestyle=self.style.grid_linestyle)
        
        # Elevation Torque
        ax2.plot(self._t_pid, self._results_pid['log_arrays']['torque_el'],
                 color=ControllerColors.PID, linewidth=lw, label='PID', alpha=alpha)
        ax2.plot(self._t_fbl, self._results_fbl['log_arrays']['torque_el'],
                 color=ControllerColors.FBL, linewidth=lw, label='FBL', alpha=alpha)
        ax2.plot(self._t_ndob, self._results_ndob['log_arrays']['torque_el'],
                 color=ControllerColors.FBL_NDOB, linewidth=lw, label='FBL+NDOB', alpha=alpha)
        ax2.axhline(tau_max, color='red', linewidth=lw, linestyle=':',
                    alpha=self.style.alpha_threshold, label='Saturation Limit')
        ax2.axhline(tau_min, color='red', linewidth=lw, linestyle=':', alpha=self.style.alpha_threshold)
        ax2.axhline(0, color='black', linewidth=lw, linestyle='--', alpha=0.5)
        ax2.set_ylabel('Elevation Torque [N·m]', fontsize=self.style.axis_label_fontsize, fontweight='bold')
        #ax2.set_title('Elevation Motor Control Effort', fontsize=self.style.title_fontsize, fontweight='bold')
        ax2.legend(loc='best', fontsize=self.style.legend_fontsize)
        ax2.grid(True, alpha=self.style.grid_alpha, linestyle=self.style.grid_linestyle)
        
        # NDOB Disturbance Estimate - Azimuth
        friction_coef = 0.1
        if 'd_hat_ndob_az' in self._results_ndob['log_arrays']:
            d_hat_az = self._results_ndob['log_arrays']['d_hat_ndob_az']
            dq_az = self._results_ndob['log_arrays']['qd_az']
            d_true_az = friction_coef * dq_az
            
            ax3.plot(self._t_ndob, d_hat_az, color=ControllerColors.FBL_NDOB,
                     linewidth=lw, label='NDOB Estimate', alpha=alpha)
            ax3.plot(self._t_ndob, d_true_az, color=ControllerColors.GROUND_TRUTH,
                     linewidth=lw, linestyle='--', label='Ground Truth (Friction)', alpha=0.7)
            ax3.axhline(0, color='black', linewidth=lw, linestyle='--', alpha=0.5)
            ax3.set_ylabel('Disturbance [N·m]', fontsize=self.style.axis_label_fontsize, fontweight='bold')
            ax3.set_xlabel('Time [s]', fontsize=self.style.axis_label_fontsize, fontweight='bold')
            #ax3.set_title('Azimuth Disturbance Estimation Accuracy',
            #              fontsize=self.style.title_fontsize, fontweight='bold')
            ax3.legend(loc='best', fontsize=self.style.legend_fontsize)
        else:
            ax3.text(0.5, 0.5, 'NDOB Not Enabled', ha='center', va='center',
                     transform=ax3.transAxes, fontsize=self.style.axis_label_fontsize)
        ax3.grid(True, alpha=self.style.grid_alpha, linestyle=self.style.grid_linestyle)
        
        # NDOB Disturbance Estimate - Elevation
        if 'd_hat_ndob_el' in self._results_ndob['log_arrays']:
            d_hat_el = self._results_ndob['log_arrays']['d_hat_ndob_el']
            dq_el = self._results_ndob['log_arrays']['qd_el']
            d_true_el = friction_coef * dq_el
            
            ax4.plot(self._t_ndob, d_hat_el, color=ControllerColors.FBL_NDOB,
                     linewidth=lw, label='NDOB Estimate', alpha=alpha)
            ax4.plot(self._t_ndob, d_true_el, color=ControllerColors.GROUND_TRUTH,
                     linewidth=lw, linestyle='--', label='Ground Truth (Friction)', alpha=0.7)
            ax4.axhline(0, color='black', linewidth=lw, linestyle='--', alpha=0.5)
            ax4.set_ylabel('Disturbance [N·m]', fontsize=self.style.axis_label_fontsize, fontweight='bold')
            ax4.set_xlabel('Time [s]', fontsize=self.style.axis_label_fontsize, fontweight='bold')
            #ax4.set_title('Elevation Disturbance Estimation Accuracy',
            #              fontsize=self.style.title_fontsize, fontweight='bold')
            ax4.legend(loc='best', fontsize=self.style.legend_fontsize)
        else:
            ax4.text(0.5, 0.5, 'NDOB Not Enabled', ha='center', va='center',
                     transform=ax4.transAxes, fontsize=self.style.axis_label_fontsize)
        ax4.grid(True, alpha=self.style.grid_alpha, linestyle=self.style.grid_linestyle)
        
        #fig.suptitle('Motor Control Torques', fontsize=self.style.suptitle_fontsize, fontweight='bold')
        
        return fig
    
    def _plot_velocities(self) -> plt.Figure:
        """Figure 4: Gimbal Angular Velocities."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.style.get_figure_size('2x1'),
                                        sharex=True, constrained_layout=self._get_layout_mode())
        
        lw = self.style.linewidth_primary
        alpha = self.style.alpha_primary
        
        # Azimuth Velocity
        ax1.plot(self._t_pid, np.rad2deg(self._results_pid['log_arrays']['qd_az']),
                 color=ControllerColors.PID, linewidth=lw, label='PID', alpha=alpha)
        ax1.plot(self._t_fbl, np.rad2deg(self._results_fbl['log_arrays']['qd_az']),
                 color=ControllerColors.FBL, linewidth=lw, label='FBL', alpha=alpha)
        ax1.plot(self._t_ndob, np.rad2deg(self._results_ndob['log_arrays']['qd_az']),
                 color=ControllerColors.FBL_NDOB, linewidth=lw, label='FBL+NDOB', alpha=alpha)
        ax1.axhline(0, color='black', linewidth=lw, linestyle='--', alpha=0.5)
        ax1.set_ylabel('Azimuth Rate [deg/s]', fontsize=self.style.axis_label_fontsize, fontweight='bold')
        #ax1.set_title('Gimbal Azimuth Velocity', fontsize=self.style.title_fontsize, fontweight='bold')
        ax1.legend(loc='best', fontsize=self.style.legend_fontsize)
        ax1.grid(True, alpha=self.style.grid_alpha, linestyle=self.style.grid_linestyle)
        
        # Elevation Velocity
        ax2.plot(self._t_pid, np.rad2deg(self._results_pid['log_arrays']['qd_el']),
                 color=ControllerColors.PID, linewidth=lw, label='PID', alpha=alpha)
        ax2.plot(self._t_fbl, np.rad2deg(self._results_fbl['log_arrays']['qd_el']),
                 color=ControllerColors.FBL, linewidth=lw, label='FBL', alpha=alpha)
        ax2.plot(self._t_ndob, np.rad2deg(self._results_ndob['log_arrays']['qd_el']),
                 color=ControllerColors.FBL_NDOB, linewidth=lw, label='FBL+NDOB', alpha=alpha)
        ax2.axhline(0, color='black', linewidth=lw, linestyle='--', alpha=0.5)
        ax2.set_ylabel('Elevation Rate [deg/s]', fontsize=self.style.axis_label_fontsize, fontweight='bold')
        ax2.set_xlabel('Time [s]', fontsize=self.style.axis_label_fontsize, fontweight='bold')
        #ax2.set_title('Gimbal Elevation Velocity', fontsize=self.style.title_fontsize, fontweight='bold')
        ax2.legend(loc='best', fontsize=self.style.legend_fontsize)
        ax2.grid(True, alpha=self.style.grid_alpha, linestyle=self.style.grid_linestyle)
        
        #fig.suptitle('Gimbal Angular Velocities', fontsize=self.style.suptitle_fontsize, fontweight='bold')
        
        return fig
    
    def _plot_phase_plane(self) -> plt.Figure:
        """Figure 5: Phase Plane Trajectories."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.style.get_figure_size('1x2'),
                                        constrained_layout=self._get_layout_mode())
        
        lw = self.style.linewidth_primary
        
        # Azimuth Phase Plane
        ax1.plot(np.rad2deg(self._results_pid['log_arrays']['q_az']),
                 np.rad2deg(self._results_pid['log_arrays']['qd_az']),
                 color=ControllerColors.PID, linewidth=lw, label='PID', alpha=0.7)
        ax1.plot(np.rad2deg(self._results_fbl['log_arrays']['q_az']),
                 np.rad2deg(self._results_fbl['log_arrays']['qd_az']),
                 color=ControllerColors.FBL, linewidth=lw, label='FBL', alpha=0.7)
        ax1.plot(np.rad2deg(self._results_ndob['log_arrays']['q_az']),
                 np.rad2deg(self._results_ndob['log_arrays']['qd_az']),
                 color=ControllerColors.FBL_NDOB, linewidth=lw, label='FBL+NDOB', alpha=0.7)
        ax1.set_xlabel('Az Angle [deg]', fontsize=self.style.axis_label_fontsize, fontweight='bold')
        ax1.set_ylabel('Az Rate [deg/s]', fontsize=self.style.axis_label_fontsize, fontweight='bold')
        #ax1.set_title('Az Phase Plane', fontsize=self.style.title_fontsize, fontweight='bold')
        ax1.legend(loc='best', fontsize=self.style.legend_fontsize)
        ax1.grid(True, alpha=self.style.grid_alpha, linestyle=self.style.grid_linestyle)
        
        # Elevation Phase Plane
        ax2.plot(np.rad2deg(self._results_pid['log_arrays']['q_el']),
                 np.rad2deg(self._results_pid['log_arrays']['qd_el']),
                 color=ControllerColors.PID, linewidth=lw, label='PID', alpha=0.7)
        ax2.plot(np.rad2deg(self._results_fbl['log_arrays']['q_el']),
                 np.rad2deg(self._results_fbl['log_arrays']['qd_el']),
                 color=ControllerColors.FBL, linewidth=lw, label='FBL', alpha=0.7)
        ax2.plot(np.rad2deg(self._results_ndob['log_arrays']['q_el']),
                 np.rad2deg(self._results_ndob['log_arrays']['qd_el']),
                 color=ControllerColors.FBL_NDOB, linewidth=lw, label='FBL+NDOB', alpha=0.7)
        ax2.set_xlabel('El Angle [deg]', fontsize=self.style.axis_label_fontsize, fontweight='bold')
        ax2.set_ylabel('El Rate [deg/s]', fontsize=self.style.axis_label_fontsize, fontweight='bold')
        #ax2.set_title('El Phase Plane', fontsize=self.style.title_fontsize, fontweight='bold')
        ax2.legend(loc='best', fontsize=self.style.legend_fontsize)
        ax2.grid(True, alpha=self.style.grid_alpha, linestyle=self.style.grid_linestyle)
        
       # fig.suptitle('Gimbal Phase Plane', fontsize=self.style.suptitle_fontsize, fontweight='bold')
        
        return fig
    
    def _plot_los_errors(self) -> plt.Figure:
        """Figure 6: Line-of-Sight Errors."""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=self.style.get_figure_size('3x1'),
                                             sharex=True, constrained_layout=self._get_layout_mode())
        
        lw = self.style.linewidth_primary
        alpha = self.style.alpha_primary
        
        # LOS Error X (Tip)
        ax1.plot(self._t_pid, np.rad2deg(self._results_pid['log_arrays']['los_error_x']),
                 color=ControllerColors.PID, linewidth=lw, label='PID', alpha=alpha)
        ax1.plot(self._t_fbl, np.rad2deg(self._results_fbl['log_arrays']['los_error_x']),
                 color=ControllerColors.FBL, linewidth=lw, label='FBL', alpha=alpha)
        ax1.plot(self._t_ndob, np.rad2deg(self._results_ndob['log_arrays']['los_error_x']),
                 color=ControllerColors.FBL_NDOB, linewidth=lw, label='FBL+NDOB', alpha=alpha)
        ax1.axhline(0, color='black', linewidth=lw, linestyle='--', alpha=0.5)
        ax1.set_ylabel('LOS Error X [deg]', fontsize=self.style.axis_label_fontsize, fontweight='bold')
        #ax1.set_title('Line-of-Sight Error X-Axis', fontsize=self.style.title_fontsize, fontweight='bold')
        ax1.legend(loc='best', fontsize=self.style.legend_fontsize)
        ax1.grid(True, alpha=self.style.grid_alpha, linestyle=self.style.grid_linestyle)
        
        # LOS Error Y (Tilt)
        ax2.plot(self._t_pid, np.rad2deg(self._results_pid['log_arrays']['los_error_y']),
                 color=ControllerColors.PID, linewidth=lw, label='PID', alpha=alpha)
        ax2.plot(self._t_fbl, np.rad2deg(self._results_fbl['log_arrays']['los_error_y']),
                 color=ControllerColors.FBL, linewidth=lw, label='FBL', alpha=alpha)
        ax2.plot(self._t_ndob, np.rad2deg(self._results_ndob['log_arrays']['los_error_y']),
                 color=ControllerColors.FBL_NDOB, linewidth=lw, label='FBL+NDOB', alpha=alpha)
        ax2.axhline(0, color='black', linewidth=lw, linestyle='--', alpha=0.5)
        ax2.set_ylabel('LOS Error Y [deg]', fontsize=self.style.axis_label_fontsize, fontweight='bold')
        #ax2.set_title('Line-of-Sight Error Y-Axis', fontsize=self.style.title_fontsize, fontweight='bold')
        ax2.legend(loc='best', fontsize=self.style.legend_fontsize)
        ax2.grid(True, alpha=self.style.grid_alpha, linestyle=self.style.grid_linestyle)
        
        # Total LOS Error
        los_total_pid = np.rad2deg(np.sqrt(self._results_pid['log_arrays']['los_error_x']**2 + 
                                           self._results_pid['log_arrays']['los_error_y']**2))
        los_total_fbl = np.rad2deg(np.sqrt(self._results_fbl['log_arrays']['los_error_x']**2 + 
                                           self._results_fbl['log_arrays']['los_error_y']**2))
        los_total_ndob = np.rad2deg(np.sqrt(self._results_ndob['log_arrays']['los_error_x']**2 + 
                                            self._results_ndob['log_arrays']['los_error_y']**2))
        
        ax3.plot(self._t_pid, los_total_pid, color=ControllerColors.PID, linewidth=lw, label='PID', alpha=alpha)
        ax3.plot(self._t_fbl, los_total_fbl, color=ControllerColors.FBL, linewidth=lw, label='FBL', alpha=alpha)
        ax3.plot(self._t_ndob, los_total_ndob, color=ControllerColors.FBL_NDOB, linewidth=lw, label='FBL+NDOB', alpha=alpha)
        ax3.axhline(0, color='black', linewidth=lw, linestyle='--', alpha=0.5)
        ax3.set_ylabel('Total LOS Error [deg]', fontsize=self.style.axis_label_fontsize, fontweight='bold')
        ax3.set_xlabel('Time [s]', fontsize=self.style.axis_label_fontsize, fontweight='bold')
        #ax3.set_title('Total Line-of-Sight Error Magnitude', fontsize=self.style.title_fontsize, fontweight='bold')
        ax3.legend(loc='best', fontsize=self.style.legend_fontsize)
        ax3.grid(True, alpha=self.style.grid_alpha, linestyle=self.style.grid_linestyle)
        
        # Add RMS to suptitle
        rms_los_ndob = np.rad2deg(np.sqrt(np.mean(
            self._results_ndob['log_arrays']['los_error_x']**2 + 
            self._results_ndob['log_arrays']['los_error_y']**2)))
        fig.suptitle(f'Line-of-Sight Pointing Errors (RMS: {rms_los_ndob:.2f} deg)',
                     fontsize=self.style.suptitle_fontsize, fontweight='bold')
        
        return fig
    
    def _plot_performance_summary(self) -> plt.Figure:
        """Figure 7: Performance Summary Bar Charts."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.style.get_figure_size('2x2'),
                                                      constrained_layout=self._get_layout_mode())
        
        # Compute metrics
        metrics_pid = compute_tracking_metrics(self._results_pid, self._target_az_rad, self._target_el_rad)
        metrics_fbl = compute_tracking_metrics(self._results_fbl, self._target_az_rad, self._target_el_rad)
        metrics_ndob = compute_tracking_metrics(self._results_ndob, self._target_az_rad, self._target_el_rad)
        
        controllers = ['PID', 'FBL', 'FBL+NDOB']
        x = np.arange(len(controllers))
        width = 0.35
        
        # Settling Time
        settling_az = [metrics_pid['settling_time_az'], metrics_fbl['settling_time_az'], metrics_ndob['settling_time_az']]
        settling_el = [metrics_pid['settling_time_el'], metrics_fbl['settling_time_el'], metrics_ndob['settling_time_el']]
        
        ax1.bar(x - width/2, settling_az, width, label='Azimuth', color=AxisColors.AZIMUTH, alpha=0.7)
        ax1.bar(x + width/2, settling_el, width, label='Elevation', color=AxisColors.ELEVATION, alpha=0.7)
        ax1.set_ylabel('Settling Time [s]', fontsize=self.style.axis_label_fontsize, fontweight='bold')
        #ax1.set_title('Settling Time (2% Criterion)', fontsize=self.style.title_fontsize, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(controllers, fontsize=self.style.legend_fontsize)
        ax1.legend(loc='best', fontsize=self.style.legend_fontsize)
        ax1.grid(True, alpha=self.style.grid_alpha, axis='y', linestyle=self.style.grid_linestyle)
        
        # Steady-State Error
        sse_az = [np.rad2deg(abs(metrics_pid['ss_error_az'])),
                  np.rad2deg(abs(metrics_fbl['ss_error_az'])),
                  np.rad2deg(abs(metrics_ndob['ss_error_az']))]
        sse_el = [np.rad2deg(abs(metrics_pid['ss_error_el'])),
                  np.rad2deg(abs(metrics_fbl['ss_error_el'])),
                  np.rad2deg(abs(metrics_ndob['ss_error_el']))]
        
        ax2.bar(x - width/2, sse_az, width, label='Azimuth', color=AxisColors.AZIMUTH, alpha=0.7)
        ax2.bar(x + width/2, sse_el, width, label='Elevation', color=AxisColors.ELEVATION, alpha=0.7)
        ax2.set_ylabel('Steady-State Error [deg]', fontsize=self.style.axis_label_fontsize, fontweight='bold')
        #ax2.set_title('Steady-State Error', fontsize=self.style.title_fontsize, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(controllers, fontsize=self.style.legend_fontsize)
        ax2.legend(loc='best', fontsize=self.style.legend_fontsize)
        ax2.grid(True, alpha=self.style.grid_alpha, axis='y', linestyle=self.style.grid_linestyle)
        ax2.set_yscale('log')
        
        # RMS LOS Error
        los_rms = [np.rad2deg(self._results_pid['los_error_rms']),
                   np.rad2deg(self._results_fbl['los_error_rms']),
                   np.rad2deg(self._results_ndob['los_error_rms'])]
        ax3.bar(x, los_rms, color=[ControllerColors.PID, ControllerColors.FBL, ControllerColors.FBL_NDOB], alpha=0.7)
        ax3.set_ylabel('LOS RMS Error [deg]', fontsize=self.style.axis_label_fontsize, fontweight='bold')
        #ax3.set_title('RMS Line-of-Sight Error', fontsize=self.style.title_fontsize, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(controllers, fontsize=self.style.legend_fontsize)
        ax3.grid(True, alpha=self.style.grid_alpha, axis='y', linestyle=self.style.grid_linestyle)
        
        # Control Effort
        torque_rms = [
            np.sqrt(self._results_pid['torque_rms_az']**2 + self._results_pid['torque_rms_el']**2),
            np.sqrt(self._results_fbl['torque_rms_az']**2 + self._results_fbl['torque_rms_el']**2),
            np.sqrt(self._results_ndob['torque_rms_az']**2 + self._results_ndob['torque_rms_el']**2)
        ]
        ax4.bar(x, np.array(torque_rms)*1e3, color=[ControllerColors.PID, ControllerColors.FBL, ControllerColors.FBL_NDOB], alpha=0.7)
        ax4.set_ylabel('Total Torque RMS [mN·m]', fontsize=self.style.axis_label_fontsize, fontweight='bold')
        #ax4.set_title('Control Effort', fontsize=self.style.title_fontsize, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(controllers, fontsize=self.style.legend_fontsize)
        ax4.grid(True, alpha=self.style.grid_alpha, axis='y', linestyle=self.style.grid_linestyle)
        
       # fig.suptitle('Performance Metrics Summary', fontsize=self.style.suptitle_fontsize, fontweight='bold')
        
        return fig
    
    def _plot_ekf_performance(self) -> plt.Figure:
        """Figure 8: EKF State Estimation Performance."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.style.get_figure_size('2x2'),
                                                      constrained_layout=self._get_layout_mode())
        
        lw = self.style.linewidth_primary
        alpha = self.style.alpha_primary
        
        # Estimated vs Actual Azimuth
        ax1.plot(self._t_pid, np.rad2deg(self._results_pid['log_arrays']['est_az']),
                 color=ControllerColors.PID, linewidth=lw, label='PID Est', alpha=alpha)
        ax1.plot(self._t_fbl, np.rad2deg(self._results_fbl['log_arrays']['est_az']),
                 color=ControllerColors.FBL, linewidth=lw, label='FBL Est', alpha=alpha)
        ax1.plot(self._t_ndob, np.rad2deg(self._results_ndob['log_arrays']['est_az']),
                 color=ControllerColors.FBL_NDOB, linewidth=lw, label='FBL+NDOB Est', alpha=alpha)
        # Ground Truth
        ax1.plot(self._t_pid, np.rad2deg(self._results_pid['log_arrays']['q_az']),
                 color=ControllerColors.PID, linewidth=lw, linestyle='--', alpha=0.4, label='PID Truth')
        ax1.plot(self._t_fbl, np.rad2deg(self._results_fbl['log_arrays']['q_az']),
                 color=ControllerColors.FBL, linewidth=lw, linestyle='--', alpha=0.4, label='FBL Truth')
        ax1.plot(self._t_ndob, np.rad2deg(self._results_ndob['log_arrays']['q_az']),
                 color=ControllerColors.FBL_NDOB, linewidth=lw, linestyle='--', alpha=0.4, label='NDOB Truth')
        ax1.set_ylabel('Azimuth [deg]', fontsize=self.style.axis_label_fontsize, fontweight='bold')
        #ax1.set_title('Azimuth Position Estimate vs Truth', fontsize=self.style.title_fontsize, fontweight='bold')
        ax1.legend(loc='best', fontsize=self.style.legend_fontsize, ncol=2)
        ax1.grid(True, alpha=self.style.grid_alpha, linestyle=self.style.grid_linestyle)
        
        # Estimated vs Actual Elevation
        ax2.plot(self._t_pid, np.rad2deg(self._results_pid['log_arrays']['est_el']),
                 color=ControllerColors.PID, linewidth=lw, label='PID Est', alpha=alpha)
        ax2.plot(self._t_fbl, np.rad2deg(self._results_fbl['log_arrays']['est_el']),
                 color=ControllerColors.FBL, linewidth=lw, label='FBL Est', alpha=alpha)
        ax2.plot(self._t_ndob, np.rad2deg(self._results_ndob['log_arrays']['est_el']),
                 color=ControllerColors.FBL_NDOB, linewidth=lw, label='FBL+NDOB Est', alpha=alpha)
        ax2.plot(self._t_pid, np.rad2deg(self._results_pid['log_arrays']['q_el']),
                 color=ControllerColors.PID, linewidth=lw, linestyle='--', alpha=0.4)
        ax2.plot(self._t_fbl, np.rad2deg(self._results_fbl['log_arrays']['q_el']),
                 color=ControllerColors.FBL, linewidth=lw, linestyle='--', alpha=0.4)
        ax2.plot(self._t_ndob, np.rad2deg(self._results_ndob['log_arrays']['q_el']),
                 color=ControllerColors.FBL_NDOB, linewidth=lw, linestyle='--', alpha=0.4)
        ax2.set_ylabel('Elevation [deg]', fontsize=self.style.axis_label_fontsize, fontweight='bold')
        #ax2.set_title('Elevation Position Estimate vs Truth', fontsize=self.style.title_fontsize, fontweight='bold')
        ax2.legend(loc='best', fontsize=self.style.legend_fontsize)
        ax2.grid(True, alpha=self.style.grid_alpha, linestyle=self.style.grid_linestyle)
        
        # Estimated vs Actual Az Rate
        ax3.plot(self._t_pid, np.rad2deg(self._results_pid['log_arrays']['est_az_dot']),
                 color=ControllerColors.PID, linewidth=lw, label='PID Est', alpha=alpha)
        ax3.plot(self._t_fbl, np.rad2deg(self._results_fbl['log_arrays']['est_az_dot']),
                 color=ControllerColors.FBL, linewidth=lw, label='FBL Est', alpha=alpha)
        ax3.plot(self._t_ndob, np.rad2deg(self._results_ndob['log_arrays']['est_az_dot']),
                 color=ControllerColors.FBL_NDOB, linewidth=lw, label='FBL+NDOB Est', alpha=alpha)
        ax3.plot(self._t_pid, np.rad2deg(self._results_pid['log_arrays']['qd_az']),
                 color=ControllerColors.PID, linewidth=lw, linestyle='--', alpha=0.4)
        ax3.plot(self._t_fbl, np.rad2deg(self._results_fbl['log_arrays']['qd_az']),
                 color=ControllerColors.FBL, linewidth=lw, linestyle='--', alpha=0.4)
        ax3.plot(self._t_ndob, np.rad2deg(self._results_ndob['log_arrays']['qd_az']),
                 color=ControllerColors.FBL_NDOB, linewidth=lw, linestyle='--', alpha=0.4)
        ax3.set_ylabel('Az Rate [deg/s]', fontsize=self.style.axis_label_fontsize, fontweight='bold')
        ax3.set_xlabel('Time [s]', fontsize=self.style.axis_label_fontsize, fontweight='bold')
        #ax3.set_title('Azimuth Rate Estimate vs Truth', fontsize=self.style.title_fontsize, fontweight='bold')
        ax3.legend(loc='best', fontsize=self.style.legend_fontsize)
        ax3.grid(True, alpha=self.style.grid_alpha, linestyle=self.style.grid_linestyle)
        
        # Estimated vs Actual El Rate
        ax4.plot(self._t_pid, np.rad2deg(self._results_pid['log_arrays']['est_el_dot']),
                 color=ControllerColors.PID, linewidth=lw, label='PID Est', alpha=alpha)
        ax4.plot(self._t_fbl, np.rad2deg(self._results_fbl['log_arrays']['est_el_dot']),
                 color=ControllerColors.FBL, linewidth=lw, label='FBL Est', alpha=alpha)
        ax4.plot(self._t_ndob, np.rad2deg(self._results_ndob['log_arrays']['est_el_dot']),
                 color=ControllerColors.FBL_NDOB, linewidth=lw, label='FBL+NDOB Est', alpha=alpha)
        ax4.plot(self._t_pid, np.rad2deg(self._results_pid['log_arrays']['qd_el']),
                 color=ControllerColors.PID, linewidth=lw, linestyle='--', alpha=0.4)
        ax4.plot(self._t_fbl, np.rad2deg(self._results_fbl['log_arrays']['qd_el']),
                 color=ControllerColors.FBL, linewidth=lw, linestyle='--', alpha=0.4)
        ax4.plot(self._t_ndob, np.rad2deg(self._results_ndob['log_arrays']['qd_el']),
                 color=ControllerColors.FBL_NDOB, linewidth=lw, linestyle='--', alpha=0.4)
        ax4.set_ylabel('El Rate [deg/s]', fontsize=self.style.axis_label_fontsize, fontweight='bold')
        ax4.set_xlabel('Time [s]', fontsize=self.style.axis_label_fontsize, fontweight='bold')
        #ax4.set_title('Elevation Rate Estimate vs Truth', fontsize=self.style.title_fontsize, fontweight='bold')
        ax4.legend(loc='best', fontsize=self.style.legend_fontsize)
        ax4.grid(True, alpha=self.style.grid_alpha, linestyle=self.style.grid_linestyle)
        
       # fig.suptitle('EKF Performance (Estimate vs Ground Truth)',
       #              fontsize=self.style.suptitle_fontsize, fontweight='bold')
        
        return fig
    
    def _plot_fsm_performance(self) -> plt.Figure:
        """Figure 9: Fine Steering Mirror Performance.
        
        Three-panel layout:
          Panel 1: FSM Tip mechanical angle [mdeg] + twin-axis command voltage [V]
          Panel 2: FSM Tilt mechanical angle [mdeg] + twin-axis command voltage [V]
          Panel 3: (Fix 4) Post-FSM residual LOS error in µrad — the definitive
                   performance metric. Converging to zero proves correct operation;
                   persistent oscillations reveal closed-loop instability.
        """
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=self.style.get_figure_size('3x1'),
                                             sharex=True, constrained_layout=self._get_layout_mode())
        
        lw = self.style.linewidth_primary * 1.5
        alpha = self.style.alpha_primary
        
        # ── Panel 1: FSM Tip ─────────────────────────────────────────────────
        ax1.plot(self._t_pid, np.rad2deg(self._results_pid['log_arrays']['fsm_tip']) * 1000,
                 color=ControllerColors.PID, linewidth=lw, label='PID Tip [mdeg]', alpha=alpha)
        ax1.plot(self._t_fbl, np.rad2deg(self._results_fbl['log_arrays']['fsm_tip']) * 1000,
                 color=ControllerColors.FBL, linewidth=lw, label='FBL Tip [mdeg]', alpha=alpha)
        ax1.plot(self._t_ndob, np.rad2deg(self._results_ndob['log_arrays']['fsm_tip']) * 1000,
                 color=ControllerColors.FBL_NDOB, linewidth=lw, label='FBL+NDOB Tip [mdeg]', alpha=alpha)
        
        # Twin axis for command voltages (avoid rad2deg on Volts!)
       # ax1_cmd = ax1.twinx()
        #ax1_cmd.set_ylabel('Command [V]', fontsize=self.style.axis_label_fontsize, fontweight='bold', color='gray')
        #ax1_cmd.plot(self._t_pid, np.array(self._results_pid['log_arrays']['fsm_cmd_tip']),
              #   color=ControllerColors.PID, linewidth=2.0, linestyle='--', alpha=0.3)
        #ax1_cmd.plot(self._t_fbl, np.array(self._results_fbl['log_arrays']['fsm_cmd_tip']),
               #  color=ControllerColors.FBL, linewidth=2.0, linestyle='--', alpha=0.3)
       # ax1_cmd.plot(self._t_ndob, np.array(self._results_ndob['log_arrays']['fsm_cmd_tip']),
                # color=ControllerColors.FBL_NDOB, linewidth=2.0, linestyle='--', alpha=0.3)

        # FSM Stroke Limit lines
        limit_rad = np.array(self._results_ndob['log_arrays'].get('fsm_stroke_limit_rad', np.full(len(self._t_ndob), 0.010)))
        limit_mdeg = np.rad2deg(limit_rad) * 1000.0
        ax1.plot(self._t_ndob, limit_mdeg, color='red', linewidth=lw, linestyle='--', alpha=0.5, label='FSM Stroke / QPD Limit')
        ax1.plot(self._t_ndob, -limit_mdeg, color='red', linewidth=lw, linestyle='--', alpha=0.5)

        ax1.set_ylabel('Tip [mdeg]', fontsize=self.style.axis_label_fontsize, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=self.style.legend_fontsize, ncol=1)
        ax1.grid(True, alpha=self.style.grid_alpha, linestyle=self.style.grid_linestyle)
        
        # ── Panel 2: FSM Tilt ────────────────────────────────────────────────
        ax2.plot(self._t_pid, np.rad2deg(self._results_pid['log_arrays']['fsm_tilt']) * 1000,
                 color=ControllerColors.PID, linewidth=lw, label='PID Tilt [mdeg]', alpha=alpha)
        ax2.plot(self._t_fbl, np.rad2deg(self._results_fbl['log_arrays']['fsm_tilt']) * 1000,
                 color=ControllerColors.FBL, linewidth=lw, label='FBL Tilt [mdeg]', alpha=alpha)
        ax2.plot(self._t_ndob, np.rad2deg(self._results_ndob['log_arrays']['fsm_tilt']) * 1000,
                 color=ControllerColors.FBL_NDOB, linewidth=lw, label='FBL+NDOB Tilt [mdeg]', alpha=alpha)
                 
       # ax2_cmd = ax2.twinx()
       # ax2_cmd.set_ylabel('Command [V]', fontsize=self.style.axis_label_fontsize, fontweight='bold', color='gray')
       # ax2_cmd.plot(self._t_pid, np.array(self._results_pid['log_arrays']['fsm_cmd_tilt']),
             #    color=ControllerColors.PID, linewidth=2.0, linestyle='--', alpha=0.3, label='PID Cmd [V]')
        #ax2_cmd.plot(self._t_fbl, np.array(self._results_fbl['log_arrays']['fsm_cmd_tilt']),
        #         color=ControllerColors.FBL, linewidth=1, linestyle='--', alpha=0.3, label='FBL Cmd [V]')
        #ax2_cmd.plot(self._t_ndob, np.array(self._results_ndob['log_arrays']['fsm_cmd_tilt']),
              #   color=ControllerColors.FBL_NDOB, linewidth=1, linestyle='--', alpha=0.3, label='NDOB Cmd [V]')

        # FSM Stroke Limit lines 
        limit_rad = np.array(self._results_ndob['log_arrays'].get('fsm_stroke_limit_rad', np.full(len(self._t_ndob), 0.010)))
        limit_mdeg = np.rad2deg(limit_rad) * 1000.0
        ax2.plot(self._t_ndob, limit_mdeg, color='red', linewidth=lw, linestyle='--', alpha=0.5, label='FSM Stroke / QPD Limit')
        ax2.plot(self._t_ndob, -limit_mdeg, color='red', linewidth=lw, linestyle='--', alpha=0.5)

        ax2.set_ylabel('Tilt [mdeg]', fontsize=self.style.axis_label_fontsize, fontweight='bold')
        ax2.legend(loc='upper left', fontsize=self.style.legend_fontsize)
       # ax2_cmd.legend(loc='upper right', fontsize=self.style.legend_fontsize)
        ax2.grid(True, alpha=self.style.grid_alpha, linestyle=self.style.grid_linestyle)
        
        # ── Panel 3 (Fix 4): Post-FSM Residual LOS Error ─────────────────────
        # This is the DEFINITIVE performance metric: what the QPD actually senses.
        # It is the LOS error AFTER the FSM has applied its correction (2×θ_fsm).
        # A converging residual ≈ 0 proves the FSM is working.
        # Persistent oscillations here indicate true closed-loop instability.
        _to_urad = 1e6  # rad → µrad conversion
        
        # Safe accessor — key present only in simulations with Fix 3 applied
        def _get_residual(results, key):
            arr = results['log_arrays'].get(key, None)
            if arr is None:
                # Fall back: reconstruct from los_error and fsm angle
                los  = np.array(results['log_arrays'].get('los_error_x' if 'x' in key else 'los_error_y', [0]))
                fsm  = np.array(results['log_arrays'].get('fsm_tip' if 'x' in key else 'fsm_tilt', [0]))
                return los - 2.0 * fsm
            return np.array(arr)

        res_pid_x  = _get_residual(self._results_pid,  'fsm_residual_error_x') * _to_urad
        res_fbl_x  = _get_residual(self._results_fbl,  'fsm_residual_error_x') * _to_urad
        res_ndob_x = _get_residual(self._results_ndob, 'fsm_residual_error_x') * _to_urad
        res_pid_y  = _get_residual(self._results_pid,  'fsm_residual_error_y') * _to_urad
        res_fbl_y  = _get_residual(self._results_fbl,  'fsm_residual_error_y') * _to_urad
        res_ndob_y = _get_residual(self._results_ndob, 'fsm_residual_error_y') * _to_urad
        
        ax3.plot(self._t_pid,  res_pid_x,  color=ControllerColors.PID,      linewidth=lw,  label='PID X',      alpha=alpha)
        ax3.plot(self._t_fbl,  res_fbl_x,  color=ControllerColors.FBL,      linewidth=lw,  label='FBL X',      alpha=alpha)
        ax3.plot(self._t_ndob, res_ndob_x, color=ControllerColors.FBL_NDOB, linewidth=lw,  label='NDOB X',     alpha=alpha)
        ax3.plot(self._t_pid,  res_pid_y,  color=ControllerColors.PID,      linewidth=lw,  label='PID Y',      alpha=alpha, linestyle='--')
        ax3.plot(self._t_fbl,  res_fbl_y,  color=ControllerColors.FBL,      linewidth=lw,  label='FBL Y',      alpha=alpha, linestyle='--')
        ax3.plot(self._t_ndob, res_ndob_y, color=ControllerColors.FBL_NDOB, linewidth=lw,  label='NDOB Y',     alpha=alpha, linestyle='--')
        
        # ±1 µrad target band (sub-µrad FSO link budget requirement)
        ax3.axhspan(-1.0, 1.0, alpha=0.12, color='green', label='±1 µrad target band')
        ax3.axhline(0, color='black', linewidth=lw, linestyle='--', alpha=0.5)
        ax3.set_ylabel('Residual LOS Error [µrad]', fontsize=self.style.axis_label_fontsize, fontweight='bold')
        ax3.set_xlabel('Time [s]', fontsize=self.style.axis_label_fontsize, fontweight='bold')
        ax3.legend(loc='upper right', fontsize=self.style.legend_fontsize, ncol=3)
        ax3.grid(True, alpha=self.style.grid_alpha, linestyle=self.style.grid_linestyle)
        
        # RMS residual for the NDOB case (best performer)
        t_ndob = np.array(self._t_ndob)
        if len(t_ndob) > 10:
            steady_start = t_ndob[-1] * 0.5          # Last 50% of simulation
            mask = t_ndob >= steady_start
            if mask.sum() > 0:
                rms_x = float(np.sqrt(np.mean(res_ndob_x[mask]**2)))
                rms_y = float(np.sqrt(np.mean(res_ndob_y[mask]**2)))
                ax3.set_title(f'Post-FSM Residual (FBL+NDOB steady-state RMS: X={rms_x:.2f} µrad, Y={rms_y:.2f} µrad)',
                              fontsize=self.style.title_fontsize - 1, fontweight='bold')
        
        return fig

    
    def _plot_internal_signals(self) -> plt.Figure:
        """Figure 10: Internal Control Signal & Disturbance Observer."""
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=self.style.get_figure_size('3x1_tall'),
                                             sharex=True, constrained_layout=self._get_layout_mode())
        
        lw = self.style.linewidth_primary
        alpha = self.style.alpha_primary
        
        # Virtual control signals
        v_az_fbl = self._results_fbl['log_arrays'].get('v_virtual_az', np.zeros_like(self._t_fbl))
        v_el_fbl = self._results_fbl['log_arrays'].get('v_virtual_el', np.zeros_like(self._t_fbl))
        v_az_ndob = self._results_ndob['log_arrays'].get('v_virtual_az', np.zeros_like(self._t_ndob))
        v_el_ndob = self._results_ndob['log_arrays'].get('v_virtual_el', np.zeros_like(self._t_ndob))
        
        # Torques
        tau_az_fbl = self._results_fbl['log_arrays'].get('tau_unsaturated_az', self._results_fbl['log_arrays']['torque_az'])
        tau_el_fbl = self._results_fbl['log_arrays'].get('tau_unsaturated_el', self._results_fbl['log_arrays']['torque_el'])
        tau_az_ndob = self._results_ndob['log_arrays'].get('tau_unsaturated_az', self._results_ndob['log_arrays']['torque_az'])
        tau_el_ndob = self._results_ndob['log_arrays'].get('tau_unsaturated_el', self._results_ndob['log_arrays']['torque_el'])
        
        # NDOB estimates
        d_hat_az_ndob = self._results_ndob['log_arrays'].get('d_hat_ndob_az', np.zeros_like(self._t_ndob))
        d_hat_el_ndob = self._results_ndob['log_arrays'].get('d_hat_ndob_el', np.zeros_like(self._t_ndob))
        
        # Ground truth friction
        friction_coef = 0.1
        d_true_az_ndob = friction_coef * self._results_ndob['log_arrays']['qd_az']
        d_true_el_ndob = friction_coef * self._results_ndob['log_arrays']['qd_el']
        
        # Subplot 1: Virtual Control
        ax1.plot(self._t_fbl, v_az_fbl, color=ControllerColors.FBL, linewidth=lw, label='FBL Az', alpha=alpha)
        ax1.plot(self._t_ndob, v_az_ndob, color=ControllerColors.FBL_NDOB, linewidth=lw, label='FBL+NDOB Az', alpha=alpha, linestyle='--')
        ax1.plot(self._t_fbl, v_el_fbl, color=ControllerColors.FBL, linewidth=lw, label='FBL El', alpha=0.7, linestyle=':')
        ax1.plot(self._t_ndob, v_el_ndob, color=ControllerColors.FBL_NDOB, linewidth=lw, label='FBL+NDOB El', alpha=0.7, linestyle='-.')
        ax1.axhline(0, color='black', linewidth=lw, linestyle='--', alpha=0.5)
        ax1.set_ylabel(r'Virtual Control $v$ [rad/s²]', fontsize=self.style.axis_label_fontsize, fontweight='bold')
        #ax1.set_title(r'Virtual Control Input (Outer Loop $v = \ddot{q}_{ref} + K_p e + K_d \dot{e}$)',
        #              fontsize=self.style.title_fontsize, fontweight='bold')
        ax1.legend(loc='best', fontsize=self.style.legend_fontsize, ncol=2)
        ax1.grid(True, alpha=self.style.grid_alpha, linestyle=self.style.grid_linestyle)
        
        # Subplot 2: Commanded Torques
        ax2.plot(self._t_fbl, tau_az_fbl, color=ControllerColors.FBL, linewidth=lw, label='FBL Az', alpha=alpha)
        ax2.plot(self._t_ndob, tau_az_ndob, color=ControllerColors.FBL_NDOB, linewidth=lw, label='FBL+NDOB Az', alpha=alpha, linestyle='--')
        ax2.plot(self._t_fbl, tau_el_fbl, color=ControllerColors.FBL, linewidth=lw, label='FBL El', alpha=0.7, linestyle=':')
        ax2.plot(self._t_ndob, tau_el_ndob, color=ControllerColors.FBL_NDOB, linewidth=lw, label='FBL+NDOB El', alpha=0.7, linestyle='-.')
        ax2.axhline(0, color='black', linewidth=lw, linestyle='--', alpha=0.5)
        ax2.axhline(1.0, color='red', linewidth=lw, linestyle=':', alpha=0.6, label='Saturation')
        ax2.axhline(-1.0, color='red', linewidth=lw, linestyle=':', alpha=0.6)
        ax2.set_ylabel(r'Torque $\tau$ [N·m]', fontsize=self.style.axis_label_fontsize, fontweight='bold')
        #ax2.set_title(r'Commanded Motor Torque ($\tau = Mv + C\dot{q} + G - \hat{d}$)',
        #              fontsize=self.style.title_fontsize, fontweight='bold')
        ax2.legend(loc='best', fontsize=self.style.legend_fontsize, ncol=2)
        ax2.grid(True, alpha=self.style.grid_alpha, linestyle=self.style.grid_linestyle)
        
        # Subplot 3: Disturbance Estimates
        ax3.plot(self._t_ndob, d_hat_az_ndob, color=ControllerColors.FBL_NDOB, linewidth=lw, label='NDOB Est Az', alpha=alpha)
        ax3.plot(self._t_ndob, d_hat_el_ndob, color='purple', linewidth=lw, label='NDOB Est El', alpha=alpha, linestyle='--')
        ax3.plot(self._t_ndob, d_true_az_ndob, color='gray', linewidth=lw, linestyle=':',
                 label='Ground Truth Az', alpha=0.7)
        ax3.plot(self._t_ndob, d_true_el_ndob, color='gray', linewidth=lw, linestyle='-.',
                 label='Ground Truth El', alpha=0.7)
        ax3.axhline(0, color='black', linewidth=lw, linestyle='--', alpha=0.5, label='Zero Disturbance')
        ax3.set_ylabel(r'Disturbance $\hat{d}$ [N·m]', fontsize=self.style.axis_label_fontsize, fontweight='bold')
        #ax3.set_title(r'NDOB Disturbance Estimate ($\hat{d} = z + L M(q) \dot{q}$)',
        #              fontsize=self.style.title_fontsize, fontweight='bold')
        ax3.legend(loc='best', fontsize=self.style.legend_fontsize, ncol=2)
        ax3.grid(True, alpha=self.style.grid_alpha, linestyle=self.style.grid_linestyle)
        
        # Subplot 4 (Fix 5): FSM PID P-term vs I-state — limit-cycle diagnostic
        # Diagnostic rule:
        #   • I-term pegged at ±50V while P-term alternates rapidly → integrator limit cycle
        #   • Both terms bounded and decreasing → healthy convergence
        for results, t_arr, label, color in [
            (self._results_pid,  self._t_pid,  'PID',      ControllerColors.PID),
            (self._results_ndob, self._t_ndob, 'FBL+NDOB', ControllerColors.FBL_NDOB),
        ]:
            log = results['log_arrays']
            n   = len(t_arr)
            p_tip = np.array(log.get('fsm_pid_p_tip', np.zeros(n)))
            i_tip = np.array(log.get('fsm_pid_i_tip', np.zeros(n)))
            ax4.plot(t_arr, p_tip, color=color, linewidth=lw, linestyle='-',  alpha=alpha,  label=f'{label} P-term [V]')
            ax4.plot(t_arr, i_tip, color=color, linewidth=lw, linestyle='--', alpha=alpha,  label=f'{label} I-state [V]')
        ax4.axhline( 50.0, color='red', linewidth=lw, linestyle=':', alpha=0.6, label='+50 V rail')
        ax4.axhline(-50.0, color='red', linewidth=lw, linestyle=':', alpha=0.6, label='-50 V rail')
        ax4.axhline(0, color='black', linewidth=lw, linestyle='--', alpha=0.4)
        ax4.set_ylabel('FSM PID Terms [V]\n(Tip axis)', fontsize=self.style.axis_label_fontsize, fontweight='bold')
        ax4.set_xlabel('Time [s]', fontsize=self.style.axis_label_fontsize, fontweight='bold')
        ax4.legend(loc='best', fontsize=self.style.legend_fontsize, ncol=2)
        ax4.grid(True, alpha=self.style.grid_alpha, linestyle=self.style.grid_linestyle)
        
        return fig
    
    def _plot_ekf_diagnostics(self) -> plt.Figure:
        """Figure 11: EKF Diagnostics (Innovation & Covariance)."""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=self.style.get_figure_size('3x1_tall'))
        
        log = self._results_ndob['log_arrays']
        t = log['time']
        
        # Subplot 1: State Covariance
        if 'ekf_cov_theta_az' in log:
            ax1.semilogy(t, log['ekf_cov_theta_az'], 'b-', linewidth=2, label=r'$P_{\theta_{Az}}$')
            ax1.semilogy(t, log['ekf_cov_theta_dot_az'], 'r-', linewidth=2, label=r'$P_{\dot{\theta}_{Az}}$')
            ax1.semilogy(t, log['ekf_cov_bias_az'], 'g-', linewidth=2, label=r'$P_{b_{Az}}$')
            ax1.set_ylabel('Covariance Diagonal', fontsize=self.style.axis_label_fontsize, fontweight='bold')
            #ax1.set_title('(a) EKF State Covariance Evolution (Log Scale)',
            #              fontsize=self.style.title_fontsize, fontweight='bold')
            ax1.legend(loc='best', fontsize=self.style.legend_fontsize, ncol=3)
        else:
            ax1.text(0.5, 0.5, 'EKF Covariance Logging Not Yet Implemented\n(Requires simulation_runner enhancement)',
                     ha='center', va='center', transform=ax1.transAxes, fontsize=self.style.axis_label_fontsize)
            ax1.set_ylabel('Covariance', fontsize=self.style.axis_label_fontsize)
            #ax1.set_title('(a) EKF State Covariance Evolution', fontsize=self.style.title_fontsize, fontweight='bold')
        ax1.grid(True, alpha=self.style.grid_alpha, linestyle=self.style.grid_linestyle)
        
        # Subplot 2: Innovation Residuals
        if 'ekf_innovation_enc_az' in log:
            ax2.plot(t, np.rad2deg(log['ekf_innovation_enc_az']), 'b-', linewidth=2, label='Encoder Az Innovation')
            ax2.plot(t, np.rad2deg(log['ekf_innovation_enc_el']), 'r-', linewidth=2, label='Encoder El Innovation')
            ax2.axhline(0, color='k', linestyle='--', linewidth=2, alpha=0.5)
            ax2.set_ylabel('Innovation [deg]', fontsize=self.style.axis_label_fontsize, fontweight='bold')
            #ax2.set_title('(b) Measurement Innovation (Encoder Residuals)',
            #              fontsize=self.style.title_fontsize, fontweight='bold')
            ax2.legend(loc='best', fontsize=self.style.legend_fontsize)
        else:
            ax2.text(0.5, 0.5, 'Innovation Logging Not Yet Implemented',
                     ha='center', va='center', transform=ax2.transAxes, fontsize=self.style.axis_label_fontsize)
            ax2.set_ylabel('Innovation [deg]', fontsize=self.style.axis_label_fontsize)
           # ax2.set_title('(b) Measurement Innovation', fontsize=self.style.title_fontsize, fontweight='bold')
        ax2.grid(True, alpha=self.style.grid_alpha, linestyle=self.style.grid_linestyle)
        
        # Subplot 3: 3-Sigma Bounds
        if 'ekf_innovation_3sigma_az' in log:
            ax3.plot(t, np.rad2deg(log['ekf_innovation_enc_az']), 'b-', linewidth=1.2, label='Innovation Az')
            ax3.plot(t, np.rad2deg(log['ekf_innovation_3sigma_az']), 'r--', linewidth=1.5, label=r'$+3\sigma$ Bound')
            ax3.plot(t, -np.rad2deg(log['ekf_innovation_3sigma_az']), 'r--', linewidth=1.5, label=r'$-3\sigma$ Bound')
            violations = np.where(np.abs(log['ekf_innovation_enc_az']) > log['ekf_innovation_3sigma_az'])[0]
            if len(violations) > 0:
                ax3.scatter(t[violations], np.rad2deg(log['ekf_innovation_enc_az'][violations]),
                           color='red', marker='x', s=100, linewidths=2, label='3σ Violation', zorder=10)
            ax3.set_xlabel('Time [s]', fontsize=self.style.axis_label_fontsize, fontweight='bold')
            ax3.set_ylabel('Innovation [deg]', fontsize=self.style.axis_label_fontsize, fontweight='bold')
            #ax3.set_title('(c) Innovation Bounds & Consistency Check',
                          #fontsize=self.style.title_fontsize, fontweight='bold')
            ax3.legend(loc='best', fontsize=self.style.legend_fontsize)
        else:
            ax3.text(0.5, 0.5, '3-Sigma Bounds Logging Not Yet Implemented',
                     ha='center', va='center', transform=ax3.transAxes, fontsize=self.style.axis_label_fontsize)
            ax3.set_xlabel('Time [s]', fontsize=self.style.axis_label_fontsize)
            ax3.set_ylabel('Innovation [deg]', fontsize=self.style.axis_label_fontsize)
           # ax3.set_title('(c) Innovation Consistency Check', fontsize=self.style.title_fontsize, fontweight='bold')
        ax3.grid(True, alpha=self.style.grid_alpha, linestyle=self.style.grid_linestyle)
        
        #fig.suptitle('Figure 11: Extended Kalman Filter Diagnostics & Adaptive Tuning',
                    # fontsize=self.style.suptitle_fontsize, fontweight='bold')
        
        return fig
    
    def _plot_disturbance_torques(self) -> plt.Figure:
        """Figure 12: Environmental Disturbance Torques."""
        fig, axes = plt.subplots(2, 2, figsize=self.style.get_figure_size('double_column'),
                                  constrained_layout=self._get_layout_mode())
        ax1, ax2 = axes[0]
        ax3, ax4 = axes[1]
        
        lw = self.style.linewidth_primary
        
        log = self._results_ndob['log_arrays']
        has_disturbance = 'tau_disturbance_az' in log and np.std(log['tau_disturbance_az']) > 1e-10
        
        if has_disturbance:
            t = log['time']
            tau_d_az = np.array(log['tau_disturbance_az']) * 1000
            tau_d_el = np.array(log['tau_disturbance_el']) * 1000
            wind_az = np.array(log['wind_torque_az']) * 1000
            wind_el = np.array(log['wind_torque_el']) * 1000
            vib_az = np.array(log['vibration_torque_az']) * 1000
            vib_el = np.array(log['vibration_torque_el']) * 1000
            
            # Total Disturbance - Azimuth
            ax1.plot(t, tau_d_az, color=AxisColors.AZIMUTH, linewidth=lw, alpha=0.9)
            ax1.fill_between(t, tau_d_az, alpha=0.2, color=AxisColors.AZIMUTH)
            ax1.axhline(0, color='black', linewidth=lw, linestyle='--', alpha=0.5)
            mean_az = np.mean(tau_d_az)
            std_az = np.std(tau_d_az)
            ax1.axhline(mean_az, color='gray', linewidth=lw, linestyle=':', alpha=0.8, label=f'Mean = {mean_az:.2f} mN·m')
            ax1.axhline(mean_az + 2*std_az, color='orange', linewidth=lw, linestyle='--', alpha=0.6, label=r'$\pm 2\sigma$')
            ax1.axhline(mean_az - 2*std_az, color='orange', linewidth=lw, linestyle='--', alpha=0.6)
            ax1.set_ylabel(r'Torque [mN·m]', fontsize=self.style.axis_label_fontsize, fontweight='bold')
            ax1.set_xlabel('Time (s)', fontsize=self.style.axis_label_fontsize, fontweight='bold')
            #ax1.set_title('(a) Total Environmental Disturbance — Azimuth Axis',
                          #fontsize=self.style.title_fontsize, fontweight='bold')
            ax1.legend(loc='upper right', fontsize=self.style.legend_fontsize, framealpha=0.95)
            ax1.grid(True, alpha=self.style.grid_alpha, linestyle=self.style.grid_linestyle)
            ax1.set_xlim([t[0], t[-1]])
            
            # Total Disturbance - Elevation
            ax2.plot(t, tau_d_el, color=AxisColors.ELEVATION, linewidth=lw, alpha=0.9)
            ax2.fill_between(t, tau_d_el, alpha=0.2, color=AxisColors.ELEVATION)
            ax2.axhline(0, color='black', linewidth=lw, linestyle='--', alpha=0.5)
            mean_el = np.mean(tau_d_el)
            std_el = np.std(tau_d_el)
            ax2.axhline(mean_el, color='gray', linewidth=lw, linestyle=':', alpha=0.8, label=f'Mean = {mean_el:.2f} mN·m')
            ax2.axhline(mean_el + 2*std_el, color='orange', linewidth=lw, linestyle='--', alpha=0.6, label=r'$\pm 2\sigma$')
            ax2.axhline(mean_el - 2*std_el, color='orange', linewidth=lw, linestyle='--', alpha=0.6)
            ax2.set_ylabel(r'Torque [mN·m]', fontsize=self.style.axis_label_fontsize, fontweight='bold')
            ax2.set_xlabel('Time (s)', fontsize=self.style.axis_label_fontsize, fontweight='bold')
            #ax2.set_title('(b) Total Environmental Disturbance — Elevation Axis',
            #              fontsize=self.style.title_fontsize, fontweight='bold')
            ax2.legend(loc='upper right', fontsize=self.style.legend_fontsize, framealpha=0.95)
            ax2.grid(True, alpha=self.style.grid_alpha, linestyle=self.style.grid_linestyle)
            ax2.set_xlim([t[0], t[-1]])
            
            # Components - Azimuth
            ax3.plot(t, wind_az, color=DisturbanceColors.WIND, linewidth=lw, label='Wind/Gust (Dryden)', alpha=0.9)
            ax3.plot(t, vib_az, color=DisturbanceColors.VIBRATION, linewidth=lw, label='Structural Vibration', alpha=0.85)
            ax3.axhline(0, color='black', linewidth=lw, linestyle='--', alpha=0.5)
            ax3.set_ylabel(r'Torque [mN·m]', fontsize=self.style.axis_label_fontsize, fontweight='bold')
            ax3.set_xlabel('Time (s)', fontsize=self.style.axis_label_fontsize, fontweight='bold')
            #ax3.set_title('(c) Disturbance Components — Azimuth Axis',
            #              fontsize=self.style.title_fontsize, fontweight='bold')
            ax3.legend(loc='best', fontsize=self.style.legend_fontsize, framealpha=0.95)
            ax3.grid(True, alpha=self.style.grid_alpha, linestyle=self.style.grid_linestyle)
            ax3.set_xlim([t[0], t[-1]])
            
            # Components - Elevation
            ax4.plot(t, wind_el, color=DisturbanceColors.WIND, linewidth=lw, label='Wind/Gust (Dryden)', alpha=0.9)
            ax4.plot(t, vib_el, color=DisturbanceColors.VIBRATION, linewidth=lw, label='Structural Vibration', alpha=0.85)
            ax4.axhline(0, color='black', linewidth=lw, linestyle='--', alpha=0.5)
            ax4.set_ylabel(r'Torque [mN·m]', fontsize=self.style.axis_label_fontsize, fontweight='bold')
            ax4.set_xlabel('Time (s)', fontsize=self.style.axis_label_fontsize, fontweight='bold')
           # ax4.set_title('(d) Disturbance Components — Elevation Axis',
           #               fontsize=self.style.title_fontsize, fontweight='bold')
            ax4.legend(loc='best', fontsize=self.style.legend_fontsize, framealpha=0.95)
            ax4.grid(True, alpha=self.style.grid_alpha, linestyle=self.style.grid_linestyle)
            ax4.set_xlim([t[0], t[-1]])
        else:
            # No disturbance data - placeholders
            for ax, title in zip([ax1, ax2, ax3, ax4],
                                  ['(a) Total Disturbance — Azimuth',
                                   '(b) Total Disturbance — Elevation',
                                   '(c) Wind & Vibration — Azimuth',
                                   '(d) Wind & Vibration — Elevation']):
                ax.text(0.5, 0.5,
                        'Environmental Disturbances Disabled\n\n'
                        'Enable with:\n'
                        '  environmental_disturbance_enabled=True\n'
                        '  environmental_disturbance_config={...}',
                        ha='center', va='center', transform=ax.transAxes, fontsize=self.style.axis_label_fontsize,
                        fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
                ax.set_xlabel('Time (s)', fontsize=self.style.axis_label_fontsize, fontweight='bold')
                ax.set_ylabel('Torque [mN·m]', fontsize=self.style.axis_label_fontsize)
              #  ax.set_title(title, fontsize=self.style.title_fontsize, fontweight='bold')
        
        #fig.suptitle(r'(Dryden Wind Turbulence + PSD-Based Structural Vibration — Plant Injection Only)',
                    # fontsize=self.style.suptitle_fontsize, fontweight='bold')
        
        return fig
    
    def _plot_disturbance_statistics(self) -> plt.Figure:
        """Figure 13: Disturbance Statistics & PSD Analysis."""
        log = self._results_ndob['log_arrays']
        has_disturbance = 'tau_disturbance_az' in log and np.std(log['tau_disturbance_az']) > 1e-10
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), constrained_layout=self._get_layout_mode())
        
        if has_disturbance:
            t = log['time']
            tau_d_az = np.array(log['tau_disturbance_az']) * 1000
            tau_d_el = np.array(log['tau_disturbance_el']) * 1000
            mean_az = np.mean(tau_d_az)
            std_az = np.std(tau_d_az)
            mean_el = np.mean(tau_d_el)
            std_el = np.std(tau_d_el)
            
            # Statistical Distribution
            bins = 40
            ax1.hist(tau_d_az, bins=bins, alpha=0.6, color=AxisColors.AZIMUTH, label='Azimuth', density=True, edgecolor='white')
            ax1.hist(tau_d_el, bins=bins, alpha=0.6, color=AxisColors.ELEVATION, label='Elevation', density=True, edgecolor='white')
            
            # Gaussian fit
            from scipy.stats import norm
            x_range_az = np.linspace(tau_d_az.min(), tau_d_az.max(), 100)
            x_range_el = np.linspace(tau_d_el.min(), tau_d_el.max(), 100)
            ax1.plot(x_range_az, norm.pdf(x_range_az, mean_az, std_az),
                     color=AxisColors.AZIMUTH, linewidth=2, linestyle='--', label=f'Az: N({mean_az:.1f}, {std_az:.1f}²)')
            ax1.plot(x_range_el, norm.pdf(x_range_el, mean_el, std_el),
                     color=AxisColors.ELEVATION, linewidth=2, linestyle='--', label=f'El: N({mean_el:.1f}, {std_el:.1f}²)')
            
            ax1.set_xlabel(r'Disturbance Torque $\tau_d$ [mN·m]', fontsize=self.style.axis_label_fontsize, fontweight='bold')
            ax1.set_ylabel('Probability Density', fontsize=self.style.axis_label_fontsize, fontweight='bold')
     #       ax1.set_title('(a) Disturbance Torque Statistical Distribution & Gaussian Fit',
                 #         fontsize=self.style.title_fontsize, fontweight='bold')
            ax1.legend(loc='upper right', fontsize=self.style.legend_fontsize, framealpha=0.95)
            ax1.grid(True, alpha=self.style.grid_alpha, linestyle=self.style.grid_linestyle)
            
            # Statistics text box
            stats_text = (
                f"Azimuth:\n"
                f"  Mean: {mean_az:.2f} mN·m\n"
                f"  Std:  {std_az:.2f} mN·m\n"
                f"  Peak: {np.max(np.abs(tau_d_az)):.2f} mN·m\n\n"
                f"Elevation:\n"
                f"  Mean: {mean_el:.2f} mN·m\n"
                f"  Std:  {std_el:.2f} mN·m\n"
                f"  Peak: {np.max(np.abs(tau_d_el)):.2f} mN·m"
            )
            ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=self.style.axis_label_fontsize,
                     verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='gray'))
            
            # PSD
            from scipy import signal as sig
            dt = t[1] - t[0] if len(t) > 1 else 0.001
            fs = 1.0 / dt
            nperseg = min(512, len(tau_d_az) // 4)
            
            if nperseg > 8:
                f_az, psd_az = sig.welch(tau_d_az, fs=fs, nperseg=nperseg)
                f_el, psd_el = sig.welch(tau_d_el, fs=fs, nperseg=nperseg)
                
                ax2.semilogy(f_az, psd_az, color=AxisColors.AZIMUTH, linewidth=2, label='Azimuth', alpha=0.9)
                ax2.semilogy(f_el, psd_el, color=AxisColors.ELEVATION, linewidth=2, label='Elevation', alpha=0.9)
                
                ax2.axvspan(0, 5, alpha=0.1, color='green', label='Low-freq (wind)')
                ax2.axvspan(10, 100, alpha=0.1, color='blue', label='Structural modes')
                
                ax2.set_xlabel('Frequency [Hz]', fontsize=self.style.axis_label_fontsize, fontweight='bold')
                ax2.set_ylabel(r'PSD [(mN·m)²/Hz]', fontsize=self.style.axis_label_fontsize, fontweight='bold')
               # ax2.set_title('(b) Power Spectral Density of Disturbance Torques',
                             # fontsize=self.style.title_fontsize, fontweight='bold')
                ax2.legend(loc='upper right', fontsize=self.style.legend_fontsize, framealpha=0.95)
                ax2.grid(True, alpha=self.style.grid_alpha, linestyle=self.style.grid_linestyle, which='both')
                ax2.set_xlim([0, min(fs/2, 200)])
            else:
                ax2.text(0.5, 0.5, 'Insufficient data for PSD\n(requires longer simulation)',
                         ha='center', va='center', transform=ax2.transAxes, fontsize=self.style.axis_label_fontsize)
                ax2.set_xlabel('Frequency [Hz]', fontsize=self.style.axis_label_fontsize)
                ax2.set_ylabel('PSD', fontsize=self.style.axis_label_fontsize)
               # ax2.set_title('(b) Power Spectral Density', fontsize=self.style.title_fontsize, fontweight='bold')
            
           # fig.suptitle('Figure 13: Environmental Disturbance Torque Statistics & Frequency Analysis',
                       # fontsize=self.style.suptitle_fontsize, fontweight='bold')
        else:
            ax1.text(0.5, 0.5, 'No disturbance data available', ha='center', va='center',
                     transform=ax1.transAxes, fontsize=self.style.axis_label_fontsize)
            # ax1.set_title('Figure 13a: Disturbance Statistics (No Data)',
            #               fontsize=self.style.title_fontsize, fontweight='bold')
            ax2.text(0.5, 0.5, 'No disturbance data available', ha='center', va='center',
                     transform=ax2.transAxes, fontsize=self.style.axis_label_fontsize)
            # ax2.set_title('Figure 13b: PSD Analysis (No Data)',
            #               fontsize=self.style.title_fontsize, fontweight='bold')
        
        return fig

    # ─────────────────────────────────────────────────────────────────────────
    def _plot_stroke_consumption(self) -> plt.Figure:
        """Figure 14: Instantaneous Stroke Utilization & Stroke Load.

        Two-panel layout:
          Panel 1: Instantaneous Stroke Utilization (%) - Azimuth (Tip).
          Panel 2: Instantaneous Stroke Utilization (%) - Elevation (Tilt).
        """
        stroke_lim_arr = self._results_ndob['log_arrays'].get('fsm_stroke_limit_rad', None)
        theta_max = float(stroke_lim_arr[0]) if stroke_lim_arr is not None else 0.010

        calculator = StrokeMetrics(theta_max=theta_max, jitter_cutoff_hz=50.0, filter_order=4)

        def _infer_dt(log_arrays):
            t = np.asarray(log_arrays['time'])
            return float(np.median(np.diff(t))) if len(t) > 1 else 1e-4

        def _get_link_active(log_arrays):
            """Extract is_beam_on_sensor as a numpy bool array, or None."""
            raw = log_arrays.get('is_beam_on_sensor', None)
            if raw is not None:
                return np.asarray(raw, dtype=bool)
            return None

        try:
            m_pid = calculator.compute(
                time=np.asarray(self._results_pid['log_arrays']['time']),
                fsm_tip=np.asarray(self._results_pid['log_arrays']['fsm_tip']),
                fsm_tilt=np.asarray(self._results_pid['log_arrays']['fsm_tilt']),
                dt=_infer_dt(self._results_pid['log_arrays']),
                link_active=_get_link_active(self._results_pid['log_arrays']),
            )
            m_fbl = calculator.compute(
                time=np.asarray(self._results_fbl['log_arrays']['time']),
                fsm_tip=np.asarray(self._results_fbl['log_arrays']['fsm_tip']),
                fsm_tilt=np.asarray(self._results_fbl['log_arrays']['fsm_tilt']),
                dt=_infer_dt(self._results_fbl['log_arrays']),
                link_active=_get_link_active(self._results_fbl['log_arrays']),
            )
            m_ndob = calculator.compute(
                time=np.asarray(self._results_ndob['log_arrays']['time']),
                fsm_tip=np.asarray(self._results_ndob['log_arrays']['fsm_tip']),
                fsm_tilt=np.asarray(self._results_ndob['log_arrays']['fsm_tilt']),
                dt=_infer_dt(self._results_ndob['log_arrays']),
                link_active=_get_link_active(self._results_ndob['log_arrays']),
            )
        except Exception as exc:
            fig, ax = plt.subplots(figsize=self.style.get_figure_size('2x1'), constrained_layout=self._get_layout_mode())
            ax.text(0.5, 0.5, f'Stroke Metrics unavailable:\n{exc}', ha='center', va='center', transform=ax.transAxes)
            ax.set_axis_off()
            return fig

        fig, (ax_tip, ax_tilt) = plt.subplots(2, 1, figsize=self.style.get_figure_size('2x1'),
                                              sharex=True, constrained_layout=self._get_layout_mode())

        lw    = self.style.linewidth_primary
        alpha = self.style.alpha_primary

        # Panel 1: Azimuth (Tip)
        for m, label, color, ls in [
            (m_pid,  'PID',      ControllerColors.PID,      '-'),
            (m_fbl,  'FBL',      ControllerColors.FBL,      '-'),
            (m_ndob, 'FBL+NDOB', ControllerColors.FBL_NDOB, '-'),
        ]:
            ax_tip.plot(m.time, m.scr_timeseries_tip, color=color, lw=lw,
                        alpha=alpha, linestyle=ls, label=f'{label}')

        ax_tip.axhline(100.0, color='red', lw=1.5, linestyle=':', label='Saturation Limit')
        
        # Calculate upper y-limit based on actual max data
        max_tip = max([np.max(m.scr_timeseries_tip) for m in [m_pid, m_fbl, m_ndob]] + [110.0])
        ax_tip.axhspan(100.0, max_tip * 1.05, alpha=0.07, color='red')
        ax_tip.axhspan(0.0, 80.0, alpha=0.04, color='green')
        ax_tip.set_ylim(bottom=0.0, top=max_tip * 1.05)
        ax_tip.set_ylabel('Utilization Ratio [%]', fontsize=self.style.axis_label_fontsize, fontweight='bold')
        ax_tip.set_title(r'(a) Azimuth Axis (Tip) — Instantaneous Stroke Utilization $\left( \frac{|\theta_{FSM}(t)|}{\Theta_{max}} \right)$',
                         fontsize=self.style.title_fontsize, fontweight='bold')
        ax_tip.legend(loc='upper right', fontsize=self.style.legend_fontsize, ncol=4)
        ax_tip.grid(True, alpha=self.style.grid_alpha, linestyle=self.style.grid_linestyle)

        # Panel 2: Elevation (Tilt)
        for m, label, color, ls in [
            (m_pid,  'PID',      ControllerColors.PID,      '-'),
            (m_fbl,  'FBL',      ControllerColors.FBL,      '-'),
            (m_ndob, 'FBL+NDOB', ControllerColors.FBL_NDOB, '-'),
        ]:
            ax_tilt.plot(m.time, m.scr_timeseries_tilt, color=color, lw=lw,
                         alpha=alpha, linestyle=ls, label=f'{label}')

        ax_tilt.axhline(100.0, color='red', lw=1.5, linestyle=':', label='Saturation Limit')
        
        # Calculate upper y-limit based on actual max data
        max_tilt = max([np.max(m.scr_timeseries_tilt) for m in [m_pid, m_fbl, m_ndob]] + [110.0])
        ax_tilt.axhspan(100.0, max_tilt * 1.05, alpha=0.07, color='red')
        ax_tilt.axhspan(0.0, 80.0, alpha=0.04, color='green')
        ax_tilt.set_ylim(bottom=0.0, top=max_tilt * 1.05)
        ax_tilt.set_ylabel('Utilization Ratio [%]', fontsize=self.style.axis_label_fontsize, fontweight='bold')
        ax_tilt.set_title(r'(b) Elevation Axis (Tilt) — Instantaneous Stroke Utilization $\left( \frac{|\theta_{FSM}(t)|}{\Theta_{max}} \right)$',
                          fontsize=self.style.title_fontsize, fontweight='bold')
        # Use same legend for consistency
        ax_tilt.legend(loc='upper right', fontsize=self.style.legend_fontsize, ncol=4)
        ax_tilt.grid(True, alpha=self.style.grid_alpha, linestyle=self.style.grid_linestyle)
        ax_tilt.set_xlabel('Time [s]', fontsize=self.style.axis_label_fontsize, fontweight='bold')

        return fig
    # ─────────────────────────────────────────────────────────────────────────
    def _plot_stroke_margin_summary(self) -> plt.Figure:
        """Figure 15: Dynamic Stroke Margin and Stroke Consumption Summary.

        Two-panel layout:
          Panel 1 (full-width): DSM grouped bar chart, per controller.
          Panel 2 (full-width): Summary table with all benchmark values.
        """
        stroke_lim_arr = self._results_ndob['log_arrays'].get('fsm_stroke_limit_rad', None)
        theta_max = float(stroke_lim_arr[0]) if stroke_lim_arr is not None else 0.010

        calculator = StrokeMetrics(theta_max=theta_max, jitter_cutoff_hz=50.0, filter_order=4)

        def _infer_dt(log_arrays):
            t = np.asarray(log_arrays['time'])
            return float(np.median(np.diff(t))) if len(t) > 1 else 1e-4

        def _get_link_active(log_arrays):
            """Extract is_beam_on_sensor as a numpy bool array, or None."""
            raw = log_arrays.get('is_beam_on_sensor', None)
            if raw is not None:
                return np.asarray(raw, dtype=bool)
            return None

        try:
            m_pid = calculator.compute(
                time=np.asarray(self._results_pid['log_arrays']['time']),
                fsm_tip=np.asarray(self._results_pid['log_arrays']['fsm_tip']),
                fsm_tilt=np.asarray(self._results_pid['log_arrays']['fsm_tilt']),
                dt=_infer_dt(self._results_pid['log_arrays']),
                link_active=_get_link_active(self._results_pid['log_arrays']),
            )
            m_fbl = calculator.compute(
                time=np.asarray(self._results_fbl['log_arrays']['time']),
                fsm_tip=np.asarray(self._results_fbl['log_arrays']['fsm_tip']),
                fsm_tilt=np.asarray(self._results_fbl['log_arrays']['fsm_tilt']),
                dt=_infer_dt(self._results_fbl['log_arrays']),
                link_active=_get_link_active(self._results_fbl['log_arrays']),
            )
            m_ndob = calculator.compute(
                time=np.asarray(self._results_ndob['log_arrays']['time']),
                fsm_tip=np.asarray(self._results_ndob['log_arrays']['fsm_tip']),
                fsm_tilt=np.asarray(self._results_ndob['log_arrays']['fsm_tilt']),
                dt=_infer_dt(self._results_ndob['log_arrays']),
                link_active=_get_link_active(self._results_ndob['log_arrays']),
            )
        except Exception as exc:
            fig, ax = plt.subplots(figsize=self.style.get_figure_size('2x1'), constrained_layout=self._get_layout_mode())
            ax.text(0.5, 0.5, f'Stroke Summary unavailable:\n{exc}', ha='center', va='center', transform=ax.transAxes)
            ax.set_axis_off()
            return fig

        fig = plt.figure(figsize=self.style.get_figure_size('1x1'), constrained_layout=self._get_layout_mode())
        ax_bar = fig.add_subplot(111)

        ctrl_labels = ['PID', 'FBL', 'FBL+NDOB']
        scr_tip_vals  = np.array([m_pid.scr_tip,  m_fbl.scr_tip,  m_ndob.scr_tip])
        scr_tilt_vals = np.array([m_pid.scr_tilt, m_fbl.scr_tilt, m_ndob.scr_tilt])
        bar_colors = [ControllerColors.PID, ControllerColors.FBL, ControllerColors.FBL_NDOB]

        x = np.arange(len(ctrl_labels))
        bar_w = 0.35
        bars_tip  = ax_bar.bar(x - bar_w / 2, scr_tip_vals,  bar_w,
                               color=bar_colors, alpha=0.85, edgecolor='black', linewidth=1.2, label='Tip')
        bars_tilt = ax_bar.bar(x + bar_w / 2, scr_tilt_vals, bar_w,
                               color=bar_colors, alpha=0.50, edgecolor='black', linewidth=1.2, hatch='///', label='Tilt')
        ax_bar.axhline(100.0, color='red', lw=2.0, linestyle='--', label='Saturation Boundary (100%)')
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(ctrl_labels, fontsize=self.style.axis_label_fontsize, fontweight='bold')
        ax_bar.set_ylabel('SCR_RMS [%]', fontsize=self.style.axis_label_fontsize, fontweight='bold')
        ax_bar.set_title('RMS Stroke Consumption Ratio (SCR_RMS)',
                         fontsize=self.style.title_fontsize, fontweight='bold')
        for bar in list(bars_tip) + list(bars_tilt):
            h = bar.get_height()
            ax_bar.annotate(
                f'{h:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4),
                textcoords='offset points',
                ha='center', fontsize=9, fontweight='bold',
                color='black' if h <= 100.0 else '#cc0000'
            )
        ax_bar.legend(loc='best', fontsize=self.style.legend_fontsize, ncol=1)
        ax_bar.grid(True, axis='y', alpha=self.style.grid_alpha, linestyle=self.style.grid_linestyle)

        return fig

    def _plot_benchmark_table(self) -> plt.Figure:
        """Figure 16: Stroke Consumption Benchmark Results summary table."""
        stroke_lim_arr = self._results_ndob['log_arrays'].get('fsm_stroke_limit_rad', None)
        theta_max = float(stroke_lim_arr[0]) if stroke_lim_arr is not None else 0.010

        calculator = StrokeMetrics(theta_max=theta_max, jitter_cutoff_hz=50.0, filter_order=4)

        def _infer_dt(log_arrays):
            t = np.asarray(log_arrays['time'])
            return float(np.median(np.diff(t))) if len(t) > 1 else 1e-4

        def _get_link_active(log_arrays):
            raw = log_arrays.get('is_beam_on_sensor', None)
            if raw is not None:
                return np.asarray(raw, dtype=bool)
            return None

        # Helper for Post-FSM residuals
        _to_urad = 1e6
        def _get_residual(results, key):
            arr = results['log_arrays'].get(key, None)
            if arr is None:
                los  = np.array(results['log_arrays'].get('los_error_x' if 'x' in key else 'los_error_y', [0]))
                fsm  = np.array(results['log_arrays'].get('fsm_tip' if 'x' in key else 'fsm_tilt', [0]))
                return los - 2.0 * fsm
            return np.array(arr)

        def _calc_rms(results, t_arr, key):
            res = _get_residual(results, key) * _to_urad
            t = np.array(t_arr)
            if len(t) > 10:
                steady_start = t[-1] * 0.5  # Last 50%
                mask = t >= steady_start
                if mask.sum() > 0:
                    return float(np.sqrt(np.mean(res[mask]**2)))
            return 0.0

        rms_pid_x = _calc_rms(self._results_pid, self._t_pid, 'fsm_residual_error_x')
        rms_fbl_x = _calc_rms(self._results_fbl, self._t_fbl, 'fsm_residual_error_x')
        rms_ndob_x = _calc_rms(self._results_ndob, self._t_ndob, 'fsm_residual_error_x')

        rms_pid_y = _calc_rms(self._results_pid, self._t_pid, 'fsm_residual_error_y')
        rms_fbl_y = _calc_rms(self._results_fbl, self._t_fbl, 'fsm_residual_error_y')
        rms_ndob_y = _calc_rms(self._results_ndob, self._t_ndob, 'fsm_residual_error_y')

        try:
            m_pid = calculator.compute(
                time=np.asarray(self._results_pid['log_arrays']['time']),
                fsm_tip=np.asarray(self._results_pid['log_arrays']['fsm_tip']),
                fsm_tilt=np.asarray(self._results_pid['log_arrays']['fsm_tilt']),
                dt=_infer_dt(self._results_pid['log_arrays']),
                link_active=_get_link_active(self._results_pid['log_arrays']),
            )
            m_fbl = calculator.compute(
                time=np.asarray(self._results_fbl['log_arrays']['time']),
                fsm_tip=np.asarray(self._results_fbl['log_arrays']['fsm_tip']),
                fsm_tilt=np.asarray(self._results_fbl['log_arrays']['fsm_tilt']),
                dt=_infer_dt(self._results_fbl['log_arrays']),
                link_active=_get_link_active(self._results_fbl['log_arrays']),
            )
            m_ndob = calculator.compute(
                time=np.asarray(self._results_ndob['log_arrays']['time']),
                fsm_tip=np.asarray(self._results_ndob['log_arrays']['fsm_tip']),
                fsm_tilt=np.asarray(self._results_ndob['log_arrays']['fsm_tilt']),
                dt=_infer_dt(self._results_ndob['log_arrays']),
                link_active=_get_link_active(self._results_ndob['log_arrays']),
            )
        except Exception as exc:
            fig, ax = plt.subplots(figsize=self.style.get_figure_size('1x1'), constrained_layout=self._get_layout_mode())
            ax.text(0.5, 0.5, f'Stroke Summary unavailable:\n{exc}', ha='center', va='center', transform=ax.transAxes)
            ax.set_axis_off()
            return fig

        fig = plt.figure(figsize=self.style.get_figure_size('1x1'), constrained_layout=self._get_layout_mode())
        ax_table = fig.add_subplot(111)

        ax_table.set_axis_off()
        table_rows = [
            [r'Post-FSM Residual Tip [µrad]',
             f'{rms_pid_x:.2f}', f'{rms_fbl_x:.2f}', f'{rms_ndob_x:.2f}'],
            [r'Post-FSM Residual Tilt [µrad]',
             f'{rms_pid_y:.2f}', f'{rms_fbl_y:.2f}', f'{rms_ndob_y:.2f}'],
            [r'SCR_{rms} Tip [%]',
             f'{m_pid.scr_tip:.1f}', f'{m_fbl.scr_tip:.1f}', f'{m_ndob.scr_tip:.1f}'],
            [r'SCR_{rms} Tilt [%]',
             f'{m_pid.scr_tilt:.1f}', f'{m_fbl.scr_tilt:.1f}', f'{m_ndob.scr_tilt:.1f}'],
            [r'$S_{bias}$ Tip [mrad]',
             f'{m_pid.s_bias_tip_mrad:.3f}', f'{m_fbl.s_bias_tip_mrad:.3f}', f'{m_ndob.s_bias_tip_mrad:.3f}'],
            [r'$S_{bias}$ Tilt [mrad]',
             f'{m_pid.s_bias_tilt_mrad:.3f}', f'{m_fbl.s_bias_tilt_mrad:.3f}', f'{m_ndob.s_bias_tilt_mrad:.3f}'],
            [r'$\sigma_{jitter}$ Tip [mrad]',
             f'{m_pid.sigma_jitter_tip*1e3:.4f}', f'{m_fbl.sigma_jitter_tip*1e3:.4f}', f'{m_ndob.sigma_jitter_tip*1e3:.4f}'],
            [r'DSM Tip [mrad]',
             f'{m_pid.dsm_tip_mrad:.3f}', f'{m_fbl.dsm_tip_mrad:.3f}', f'{m_ndob.dsm_tip_mrad:.3f}'],
            [r'DSM Tilt [mrad]',
             f'{m_pid.dsm_tilt_mrad:.3f}', f'{m_fbl.dsm_tilt_mrad:.3f}', f'{m_ndob.dsm_tilt_mrad:.3f}'],
            #[r'Link Safe (Tip)',
            # '✓' if m_pid.is_link_safe_tip  else '✗',
            # '✓' if m_fbl.is_link_safe_tip  else '✗',
            # '✓' if m_ndob.is_link_safe_tip else '✗'],
        ]
        col_labels = ['Metric', 'PID', 'FBL', 'FBL+NDOB']
        col_widths = [0.25, 0.20, 0.20, 0.25]
        tbl = ax_table.table(
            cellText=table_rows,
            colLabels=col_labels,
            colWidths=col_widths,
            loc='center',
            cellLoc='center',
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1.0, 1.8)
        for (row, col), cell in tbl.get_celld().items():
            if row == 0:
                cell.set_facecolor('#2d3e50')
                cell.set_text_props(color='white', fontweight='bold')
            elif col == 3:
                cell.set_facecolor('#e8f4ea')
            elif row % 2 == 0:
                cell.set_facecolor('#f5f5f5')
        ax_table.set_title('Stroke Consumption Benchmark Results',
                           fontsize=self.style.title_fontsize, fontweight='bold', pad=10)

        return fig

    def _save_all_figures(self) -> None:
        """Save all generated figures to disk in journal-quality PDF format."""
        mode_str = "interactive" if self.interactive else "static"
        print(f"\n[OK] Generated 16 research-quality {mode_str} figures (300 DPI, LaTeX labels)")
        print("Saving figures to disk in journal-quality PDF format...")
        
        figure_names = {
            'fig1_position': 'fig1_position_tracking.pdf',
            'fig2_error': 'fig2_tracking_error_handover.pdf',
            'fig3_torque': 'fig3_torque_ndob.pdf',
            'fig4_velocity': 'fig4_velocities.pdf',
            'fig5_phase': 'fig5_phase_plane.pdf',
            'fig6_los': 'fig6_los_errors.pdf',
            'fig7_summary': 'fig7_performance_summary.pdf',
            'fig8_ekf': 'fig8_state_estimates.pdf',
            'fig9_fsm': 'fig9_fsm_performance.pdf',
            'fig10_internal': 'fig10_internal_signals.pdf',
            'fig11_ekf_diag': 'fig11_ekf_adaptive_tuning.pdf',
            'fig12_disturbance': 'fig12_environmental_disturbances.pdf',
            'fig13_statistics': 'fig13_disturbance_statistics.pdf',
            'fig14_stroke_metrics': 'fig14_stroke_consumption_ratio.pdf',
            'fig15_stroke_summary': 'fig15_stroke_margin_summary.pdf',
            'fig16_benchmark_table': 'fig16_benchmark_table_results.pdf',
        }
        
        for key, filename in figure_names.items():
            if key in self.figures:
                filepath = self.style.output_dir / filename
                try:
                    # Journal-quality PDF save parameters
                    self.figures[key].savefig(
                        str(filepath),  # Convert Path to string for Windows compatibility
                        format='pdf',
                        dpi=300,  # Journal standard resolution
                        bbox_inches='tight',
                        facecolor='white',
                        edgecolor='none',
                        metadata={
                            'Title': filename.replace('.pdf', '').replace('_', ' ').title(),
                            'Author': 'MicroPrecisionGimbal Digital Twin',
                            'Subject': 'Aerospace Gimbal Control Research',
                            'Creator': 'matplotlib + ResearchComparisonPlotter',
                        }
                    )
                    # Ensure file is written and closed properly
                    if hasattr(self.figures[key].canvas, 'flush_events'):
                        self.figures[key].canvas.flush_events()
                    time.sleep(0.01)  # Brief pause for file system sync
                except Exception as e:
                    print(f"  [ERROR] Failed to save {filename}: {e}")
        
        print(f"  [OK] Saved 16 PDF figures to {self.style.output_dir.absolute()}/")
        print("  [OK] Format: PDF (vector), 300 DPI, bbox='tight' (journal-ready)")
        print("FIGURE GENERATION COMPLETE")
        print("=" * 70)


def plot_research_comparison(
    results_pid: Dict,
    results_fbl: Dict,
    results_ndob: Dict,
    target_az_deg: float,
    target_el_deg: float
) -> None:
    """
    Backward-compatible wrapper for ResearchComparisonPlotter.
    
    This function provides the same interface as the original
    plot_research_comparison function for drop-in replacement.
    
    Parameters
    ----------
    results_pid : Dict
        PID controller simulation results
    results_fbl : Dict
        FBL controller simulation results
    results_ndob : Dict
        FBL+NDOB controller simulation results
    target_az_deg : float
        Target azimuth [degrees]
    target_el_deg : float
        Target elevation [degrees]
    """
    plotter = ResearchComparisonPlotter(save_figures=True, show_figures=True)
    plotter.plot_all(results_pid, results_fbl, results_ndob, target_az_deg, target_el_deg)
