"""
Research-Quality Comparative Plotter for Three-Way Controller Analysis.

This module provides publication-quality visualization for comparing
PID, Feedback Linearization (FBL), and FBL+NDOB controllers in the
lasercom gimbal pointing system.

Figures Generated
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

from lasercom_digital_twin.core.plots.style_config import (
    PlotStyleConfig,
    ControllerColors,
    AxisColors,
    DisturbanceColors,
    configure_matplotlib_defaults
)
from lasercom_digital_twin.core.plots.metrics_utils import compute_tracking_metrics


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
        show_figures: bool = True
    ):
        """Initialize plotter with style configuration."""
        self.style = style or PlotStyleConfig()
        self.save_figures = save_figures
        self.show_figures = show_figures
        
        # Ensure output directory exists
        if self.save_figures:
            self.style.output_dir.mkdir(exist_ok=True)
        
        # Store generated figures for access
        self.figures: Dict[str, plt.Figure] = {}
    
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
        
        # Save figures
        if self.save_figures:
            self._save_all_figures()
        
        # Show figures
        if self.show_figures:
            plt.show()
        
        return self.figures
    
    def _plot_position_tracking(self) -> plt.Figure:
        """Figure 1: Gimbal Position Tracking (Az & El with Commands)."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.style.get_figure_size('2x1'),
                                        sharex=True, constrained_layout=True)
        
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
        ax1.set_title('Gimbal Azimuth Position', fontsize=self.style.title_fontsize,
                      fontweight='bold')
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
        ax2.set_title('Gimbal Elevation Position', fontsize=self.style.title_fontsize,
                      fontweight='bold')
        ax2.legend(loc='best', fontsize=self.style.legend_fontsize)
        ax2.grid(True, alpha=self.style.grid_alpha, linestyle=self.style.grid_linestyle)
        
        fig.suptitle('Gimbal Position Tracking', fontsize=self.style.suptitle_fontsize,
                     fontweight='bold')
        
        return fig
    
    def _plot_tracking_error(self) -> plt.Figure:
        """Figure 2: Tracking Error with Handover Thresholds (THE MONEY SHOT)."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.style.get_figure_size('2x1'),
                                        sharex=True, constrained_layout=True)
        
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
        
        # Threshold lines
        ax1.axhline(0.8, color=ControllerColors.HANDOVER, linewidth=lw, linestyle=':',
                    alpha=self.style.alpha_threshold, label='FSM Handover (0.8°)')
        ax1.axhline(1.0, color=ControllerColors.THRESHOLD, linewidth=lw, linestyle=':',
                    alpha=0.5, label='Performance Limit (1.0°)')
        
        ax1.set_ylabel('Azimuth Error [deg]', fontsize=self.style.axis_label_fontsize,
                       fontweight='bold')
        ax1.set_title('Azimuth Tracking Error (with FSM Handover Threshold)',
                      fontsize=self.style.title_fontsize, fontweight='bold')
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
        
        ax2.axhline(0.8, color=ControllerColors.HANDOVER, linewidth=lw, linestyle=':',
                    alpha=self.style.alpha_threshold, label='FSM Handover (0.8°)')
        ax2.axhline(1.0, color=ControllerColors.THRESHOLD, linewidth=lw, linestyle=':',
                    alpha=0.5, label='Performance Limit (1.0°)')
        
        ax2.set_ylabel('Elevation Error [deg]', fontsize=self.style.axis_label_fontsize,
                       fontweight='bold')
        ax2.set_xlabel('Time [s]', fontsize=self.style.axis_label_fontsize, fontweight='bold')
        ax2.set_title('Elevation Tracking Error (with FSM Handover Threshold)',
                      fontsize=self.style.title_fontsize, fontweight='bold')
        ax2.legend(loc='best', fontsize=self.style.legend_fontsize)
        ax2.grid(True, alpha=self.style.grid_alpha, linestyle=self.style.grid_linestyle)
        ax2.set_yscale('log')
        
        fig.suptitle('Tracking Error with Precision Thresholds',
                     fontsize=self.style.suptitle_fontsize, fontweight='bold')
        
        return fig
    
    def _plot_control_torques(self) -> plt.Figure:
        """Figure 3: Control Torques & NDOB Disturbance Estimation."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.style.get_figure_size('2x2'),
                                                      constrained_layout=True)
        
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
        ax1.set_title('Azimuth Motor Control Effort', fontsize=self.style.title_fontsize, fontweight='bold')
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
        ax2.set_title('Elevation Motor Control Effort', fontsize=self.style.title_fontsize, fontweight='bold')
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
            ax3.set_title('Azimuth Disturbance Estimation Accuracy',
                          fontsize=self.style.title_fontsize, fontweight='bold')
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
            ax4.set_title('Elevation Disturbance Estimation Accuracy',
                          fontsize=self.style.title_fontsize, fontweight='bold')
            ax4.legend(loc='best', fontsize=self.style.legend_fontsize)
        else:
            ax4.text(0.5, 0.5, 'NDOB Not Enabled', ha='center', va='center',
                     transform=ax4.transAxes, fontsize=self.style.axis_label_fontsize)
        ax4.grid(True, alpha=self.style.grid_alpha, linestyle=self.style.grid_linestyle)
        
        fig.suptitle('Motor Control Torques', fontsize=self.style.suptitle_fontsize, fontweight='bold')
        
        return fig
    
    def _plot_velocities(self) -> plt.Figure:
        """Figure 4: Gimbal Angular Velocities."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.style.get_figure_size('2x1'),
                                        sharex=True, constrained_layout=True)
        
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
        ax1.set_title('Gimbal Azimuth Velocity', fontsize=self.style.title_fontsize, fontweight='bold')
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
        ax2.set_title('Gimbal Elevation Velocity', fontsize=self.style.title_fontsize, fontweight='bold')
        ax2.legend(loc='best', fontsize=self.style.legend_fontsize)
        ax2.grid(True, alpha=self.style.grid_alpha, linestyle=self.style.grid_linestyle)
        
        fig.suptitle('Gimbal Angular Velocities', fontsize=self.style.suptitle_fontsize, fontweight='bold')
        
        return fig
    
    def _plot_phase_plane(self) -> plt.Figure:
        """Figure 5: Phase Plane Trajectories."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.style.get_figure_size('1x2'),
                                        constrained_layout=True)
        
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
        ax1.set_title('Az Phase Plane', fontsize=self.style.title_fontsize, fontweight='bold')
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
        ax2.set_title('El Phase Plane', fontsize=self.style.title_fontsize, fontweight='bold')
        ax2.legend(loc='best', fontsize=self.style.legend_fontsize)
        ax2.grid(True, alpha=self.style.grid_alpha, linestyle=self.style.grid_linestyle)
        
        fig.suptitle('Gimbal Phase Plane', fontsize=self.style.suptitle_fontsize, fontweight='bold')
        
        return fig
    
    def _plot_los_errors(self) -> plt.Figure:
        """Figure 6: Line-of-Sight Errors."""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=self.style.get_figure_size('3x1'),
                                             sharex=True, constrained_layout=True)
        
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
        ax1.set_title('Line-of-Sight Error X-Axis', fontsize=self.style.title_fontsize, fontweight='bold')
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
        ax2.set_title('Line-of-Sight Error Y-Axis', fontsize=self.style.title_fontsize, fontweight='bold')
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
        ax3.set_title('Total Line-of-Sight Error Magnitude', fontsize=self.style.title_fontsize, fontweight='bold')
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
                                                      constrained_layout=True)
        
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
        ax1.set_title('Settling Time (2% Criterion)', fontsize=self.style.title_fontsize, fontweight='bold')
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
        ax2.set_title('Steady-State Error', fontsize=self.style.title_fontsize, fontweight='bold')
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
        ax3.set_title('RMS Line-of-Sight Error', fontsize=self.style.title_fontsize, fontweight='bold')
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
        ax4.set_title('Control Effort', fontsize=self.style.title_fontsize, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(controllers, fontsize=self.style.legend_fontsize)
        ax4.grid(True, alpha=self.style.grid_alpha, axis='y', linestyle=self.style.grid_linestyle)
        
        fig.suptitle('Performance Metrics Summary', fontsize=self.style.suptitle_fontsize, fontweight='bold')
        
        return fig
    
    def _plot_ekf_performance(self) -> plt.Figure:
        """Figure 8: EKF State Estimation Performance."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.style.get_figure_size('2x2'),
                                                      constrained_layout=True)
        
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
        ax1.set_title('Azimuth Position Estimate vs Truth', fontsize=self.style.title_fontsize, fontweight='bold')
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
        ax2.set_title('Elevation Position Estimate vs Truth', fontsize=self.style.title_fontsize, fontweight='bold')
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
        ax3.set_title('Azimuth Rate Estimate vs Truth', fontsize=self.style.title_fontsize, fontweight='bold')
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
        ax4.set_title('Elevation Rate Estimate vs Truth', fontsize=self.style.title_fontsize, fontweight='bold')
        ax4.legend(loc='best', fontsize=self.style.legend_fontsize)
        ax4.grid(True, alpha=self.style.grid_alpha, linestyle=self.style.grid_linestyle)
        
        fig.suptitle('EKF Performance (Estimate vs Ground Truth)',
                     fontsize=self.style.suptitle_fontsize, fontweight='bold')
        
        return fig
    
    def _plot_fsm_performance(self) -> plt.Figure:
        """Figure 9: Fine Steering Mirror Performance."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.style.get_figure_size('2x1'),
                                        sharex=True, constrained_layout=True)
        
        lw = self.style.linewidth_primary
        alpha = self.style.alpha_primary
        
        # FSM Tip
        ax1.plot(self._t_pid, np.rad2deg(self._results_pid['log_arrays']['fsm_tip']),
                 color=ControllerColors.PID, linewidth=lw, label='PID Tip', alpha=alpha)
        ax1.plot(self._t_fbl, np.rad2deg(self._results_fbl['log_arrays']['fsm_tip']),
                 color=ControllerColors.FBL, linewidth=lw, label='FBL Tip', alpha=alpha)
        ax1.plot(self._t_ndob, np.rad2deg(self._results_ndob['log_arrays']['fsm_tip']),
                 color=ControllerColors.FBL_NDOB, linewidth=lw, label='FBL+NDOB Tip', alpha=alpha)
        # Commands
        ax1.plot(self._t_pid, np.rad2deg(self._results_pid['log_arrays']['fsm_cmd_tip']),
                 color=ControllerColors.PID, linewidth=lw, linestyle='--', alpha=0.5, label='PID Cmd')
        ax1.plot(self._t_fbl, np.rad2deg(self._results_fbl['log_arrays']['fsm_cmd_tip']),
                 color=ControllerColors.FBL, linewidth=lw, linestyle='--', alpha=0.5, label='FBL Cmd')
        ax1.plot(self._t_ndob, np.rad2deg(self._results_ndob['log_arrays']['fsm_cmd_tip']),
                 color=ControllerColors.FBL_NDOB, linewidth=lw, linestyle='--', alpha=0.5, label='NDOB Cmd')
        ax1.set_ylabel('Tip [deg]', fontsize=self.style.axis_label_fontsize, fontweight='bold')
        ax1.set_title('FSM Tip Axis Performance', fontsize=self.style.title_fontsize, fontweight='bold')
        ax1.legend(loc='best', fontsize=self.style.legend_fontsize, ncol=2)
        ax1.grid(True, alpha=self.style.grid_alpha, linestyle=self.style.grid_linestyle)
        
        # FSM Tilt
        ax2.plot(self._t_pid, np.rad2deg(self._results_pid['log_arrays']['fsm_tilt']),
                 color=ControllerColors.PID, linewidth=lw, label='PID Tilt', alpha=alpha)
        ax2.plot(self._t_fbl, np.rad2deg(self._results_fbl['log_arrays']['fsm_tilt']),
                 color=ControllerColors.FBL, linewidth=lw, label='FBL Tilt', alpha=alpha)
        ax2.plot(self._t_ndob, np.rad2deg(self._results_ndob['log_arrays']['fsm_tilt']),
                 color=ControllerColors.FBL_NDOB, linewidth=lw, label='FBL+NDOB Tilt', alpha=alpha)
        ax2.plot(self._t_pid, np.rad2deg(self._results_pid['log_arrays']['fsm_cmd_tilt']),
                 color=ControllerColors.PID, linewidth=lw, linestyle='--', alpha=0.5)
        ax2.plot(self._t_fbl, np.rad2deg(self._results_fbl['log_arrays']['fsm_cmd_tilt']),
                 color=ControllerColors.FBL, linewidth=lw, linestyle='--', alpha=0.5)
        ax2.plot(self._t_ndob, np.rad2deg(self._results_ndob['log_arrays']['fsm_cmd_tilt']),
                 color=ControllerColors.FBL_NDOB, linewidth=lw, linestyle='--', alpha=0.5)
        ax2.set_ylabel('Tilt [deg]', fontsize=self.style.axis_label_fontsize, fontweight='bold')
        ax2.set_xlabel('Time [s]', fontsize=self.style.axis_label_fontsize, fontweight='bold')
        ax2.set_title('FSM Tilt Axis Performance', fontsize=self.style.title_fontsize, fontweight='bold')
        ax2.legend(loc='best', fontsize=self.style.legend_fontsize)
        ax2.grid(True, alpha=self.style.grid_alpha, linestyle=self.style.grid_linestyle)
        
        fig.suptitle('FSM Tip/Tilt Response vs Commands',
                     fontsize=self.style.suptitle_fontsize, fontweight='bold')
        
        return fig
    
    def _plot_internal_signals(self) -> plt.Figure:
        """Figure 10: Internal Control Signal & Disturbance Observer."""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=self.style.get_figure_size('3x1_tall'),
                                             sharex=True, constrained_layout=True)
        
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
        ax1.set_title(r'Virtual Control Input (Outer Loop $v = \ddot{q}_{ref} + K_p e + K_d \dot{e}$)',
                      fontsize=self.style.title_fontsize, fontweight='bold')
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
        ax2.set_title(r'Commanded Motor Torque ($\tau = Mv + C\dot{q} + G - \hat{d}$)',
                      fontsize=self.style.title_fontsize, fontweight='bold')
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
        ax3.set_xlabel('Time [s]', fontsize=self.style.axis_label_fontsize, fontweight='bold')
        ax3.set_title(r'NDOB Disturbance Estimate ($\hat{d} = z + L M(q) \dot{q}$)',
                      fontsize=self.style.title_fontsize, fontweight='bold')
        ax3.legend(loc='best', fontsize=self.style.legend_fontsize, ncol=2)
        ax3.grid(True, alpha=self.style.grid_alpha, linestyle=self.style.grid_linestyle)
        
        fig.suptitle('Figure 10: Internal Control Signal & Disturbance Observer Analysis',
                     fontsize=self.style.suptitle_fontsize, fontweight='bold')
        
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
            ax1.set_title('(a) EKF State Covariance Evolution (Log Scale)',
                          fontsize=self.style.title_fontsize, fontweight='bold')
            ax1.legend(loc='best', fontsize=self.style.legend_fontsize, ncol=3)
        else:
            ax1.text(0.5, 0.5, 'EKF Covariance Logging Not Yet Implemented\n(Requires simulation_runner enhancement)',
                     ha='center', va='center', transform=ax1.transAxes, fontsize=self.style.axis_label_fontsize)
            ax1.set_ylabel('Covariance', fontsize=self.style.axis_label_fontsize)
            ax1.set_title('(a) EKF State Covariance Evolution', fontsize=self.style.title_fontsize, fontweight='bold')
        ax1.grid(True, alpha=self.style.grid_alpha, linestyle=self.style.grid_linestyle)
        
        # Subplot 2: Innovation Residuals
        if 'ekf_innovation_enc_az' in log:
            ax2.plot(t, np.rad2deg(log['ekf_innovation_enc_az']), 'b-', linewidth=2, label='Encoder Az Innovation')
            ax2.plot(t, np.rad2deg(log['ekf_innovation_enc_el']), 'r-', linewidth=2, label='Encoder El Innovation')
            ax2.axhline(0, color='k', linestyle='--', linewidth=2, alpha=0.5)
            ax2.set_ylabel('Innovation [deg]', fontsize=self.style.axis_label_fontsize, fontweight='bold')
            ax2.set_title('(b) Measurement Innovation (Encoder Residuals)',
                          fontsize=self.style.title_fontsize, fontweight='bold')
            ax2.legend(loc='best', fontsize=self.style.legend_fontsize)
        else:
            ax2.text(0.5, 0.5, 'Innovation Logging Not Yet Implemented',
                     ha='center', va='center', transform=ax2.transAxes, fontsize=self.style.axis_label_fontsize)
            ax2.set_ylabel('Innovation [deg]', fontsize=self.style.axis_label_fontsize)
            ax2.set_title('(b) Measurement Innovation', fontsize=self.style.title_fontsize, fontweight='bold')
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
            ax3.set_title('(c) Innovation Bounds & Consistency Check',
                          fontsize=self.style.title_fontsize, fontweight='bold')
            ax3.legend(loc='best', fontsize=self.style.legend_fontsize)
        else:
            ax3.text(0.5, 0.5, '3-Sigma Bounds Logging Not Yet Implemented',
                     ha='center', va='center', transform=ax3.transAxes, fontsize=self.style.axis_label_fontsize)
            ax3.set_xlabel('Time [s]', fontsize=self.style.axis_label_fontsize)
            ax3.set_ylabel('Innovation [deg]', fontsize=self.style.axis_label_fontsize)
            ax3.set_title('(c) Innovation Consistency Check', fontsize=self.style.title_fontsize, fontweight='bold')
        ax3.grid(True, alpha=self.style.grid_alpha, linestyle=self.style.grid_linestyle)
        
        fig.suptitle('Figure 11: Extended Kalman Filter Diagnostics & Adaptive Tuning',
                     fontsize=self.style.suptitle_fontsize, fontweight='bold')
        
        return fig
    
    def _plot_disturbance_torques(self) -> plt.Figure:
        """Figure 12: Environmental Disturbance Torques."""
        fig, axes = plt.subplots(2, 2, figsize=self.style.get_figure_size('double_column'),
                                  constrained_layout=True)
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
            ax1.set_title('(a) Total Environmental Disturbance — Azimuth Axis',
                          fontsize=self.style.title_fontsize, fontweight='bold')
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
            ax2.set_title('(b) Total Environmental Disturbance — Elevation Axis',
                          fontsize=self.style.title_fontsize, fontweight='bold')
            ax2.legend(loc='upper right', fontsize=self.style.legend_fontsize, framealpha=0.95)
            ax2.grid(True, alpha=self.style.grid_alpha, linestyle=self.style.grid_linestyle)
            ax2.set_xlim([t[0], t[-1]])
            
            # Components - Azimuth
            ax3.plot(t, wind_az, color=DisturbanceColors.WIND, linewidth=lw, label='Wind/Gust (Dryden)', alpha=0.9)
            ax3.plot(t, vib_az, color=DisturbanceColors.VIBRATION, linewidth=lw, label='Structural Vibration', alpha=0.85)
            ax3.axhline(0, color='black', linewidth=lw, linestyle='--', alpha=0.5)
            ax3.set_ylabel(r'Torque [mN·m]', fontsize=self.style.axis_label_fontsize, fontweight='bold')
            ax3.set_xlabel('Time (s)', fontsize=self.style.axis_label_fontsize, fontweight='bold')
            ax3.set_title('(c) Disturbance Components — Azimuth Axis',
                          fontsize=self.style.title_fontsize, fontweight='bold')
            ax3.legend(loc='best', fontsize=self.style.legend_fontsize, framealpha=0.95)
            ax3.grid(True, alpha=self.style.grid_alpha, linestyle=self.style.grid_linestyle)
            ax3.set_xlim([t[0], t[-1]])
            
            # Components - Elevation
            ax4.plot(t, wind_el, color=DisturbanceColors.WIND, linewidth=lw, label='Wind/Gust (Dryden)', alpha=0.9)
            ax4.plot(t, vib_el, color=DisturbanceColors.VIBRATION, linewidth=lw, label='Structural Vibration', alpha=0.85)
            ax4.axhline(0, color='black', linewidth=lw, linestyle='--', alpha=0.5)
            ax4.set_ylabel(r'Torque [mN·m]', fontsize=self.style.axis_label_fontsize, fontweight='bold')
            ax4.set_xlabel('Time (s)', fontsize=self.style.axis_label_fontsize, fontweight='bold')
            ax4.set_title('(d) Disturbance Components — Elevation Axis',
                          fontsize=self.style.title_fontsize, fontweight='bold')
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
                ax.set_title(title, fontsize=self.style.title_fontsize, fontweight='bold')
        
        fig.suptitle(r'(Dryden Wind Turbulence + PSD-Based Structural Vibration — Plant Injection Only)',
                     fontsize=self.style.suptitle_fontsize, fontweight='bold')
        
        return fig
    
    def _plot_disturbance_statistics(self) -> plt.Figure:
        """Figure 13: Disturbance Statistics & PSD Analysis."""
        log = self._results_ndob['log_arrays']
        has_disturbance = 'tau_disturbance_az' in log and np.std(log['tau_disturbance_az']) > 1e-10
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), constrained_layout=True)
        
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
            ax1.set_title('(a) Disturbance Torque Statistical Distribution & Gaussian Fit',
                          fontsize=self.style.title_fontsize, fontweight='bold')
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
                ax2.set_title('(b) Power Spectral Density of Disturbance Torques',
                              fontsize=self.style.title_fontsize, fontweight='bold')
                ax2.legend(loc='upper right', fontsize=self.style.legend_fontsize, framealpha=0.95)
                ax2.grid(True, alpha=self.style.grid_alpha, linestyle=self.style.grid_linestyle, which='both')
                ax2.set_xlim([0, min(fs/2, 200)])
            else:
                ax2.text(0.5, 0.5, 'Insufficient data for PSD\n(requires longer simulation)',
                         ha='center', va='center', transform=ax2.transAxes, fontsize=self.style.axis_label_fontsize)
                ax2.set_xlabel('Frequency [Hz]', fontsize=self.style.axis_label_fontsize)
                ax2.set_ylabel('PSD', fontsize=self.style.axis_label_fontsize)
                ax2.set_title('(b) Power Spectral Density', fontsize=self.style.title_fontsize, fontweight='bold')
            
            fig.suptitle('Figure 13: Environmental Disturbance Torque Statistics & Frequency Analysis',
                         fontsize=self.style.suptitle_fontsize, fontweight='bold')
        else:
            ax1.text(0.5, 0.5, 'No disturbance data available', ha='center', va='center',
                     transform=ax1.transAxes, fontsize=self.style.axis_label_fontsize)
            ax1.set_title('Figure 13a: Disturbance Statistics (No Data)',
                          fontsize=self.style.title_fontsize, fontweight='bold')
            ax2.text(0.5, 0.5, 'No disturbance data available', ha='center', va='center',
                     transform=ax2.transAxes, fontsize=self.style.axis_label_fontsize)
            ax2.set_title('Figure 13b: PSD Analysis (No Data)',
                          fontsize=self.style.title_fontsize, fontweight='bold')
        
        return fig
    
    def _save_all_figures(self) -> None:
        """Save all generated figures to disk."""
        print("\n[OK] Generated 13 research-quality figures (300 DPI, LaTeX labels)")
        print("Saving figures to disk...")
        
        figure_names = {
            'fig1_position': 'fig1_position_tracking.png',
            'fig2_error': 'fig2_tracking_error_handover.png',
            'fig3_torque': 'fig3_torque_ndob.png',
            'fig4_velocity': 'fig4_velocities.png',
            'fig5_phase': 'fig5_phase_plane.png',
            'fig6_los': 'fig6_los_errors.png',
            'fig7_summary': 'fig7_performance_summary.png',
            'fig8_ekf': 'fig8_state_estimates.png',
            'fig9_fsm': 'fig9_fsm_performance.png',
            'fig10_internal': 'fig10_internal_signals.png',
            'fig11_ekf_diag': 'fig11_ekf_adaptive_tuning.png',
            'fig12_disturbance': 'fig12_environmental_disturbances.png',
            'fig13_statistics': 'fig13_disturbance_statistics.png',
        }
        
        for key, filename in figure_names.items():
            if key in self.figures:
                self.figures[key].savefig(
                    self.style.output_dir / filename,
                    dpi=self.style.dpi,
                    bbox_inches='tight'
                )
        
        print(f"  [OK] Saved 13 figures to {self.style.output_dir.absolute()}/")
        print("  [OK] Format: PNG, 300 DPI, bbox='tight' (publication-ready)")
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
