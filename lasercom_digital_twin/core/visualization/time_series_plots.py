"""
Time-Series Plots for Control System Debugging

This module provides specialized time-series visualizations for diagnosing
control system performance, estimator convergence, and actuator saturation.

Key Features:
-------------
- True vs Estimated LOS error comparison (EKF validation)
- Commanded vs Actual torque (saturation/anti-windup diagnosis)
- FSM performance tracking (fine pointing stage validation)
- Multi-axis synchronized plots for correlation analysis
- Automatic detection of anomalies (saturation events, estimator divergence)

Design Philosophy:
------------------
Every plot is designed to answer a specific debugging question:
- Is the estimator tracking correctly?
- Are actuators saturating?
- Is the FSM authority sufficient?
- Where are the transient disturbances occurring?
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Optional, Dict, List, Tuple, Union
import pandas as pd
import warnings


class TimelinePlotter:
    """
    Time-series plotter for control system debugging.
    
    Generates synchronized multi-axis plots showing the evolution of
    key system states, commands, and errors over time.
    
    Usage:
    ------
    >>> plotter = TimelinePlotter()
    >>> 
    >>> # Generate comprehensive debug plots
    >>> fig = plotter.plot_full_debug_suite(telemetry)
    >>> plt.savefig('debug_timeline.png', dpi=300)
    >>> 
    >>> # Or individual plots
    >>> fig, ax = plotter.plot_estimator_performance(telemetry)
    >>> fig, ax = plotter.plot_control_effort(telemetry)
    >>> fig, ax = plotter.plot_fsm_tracking(telemetry)
    """
    
    def __init__(
        self,
        figure_size: Tuple[int, int] = (14, 10),
        time_unit: str = 's',  # 's' or 'ms'
    ):
        """
        Initialize timeline plotter.
        
        Parameters
        ----------
        figure_size : tuple
            Default figure size (width, height) in inches
        time_unit : str
            Time axis unit: 's' (seconds) or 'ms' (milliseconds)
        """
        self.figure_size = figure_size
        self.time_unit = time_unit
        self.time_scale = 1000.0 if time_unit == 'ms' else 1.0
        self.time_label = 'Time (ms)' if time_unit == 'ms' else 'Time (s)'
    
    def plot_estimator_performance(
        self,
        telemetry: Union[Dict[str, List[float]], pd.DataFrame],
        time_window: Optional[Tuple[float, float]] = None,
        show_innovation: bool = True,
        title: Optional[str] = None,
        ax: Optional[plt.Axes] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot True vs Estimated LOS error for EKF validation.
        
        This is THE critical plot for debugging estimator performance.
        Shows whether the Extended Kalman Filter is properly tracking
        the true line-of-sight error.
        
        Parameters
        ----------
        telemetry : dict or DataFrame
            Telemetry data containing:
            - 'time': Time vector [s]
            - 'los_error_x', 'los_error_y': True LOS error [µrad]
            - 'est_error_x', 'est_error_y': Estimated LOS error [µrad]
            - 'estimator_converged': Convergence flag (optional)
        time_window : tuple, optional
            (start, end) time range [s]
        show_innovation : bool
            Plot innovation (true - estimated) as third panel
        title : str, optional
            Plot title
        ax : plt.Axes or array of Axes, optional
            Existing axes (must be 2 or 3 depending on show_innovation)
            
        Returns
        -------
        fig : plt.Figure
        ax : plt.Axes or array
            
        Example
        -------
        >>> plotter = TimelinePlotter()
        >>> fig, axes = plotter.plot_estimator_performance(telemetry)
        >>> # Check convergence: True and Estimated should align after ~1s
        """
        # Extract and mask data
        df = self._to_dataframe(telemetry, time_window)
        time = df['time'].values * self.time_scale
        
        # Extract errors
        true_x = df['los_error_x'].values
        true_y = df['los_error_y'].values
        est_x = df['est_error_x'].values if 'est_error_x' in df.columns else np.zeros_like(true_x)
        est_y = df['est_error_y'].values if 'est_error_y' in df.columns else np.zeros_like(true_y)
        
        # Check for convergence flag
        converged = df['estimator_converged'].values if 'estimator_converged' in df.columns else np.ones_like(time, dtype=bool)
        
        # Create figure
        if ax is None:
            n_panels = 3 if show_innovation else 2
            fig, axes = plt.subplots(n_panels, 1, figsize=self.figure_size, sharex=True)
            if not show_innovation:
                axes = [axes[0], axes[1]]
        else:
            fig = ax[0].figure if hasattr(ax, '__iter__') else ax.figure
            axes = ax if hasattr(ax, '__iter__') else [ax]
        
        # Panel 1: X-axis error
        axes[0].plot(time, true_x, 'b-', linewidth=1.5, label='True Error', alpha=0.8)
        axes[0].plot(time, est_x, 'r--', linewidth=1.5, label='Estimated Error', alpha=0.8)
        
        # Shade non-converged region
        if not np.all(converged):
            axes[0].axvspan(time[0], time[np.argmax(converged)], 
                           alpha=0.2, color='yellow', label='Convergence')
        
        axes[0].set_ylabel('X Error (µrad)', fontsize=11, fontweight='bold')
        axes[0].legend(loc='upper right', fontsize=9)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title(title or 'Estimator Performance: True vs Estimated LOS Error',
                         fontsize=13, fontweight='bold')
        
        # Panel 2: Y-axis error
        axes[1].plot(time, true_y, 'b-', linewidth=1.5, label='True Error', alpha=0.8)
        axes[1].plot(time, est_y, 'r--', linewidth=1.5, label='Estimated Error', alpha=0.8)
        
        if not np.all(converged):
            axes[1].axvspan(time[0], time[np.argmax(converged)],
                           alpha=0.2, color='yellow')
        
        axes[1].set_ylabel('Y Error (µrad)', fontsize=11, fontweight='bold')
        axes[1].legend(loc='upper right', fontsize=9)
        axes[1].grid(True, alpha=0.3)
        
        # Panel 3: Innovation (estimation error)
        if show_innovation:
            innov_x = true_x - est_x
            innov_y = true_y - est_y
            innov_mag = np.sqrt(innov_x**2 + innov_y**2)
            
            axes[2].plot(time, innov_mag, 'g-', linewidth=1.5, label='Innovation Magnitude')
            axes[2].axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
            
            # Mark RMS innovation
            rms_innov = np.sqrt(np.mean(innov_mag[converged]**2))
            axes[2].axhline(rms_innov, color='r', linestyle=':', linewidth=2,
                           label=f'RMS: {rms_innov:.2f} µrad')
            
            axes[2].set_ylabel('Innovation (µrad)', fontsize=11, fontweight='bold')
            axes[2].set_xlabel(self.time_label, fontsize=11, fontweight='bold')
            axes[2].legend(loc='upper right', fontsize=9)
            axes[2].grid(True, alpha=0.3)
        else:
            axes[1].set_xlabel(self.time_label, fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        
        return fig, axes
    
    def plot_control_effort(
        self,
        telemetry: Union[Dict[str, List[float]], pd.DataFrame],
        time_window: Optional[Tuple[float, float]] = None,
        show_saturation: bool = True,
        axes_names: List[str] = ['Az', 'El'],
        title: Optional[str] = None,
        ax: Optional[plt.Axes] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot Commanded vs Actual torque for saturation diagnosis.
        
        This plot is essential for identifying:
        - Actuator saturation events (commanded ≠ actual)
        - Anti-windup activation
        - Control authority limits
        
        Parameters
        ----------
        telemetry : dict or DataFrame
            Telemetry containing:
            - 'time'
            - 'torque_cmd_az', 'torque_cmd_el': Commanded torque [N·m]
            - 'torque_act_az', 'torque_act_el': Actual torque [N·m]
            - 'saturated_az', 'saturated_el': Saturation flags (optional)
        time_window : tuple, optional
            Time range
        show_saturation : bool
            Highlight saturation events
        axes_names : list
            Axis names for labels
        title : str, optional
            Plot title
        ax : plt.Axes or array, optional
            Existing axes
            
        Returns
        -------
        fig : plt.Figure
        ax : array of plt.Axes
        """
        # Extract data
        df = self._to_dataframe(telemetry, time_window)
        time = df['time'].values * self.time_scale
        
        # Create figure
        if ax is None:
            fig, axes = plt.subplots(2, 1, figsize=self.figure_size, sharex=True)
        else:
            fig = ax[0].figure
            axes = ax
        
        # Axis names
        axis_keys = ['az', 'el']
        
        for i, (axis_key, axis_name) in enumerate(zip(axis_keys, axes_names)):
            # Extract torques
            cmd_key = f'torque_cmd_{axis_key}' if f'torque_cmd_{axis_key}' in df.columns else f'coarse_torque_{axis_key}_cmd'
            act_key = f'torque_act_{axis_key}' if f'torque_act_{axis_key}' in df.columns else f'coarse_torque_{axis_key}'
            
            if cmd_key not in df.columns or act_key not in df.columns:
                warnings.warn(f"Torque data for {axis_name} not found, skipping")
                continue
            
            torque_cmd = df[cmd_key].values
            torque_act = df[act_key].values
            
            # Plot commanded and actual
            axes[i].plot(time, torque_cmd, 'b-', linewidth=1.5, 
                        label='Commanded', alpha=0.8)
            axes[i].plot(time, torque_act, 'r--', linewidth=1.5,
                        label='Actual', alpha=0.8)
            
            # Highlight saturation events
            if show_saturation:
                sat_key = f'saturated_{axis_key}'
                if sat_key in df.columns:
                    saturated = df[sat_key].values.astype(bool)
                else:
                    # Detect saturation by command != actual
                    saturated = np.abs(torque_cmd - torque_act) > 1e-6
                
                if np.any(saturated):
                    # Shade saturation regions
                    sat_regions = self._get_contiguous_regions(saturated)
                    for start_idx, end_idx in sat_regions:
                        axes[i].axvspan(
                            time[start_idx], time[end_idx],
                            alpha=0.3, color='red', label='Saturated' if start_idx == sat_regions[0][0] else ''
                        )
                    
                    # Calculate saturation percentage
                    sat_pct = 100 * np.sum(saturated) / len(saturated)
                    axes[i].text(
                        0.98, 0.95, f'Saturation: {sat_pct:.1f}%',
                        transform=axes[i].transAxes,
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                        fontsize=9, fontweight='bold'
                    )
            
            # Labels
            axes[i].set_ylabel(f'{axis_name} Torque (N·m)', fontsize=11, fontweight='bold')
            axes[i].legend(loc='upper left', fontsize=9)
            axes[i].grid(True, alpha=0.3)
            axes[i].axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
        
        axes[0].set_title(title or 'Control Effort: Commanded vs Actual Torque',
                         fontsize=13, fontweight='bold')
        axes[-1].set_xlabel(self.time_label, fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        
        return fig, axes
    
    def plot_fsm_tracking(
        self,
        telemetry: Union[Dict[str, List[float]], pd.DataFrame],
        time_window: Optional[Tuple[float, float]] = None,
        show_saturation_flag: bool = True,
        fsm_limit: float = 400.0,  # µrad
        title: Optional[str] = None,
        ax: Optional[plt.Axes] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot FSM commanded vs actual position for fine pointing validation.
        
        Shows:
        - FSM command tracking performance
        - Saturation events (command at limits)
        - Actuator bandwidth limitations
        
        Parameters
        ----------
        telemetry : dict or DataFrame
            Telemetry containing:
            - 'time'
            - 'fsm_cmd_alpha', 'fsm_cmd_beta': Commanded angles [µrad]
            - 'fsm_pos_alpha', 'fsm_pos_beta': Actual angles [µrad] (optional)
            - 'fsm_saturated': Saturation flag (optional)
        time_window : tuple, optional
            Time range
        show_saturation_flag : bool
            Plot saturation indicator
        fsm_limit : float
            FSM angular limit [µrad]
        title : str, optional
            Plot title
        ax : plt.Axes or array, optional
            Existing axes
            
        Returns
        -------
        fig : plt.Figure
        ax : array of plt.Axes
        """
        # Extract data
        df = self._to_dataframe(telemetry, time_window)
        time = df['time'].values * self.time_scale
        
        # Create figure
        n_panels = 3 if show_saturation_flag else 2
        if ax is None:
            fig, axes = plt.subplots(n_panels, 1, figsize=self.figure_size, sharex=True)
        else:
            fig = ax[0].figure
            axes = ax
        
        # Extract FSM data
        fsm_cmd_alpha = df['fsm_cmd_alpha'].values if 'fsm_cmd_alpha' in df.columns else df.get('fsm_cmd_az', np.zeros(len(time)))
        fsm_cmd_beta = df['fsm_cmd_beta'].values if 'fsm_cmd_beta' in df.columns else df.get('fsm_cmd_el', np.zeros(len(time)))
        
        fsm_pos_alpha = df.get('fsm_pos_alpha', fsm_cmd_alpha)  # Use command if no position feedback
        fsm_pos_beta = df.get('fsm_pos_beta', fsm_cmd_beta)
        
        # Panel 1: Alpha axis
        axes[0].plot(time, fsm_cmd_alpha, 'b-', linewidth=1.5, label='Command', alpha=0.8)
        axes[0].plot(time, fsm_pos_alpha, 'r--', linewidth=1.5, label='Actual', alpha=0.8)
        axes[0].axhline(fsm_limit, color='k', linestyle=':', linewidth=1.5, label='Limit')
        axes[0].axhline(-fsm_limit, color='k', linestyle=':', linewidth=1.5)
        
        axes[0].set_ylabel('Alpha (µrad)', fontsize=11, fontweight='bold')
        axes[0].legend(loc='upper right', fontsize=9)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title(title or 'FSM Tracking Performance',
                         fontsize=13, fontweight='bold')
        
        # Panel 2: Beta axis
        axes[1].plot(time, fsm_cmd_beta, 'b-', linewidth=1.5, label='Command', alpha=0.8)
        axes[1].plot(time, fsm_pos_beta, 'r--', linewidth=1.5, label='Actual', alpha=0.8)
        axes[1].axhline(fsm_limit, color='k', linestyle=':', linewidth=1.5, label='Limit')
        axes[1].axhline(-fsm_limit, color='k', linestyle=':', linewidth=1.5)
        
        axes[1].set_ylabel('Beta (µrad)', fontsize=11, fontweight='bold')
        axes[1].legend(loc='upper right', fontsize=9)
        axes[1].grid(True, alpha=0.3)
        
        # Panel 3: Saturation flag
        if show_saturation_flag:
            # Compute saturation (command magnitude >= limit)
            cmd_magnitude = np.sqrt(fsm_cmd_alpha**2 + fsm_cmd_beta**2)
            saturated = cmd_magnitude >= fsm_limit
            
            if 'fsm_saturated' in df.columns:
                saturated = df['fsm_saturated'].values.astype(bool)
            
            # Plot as binary signal
            axes[2].fill_between(time, 0, saturated.astype(float), 
                                alpha=0.5, color='red', label='Saturated')
            axes[2].set_ylabel('Saturation', fontsize=11, fontweight='bold')
            axes[2].set_yticks([0, 1])
            axes[2].set_yticklabels(['OK', 'SAT'])
            axes[2].grid(True, alpha=0.3)
            axes[2].set_xlabel(self.time_label, fontsize=11, fontweight='bold')
            
            # Calculate saturation percentage
            sat_pct = 100 * np.sum(saturated) / len(saturated)
            axes[2].text(
                0.02, 0.95, f'Saturation: {sat_pct:.1f}%',
                transform=axes[2].transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=9, fontweight='bold'
            )
        else:
            axes[1].set_xlabel(self.time_label, fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        
        return fig, axes
    
    def plot_full_debug_suite(
        self,
        telemetry: Union[Dict[str, List[float]], pd.DataFrame],
        time_window: Optional[Tuple[float, float]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Generate comprehensive debug plot suite with all key metrics.
        
        Creates a single figure with 4 rows:
        1. Estimator Performance (True vs Estimated)
        2. Control Effort (Commanded vs Actual Torque)
        3. FSM Tracking
        4. Overall LOS Error Timeline
        
        This is the go-to plot for post-simulation analysis.
        
        Parameters
        ----------
        telemetry : dict or DataFrame
            Complete telemetry data
        time_window : tuple, optional
            Time range
        save_path : str, optional
            If provided, save figure to this path
            
        Returns
        -------
        fig : plt.Figure
            
        Example
        -------
        >>> plotter = TimelinePlotter()
        >>> fig = plotter.plot_full_debug_suite(
        ...     telemetry,
        ...     time_window=(5, 15),
        ...     save_path='debug_suite.png'
        ... )
        """
        # Extract data
        df = self._to_dataframe(telemetry, time_window)
        time = df['time'].values * self.time_scale
        
        # Create figure with custom layout
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Row 1: Estimator Performance (2 panels side-by-side)
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, :], sharex=ax1)
        self.plot_estimator_performance(df, ax=[ax1, ax2], show_innovation=False)
        
        # Row 2: Control Effort
        ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
        ax4 = fig.add_subplot(gs[2, 1], sharex=ax1)
        self.plot_control_effort(df, ax=[ax3, ax4])
        
        # Row 3: FSM Tracking
        ax5 = fig.add_subplot(gs[3, 0], sharex=ax1)
        ax6 = fig.add_subplot(gs[3, 1], sharex=ax1)
        self.plot_fsm_tracking(df, ax=[ax5, ax6], show_saturation_flag=False)
        
        # Main title
        fig.suptitle('Digital Twin Debug Suite - Complete System Timeline',
                    fontsize=16, fontweight='bold', y=0.995)
        
        # Save if requested
        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Debug suite saved to {save_path}")
        
        return fig
    
    def plot_pointing_error_timeline(
        self,
        telemetry: Union[Dict[str, List[float]], pd.DataFrame],
        time_window: Optional[Tuple[float, float]] = None,
        show_components: bool = True,
        show_requirements: bool = True,
        rms_requirement: float = 10.0,  # µrad
        peak_requirement: float = 50.0,  # µrad
        title: Optional[str] = None,
        ax: Optional[plt.Axes] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot total pointing error magnitude over time.
        
        Shows the PRIMARY performance metric evolving over time,
        with requirement lines for pass/fail assessment.
        
        Parameters
        ----------
        telemetry : dict or DataFrame
            Telemetry data
        time_window : tuple, optional
            Time range
        show_components : bool
            Show X and Y components separately
        show_requirements : bool
            Draw requirement lines
        rms_requirement : float
            RMS error requirement [µrad]
        peak_requirement : float
            Peak error requirement [µrad]
        title : str, optional
            Plot title
        ax : plt.Axes or array, optional
            Existing axes
            
        Returns
        -------
        fig : plt.Figure
        ax : plt.Axes or array
        """
        # Extract data
        df = self._to_dataframe(telemetry, time_window)
        time = df['time'].values * self.time_scale
        
        error_x = df['los_error_x'].values
        error_y = df['los_error_y'].values
        error_mag = np.sqrt(error_x**2 + error_y**2)
        
        # Create figure
        if show_components:
            if ax is None:
                fig, axes = plt.subplots(2, 1, figsize=self.figure_size, sharex=True)
            else:
                fig = ax[0].figure
                axes = ax
            
            # Panel 1: Components
            axes[0].plot(time, error_x, 'b-', linewidth=1.5, label='X Component', alpha=0.8)
            axes[0].plot(time, error_y, 'r-', linewidth=1.5, label='Y Component', alpha=0.8)
            axes[0].set_ylabel('Error (µrad)', fontsize=11, fontweight='bold')
            axes[0].legend(loc='upper right', fontsize=9)
            axes[0].grid(True, alpha=0.3)
            axes[0].axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
            axes[0].set_title(title or 'Pointing Error Timeline',
                            fontsize=13, fontweight='bold')
            
            # Panel 2: Magnitude
            axes[1].plot(time, error_mag, 'g-', linewidth=2, label='Total Error', alpha=0.8)
            
            if show_requirements:
                axes[1].axhline(rms_requirement, color='orange', linestyle='--',
                              linewidth=2, label=f'RMS Req: {rms_requirement} µrad')
                axes[1].axhline(peak_requirement, color='red', linestyle=':',
                              linewidth=2, label=f'Peak Req: {peak_requirement} µrad')
            
            # Compute and display RMS
            rms = np.sqrt(np.mean(error_mag**2))
            axes[1].axhline(rms, color='blue', linestyle='-.', linewidth=2,
                          label=f'Actual RMS: {rms:.2f} µrad')
            
            axes[1].set_ylabel('Magnitude (µrad)', fontsize=11, fontweight='bold')
            axes[1].set_xlabel(self.time_label, fontsize=11, fontweight='bold')
            axes[1].legend(loc='upper right', fontsize=9)
            axes[1].grid(True, alpha=0.3)
            
            ax_out = axes
        else:
            if ax is None:
                fig, ax_single = plt.subplots(figsize=self.figure_size)
            else:
                fig = ax.figure
                ax_single = ax
            
            ax_single.plot(time, error_mag, 'g-', linewidth=2, label='Total Error')
            
            if show_requirements:
                ax_single.axhline(rms_requirement, color='orange', linestyle='--',
                                linewidth=2, label=f'RMS Req: {rms_requirement} µrad')
                ax_single.axhline(peak_requirement, color='red', linestyle=':',
                                linewidth=2, label=f'Peak Req: {peak_requirement} µrad')
            
            rms = np.sqrt(np.mean(error_mag**2))
            ax_single.axhline(rms, color='blue', linestyle='-.', linewidth=2,
                            label=f'Actual RMS: {rms:.2f} µrad')
            
            ax_single.set_ylabel('Pointing Error (µrad)', fontsize=11, fontweight='bold')
            ax_single.set_xlabel(self.time_label, fontsize=11, fontweight='bold')
            ax_single.set_title(title or 'Pointing Error Timeline',
                              fontsize=13, fontweight='bold')
            ax_single.legend(loc='upper right', fontsize=10)
            ax_single.grid(True, alpha=0.3)
            
            ax_out = ax_single
        
        plt.tight_layout()
        
        return fig, ax_out
    
    def _to_dataframe(
        self,
        telemetry: Union[Dict, pd.DataFrame],
        time_window: Optional[Tuple[float, float]]
    ) -> pd.DataFrame:
        """Convert telemetry to DataFrame and apply time window."""
        if isinstance(telemetry, dict):
            df = pd.DataFrame(telemetry)
        else:
            df = telemetry.copy()
        
        if time_window is not None:
            mask = (df['time'] >= time_window[0]) & (df['time'] <= time_window[1])
            df = df[mask].reset_index(drop=True)
        
        return df
    
    def _get_contiguous_regions(self, condition: np.ndarray) -> List[Tuple[int, int]]:
        """
        Find contiguous regions where condition is True.
        
        Returns list of (start_idx, end_idx) tuples.
        """
        # Find transitions
        d = np.diff(np.concatenate(([False], condition, [False])).astype(int))
        starts = np.where(d == 1)[0]
        ends = np.where(d == -1)[0]
        
        return list(zip(starts, ends))
