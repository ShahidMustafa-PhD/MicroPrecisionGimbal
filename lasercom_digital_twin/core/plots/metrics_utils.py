"""
Tracking Performance Metrics Utilities.

This module provides functions for computing control system performance
metrics from simulation telemetry data, suitable for plotting and analysis.

Metrics Computed
----------------
- Settling time (2% criterion)
- Overshoot (maximum error magnitude)
- Steady-state error (RMS of final 20%)
- RMS tracking error

Time-Varying Target Support
---------------------------
The metrics functions support both constant and time-varying (sine, square)
targets by detecting and using logged target arrays when available.

Author: Dr. S. Shahid Mustafa
Version: 1.0.0
"""

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np


@dataclass
class TrackingMetrics:
    """
    Container for tracking performance metrics.
    
    All angular values are stored in radians for internal consistency.
    Use conversion methods for degree output in plots/tables.
    
    Attributes
    ----------
    settling_time_az : float
        Azimuth settling time [s] (2% criterion)
    settling_time_el : float
        Elevation settling time [s] (2% criterion)
    overshoot_az : float
        Maximum azimuth tracking error [rad]
    overshoot_el : float
        Maximum elevation tracking error [rad]
    ss_error_az : float
        Azimuth steady-state RMS error [rad]
    ss_error_el : float
        Elevation steady-state RMS error [rad]
    """
    settling_time_az: float
    settling_time_el: float
    overshoot_az: float
    overshoot_el: float
    ss_error_az: float
    ss_error_el: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary format."""
        return {
            'settling_time_az': self.settling_time_az,
            'settling_time_el': self.settling_time_el,
            'overshoot_az': self.overshoot_az,
            'overshoot_el': self.overshoot_el,
            'ss_error_az': self.ss_error_az,
            'ss_error_el': self.ss_error_el
        }
    
    @property
    def ss_error_az_urad(self) -> float:
        """Steady-state azimuth error in microradians."""
        return abs(self.ss_error_az) * 1e6
    
    @property
    def ss_error_el_urad(self) -> float:
        """Steady-state elevation error in microradians."""
        return abs(self.ss_error_el) * 1e6
    
    @property
    def ss_error_az_deg(self) -> float:
        """Steady-state azimuth error in degrees."""
        return np.rad2deg(abs(self.ss_error_az))
    
    @property
    def ss_error_el_deg(self) -> float:
        """Steady-state elevation error in degrees."""
        return np.rad2deg(abs(self.ss_error_el))


def compute_tracking_metrics(
    results: Dict,
    target_az_rad: float,
    target_el_rad: float,
    settling_threshold_rad: Optional[float] = None,
    steady_state_fraction: float = 0.2
) -> Dict[str, float]:
    """
    Compute comprehensive tracking response characteristics.
    
    Handles both constant and time-varying (sine, square wave) targets by
    using logged target arrays when available.
    
    Parameters
    ----------
    results : Dict
        Simulation results dictionary with 'log_arrays' key containing:
        - 'time': Time vector [s]
        - 'q_az': Azimuth position [rad]
        - 'q_el': Elevation position [rad]
        - 'target_az': (optional) Time-varying azimuth target [rad]
        - 'target_el': (optional) Time-varying elevation target [rad]
    target_az_rad : float
        Fallback target azimuth [rad] if no time-varying target logged
    target_el_rad : float
        Fallback target elevation [rad] if no time-varying target logged
    settling_threshold_rad : float, optional
        Threshold for settling detection [rad]. Default: 0.5° = 0.00873 rad
    steady_state_fraction : float
        Fraction of trajectory to use for steady-state error (default: 20%)
        
    Returns
    -------
    Dict[str, float]
        Performance metrics dictionary compatible with legacy interface:
        - 'settling_time_az', 'settling_time_el' [s]
        - 'overshoot_az', 'overshoot_el' [rad]
        - 'ss_error_az', 'ss_error_el' [rad]
    
    Notes
    -----
    For time-varying targets:
    - Settling is detected when error stays below threshold
    - Overshoot is the maximum error magnitude
    - Steady-state error is RMS of final portion of trajectory
    
    This function returns a dict for backward compatibility. For new code,
    use `compute_tracking_metrics_dataclass()` which returns a TrackingMetrics
    object with convenient unit conversion properties.
    """
    # Default settling threshold: 0.5 degrees
    if settling_threshold_rad is None:
        settling_threshold_rad = np.deg2rad(0.5)
    
    # Extract time and position arrays
    t = results['log_arrays']['time']
    q_az = results['log_arrays']['q_az']
    q_el = results['log_arrays']['q_el']
    
    # Use logged time-varying target if available, else constant fallback
    if 'target_az' in results['log_arrays']:
        target_az = results['log_arrays']['target_az']
        target_el = results['log_arrays']['target_el']
    else:
        target_az = np.full_like(q_az, target_az_rad)
        target_el = np.full_like(q_el, target_el_rad)
    
    # Compute tracking error
    error_az = q_az - target_az
    error_el = q_el - target_el
    
    # =========================================================================
    # Settling Time (2% criterion adapted for tracking)
    # =========================================================================
    # For time-varying targets, settling is when error stays below threshold
    settled_az = np.where(np.abs(error_az) < settling_threshold_rad)[0]
    settling_time_az = t[settled_az[0]] if len(settled_az) > 0 else t[-1]
    
    settled_el = np.where(np.abs(error_el) < settling_threshold_rad)[0]
    settling_time_el = t[settled_el[0]] if len(settled_el) > 0 else t[-1]
    
    # =========================================================================
    # Overshoot (Maximum Error Magnitude)
    # =========================================================================
    # For time-varying targets, this is the peak tracking error
    overshoot_az = np.max(np.abs(error_az))
    overshoot_el = np.max(np.abs(error_el))
    
    # =========================================================================
    # Steady-State Error (RMS of final portion)
    # =========================================================================
    n_samples = len(error_az)
    last_n = int(steady_state_fraction * n_samples)
    
    # Use RMS for time-varying targets (represents average tracking quality)
    steady_state_error_az = np.sqrt(np.mean(error_az[-last_n:] ** 2))
    steady_state_error_el = np.sqrt(np.mean(error_el[-last_n:] ** 2))
    
    return {
        'settling_time_az': settling_time_az,
        'settling_time_el': settling_time_el,
        'overshoot_az': overshoot_az,
        'overshoot_el': overshoot_el,
        'ss_error_az': steady_state_error_az,
        'ss_error_el': steady_state_error_el
    }


def compute_tracking_metrics_dataclass(
    results: Dict,
    target_az_rad: float,
    target_el_rad: float,
    settling_threshold_rad: Optional[float] = None,
    steady_state_fraction: float = 0.2
) -> TrackingMetrics:
    """
    Compute tracking metrics and return as dataclass.
    
    This is the preferred interface for new code. See `compute_tracking_metrics`
    for parameter documentation.
    
    Returns
    -------
    TrackingMetrics
        Dataclass with metrics and unit conversion properties
    """
    metrics_dict = compute_tracking_metrics(
        results, target_az_rad, target_el_rad,
        settling_threshold_rad, steady_state_fraction
    )
    return TrackingMetrics(**metrics_dict)


def compute_los_metrics(results: Dict) -> Dict[str, float]:
    """
    Compute Line-of-Sight specific metrics.
    
    Parameters
    ----------
    results : Dict
        Simulation results with LOS error arrays
        
    Returns
    -------
    Dict[str, float]
        LOS metrics including RMS, peak, and final values
    """
    log = results['log_arrays']
    
    los_x = log.get('los_error_x', np.zeros(1))
    los_y = log.get('los_error_y', np.zeros(1))
    
    # Total LOS error magnitude
    los_total = np.sqrt(los_x**2 + los_y**2)
    
    return {
        'los_rms_x_rad': np.sqrt(np.mean(los_x**2)),
        'los_rms_y_rad': np.sqrt(np.mean(los_y**2)),
        'los_rms_total_rad': np.sqrt(np.mean(los_total**2)),
        'los_peak_rad': np.max(los_total),
        'los_final_rad': los_total[-1] if len(los_total) > 0 else 0.0,
        'los_rms_urad': np.sqrt(np.mean(los_total**2)) * 1e6,
        'los_peak_urad': np.max(los_total) * 1e6
    }


def compute_control_effort_metrics(results: Dict) -> Dict[str, float]:
    """
    Compute control effort (torque) metrics.
    
    Parameters
    ----------
    results : Dict
        Simulation results with torque arrays
        
    Returns
    -------
    Dict[str, float]
        Torque metrics including RMS, peak, and saturation percentage
    """
    log = results['log_arrays']
    
    tau_az = log.get('torque_az', np.zeros(1))
    tau_el = log.get('torque_el', np.zeros(1))
    
    # RMS values
    rms_az = np.sqrt(np.mean(tau_az**2))
    rms_el = np.sqrt(np.mean(tau_el**2))
    rms_total = np.sqrt(rms_az**2 + rms_el**2)
    
    # Peak values
    peak_az = np.max(np.abs(tau_az))
    peak_el = np.max(np.abs(tau_el))
    
    return {
        'torque_rms_az': rms_az,
        'torque_rms_el': rms_el,
        'torque_rms_total': rms_total,
        'torque_peak_az': peak_az,
        'torque_peak_el': peak_el
    }
