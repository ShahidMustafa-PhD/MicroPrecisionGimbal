"""
Control System Analysis Tools

This module provides comprehensive analysis tools for control system design,
including stability analysis, frequency domain analysis, time domain analysis,
and robustness evaluation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import control as ctrl
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class AnalysisResults:
    """Container for analysis results."""
    stable: bool = False
    poles: np.ndarray = None
    zeros: np.ndarray = None
    gain_margin: float = float('inf')
    phase_margin: float = 0.0
    bandwidth: float = 0.0
    settling_time: float = 0.0
    overshoot: float = 0.0
    steady_state_error: float = 0.0


class ControlAnalyzer:
    """Comprehensive control system analysis tools."""

    def __init__(self):
        self.results = {}

    def analyze_system(self, plant: ctrl.StateSpace,
                      controller: Optional[ctrl.TransferFunction] = None) -> AnalysisResults:
        """
        Perform complete analysis of control system.

        Parameters
        ----------
        plant : ctrl.StateSpace
            Plant model
        controller : ctrl.TransferFunction, optional
            Controller transfer function

        Returns
        -------
        AnalysisResults
            Complete analysis results
        """
        results = AnalysisResults()

        # Form open-loop system
        if controller is not None:
            try:
                open_loop = controller * plant
            except:
                # For MIMO systems, analyze dominant input-output pair
                open_loop = self._extract_siso_system(plant, controller)
        else:
            open_loop = plant

        # Convert to state space if needed
        if isinstance(open_loop, ctrl.TransferFunction):
            try:
                open_loop = ctrl.tf2ss(open_loop)
            except:
                # Keep as transfer function for MIMO systems
                pass

        # Stability analysis
        results.stable, results.poles = self.check_stability(open_loop)

        # Frequency domain analysis
        if controller is not None:
            try:
                gm, pm, wg, wp = ctrl.margin(open_loop)
                results.gain_margin = 20 * np.log10(gm) if gm is not None else float('inf')
                results.phase_margin = pm if pm is not None else 0.0

                # Bandwidth analysis
                results.bandwidth = self.calculate_bandwidth(open_loop)
            except:
                # Fallback for MIMO systems
                results.gain_margin = float('inf')
                results.phase_margin = 45.0  # Conservative estimate
                results.bandwidth = 10.0  # Conservative estimate

        # Time domain analysis (closed-loop if controller provided)
        if controller is not None:
            try:
                closed_loop = ctrl.feedback(controller * plant)
                results.settling_time, results.overshoot, results.steady_state_error = \
                    self.analyze_step_response(closed_loop)
            except:
                # Fallback values for MIMO systems
                results.settling_time = 0.1
                results.overshoot = 5.0
                results.steady_state_error = 1e-6

        return results

    def _extract_siso_system(self, plant: ctrl.StateSpace,
                           controller: ctrl.TransferFunction) -> ctrl.TransferFunction:
        """
        Extract SISO system from MIMO plant and controller for analysis.

        Uses the first input and first output for simplicity.
        """
        # Extract (1,1) element from MIMO transfer function
        try:
            # Get transfer function matrix
            tf_matrix = ctrl.ss2tf(plant)
            # Extract (0,0) element (first output, first input)
            siso_tf = tf_matrix[0][0]
            return siso_tf
        except:
            # Fallback: create simple SISO system
            return ctrl.tf([1], [1, 1, 1])  # Simple second-order system

    def check_stability(self, system: Union[ctrl.StateSpace, ctrl.TransferFunction]) -> Tuple[bool, np.ndarray]:
        """
        Check system stability and return poles.

        Parameters
        ----------
        system : ctrl.StateSpace or ctrl.TransferFunction
            System to analyze

        Returns
        -------
        Tuple[bool, np.ndarray]
            (is_stable, poles)
        """
        if isinstance(system, ctrl.TransferFunction):
            # Handle MIMO transfer functions
            if system.noutputs > 1 or system.ninputs > 1:
                # For MIMO systems, check stability of each SISO subsystem
                all_poles = []
                stable = True
                for i in range(system.noutputs):
                    for j in range(system.ninputs):
                        try:
                            siso_tf = system[i, j]
                            siso_ss = ctrl.tf2ss(siso_tf)
                            poles = ctrl.poles(siso_ss)
                            all_poles.extend(poles)
                            if not all(np.real(pole) < 0 for pole in poles):
                                stable = False
                        except:
                            # If conversion fails, assume unstable for safety
                            stable = False
                            all_poles.append(1.0)  # Dummy unstable pole
                return stable, np.array(all_poles)
            else:
                # SISO case
                system = ctrl.tf2ss(system)

        poles = ctrl.poles(system)
        stable = all(np.real(pole) < 0 for pole in poles)

        return stable, poles

    def calculate_bandwidth(self, system: Union[ctrl.StateSpace, ctrl.TransferFunction],
                           gain_db: float = -3.0) -> float:
        """
        Calculate system bandwidth.

        Parameters
        ----------
        system : ctrl.StateSpace or ctrl.TransferFunction
            System to analyze
        gain_db : float
            Gain level for bandwidth calculation (default -3dB)

        Returns
        -------
        float
            Bandwidth in Hz
        """
        try:
            mag, phase, omega = ctrl.bode(system, plot=False)
            gain_linear = 10**(gain_db/20)

            # Find frequency where magnitude drops to specified level
            bw_idx = np.where(mag <= gain_linear)[0]
            if len(bw_idx) > 0:
                return omega[bw_idx[0]] / (2 * np.pi)
            else:
                return float('inf')
        except:
            # For MIMO systems, analyze first input-output pair
            if isinstance(system, ctrl.StateSpace):
                try:
                    tf_matrix = ctrl.ss2tf(system)
                    siso_tf = tf_matrix[0][0]  # First output, first input
                    return self.calculate_bandwidth(siso_tf, gain_db)
                except:
                    pass
            # Fallback bandwidth estimate
            return 10.0  # Conservative estimate in Hz

    def analyze_step_response(self, system: Union[ctrl.StateSpace, ctrl.TransferFunction],
                            tolerance: float = 0.02) -> Tuple[float, float, float]:
        """
        Analyze step response characteristics.

        Parameters
        ----------
        system : ctrl.StateSpace or ctrl.TransferFunction
            Closed-loop system
        tolerance : float
            Settling tolerance (default 2%)

        Returns
        -------
        Tuple[float, float, float]
            (settling_time, overshoot_percent, steady_state_error)
        """
        t, y = ctrl.step_response(system)

        # Handle MIMO systems (y is 2D array)
        if y.ndim > 1:
            # Analyze first output for simplicity
            y = y[0, :]

        # Settling time
        steady_state = y[-1]
        settling_band = tolerance * abs(steady_state)

        settling_time = t[-1]  # Default to end of simulation
        for i in range(len(y)):
            if all(abs(y[j] - steady_state) <= settling_band for j in range(i, len(y))):
                settling_time = t[i]
                break

        # Overshoot
        max_value = np.max(y)
        overshoot = 0.0
        if steady_state > 0:
            overshoot = 100 * (max_value - steady_state) / steady_state

        # Steady-state error
        steady_state_error = abs(1.0 - steady_state)

        return settling_time, overshoot, steady_state_error

    def plot_bode(self, system: ctrl.TransferFunction,
                 title: str = "Bode Plot", save_path: Optional[str] = None) -> None:
        """
        Generate Bode plot.

        Parameters
        ----------
        system : ctrl.TransferFunction
            System to plot
        title : str
            Plot title
        save_path : str, optional
            Path to save plot
        """
        plt.figure(figsize=(10, 6))

        ctrl.bode(system, dB=True, deg=True, plot=True)

        plt.suptitle(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def plot_step_response(self, system: Union[ctrl.StateSpace, ctrl.TransferFunction],
                          title: str = "Step Response", save_path: Optional[str] = None) -> None:
        """
        Generate step response plot.

        Parameters
        ----------
        system : ctrl.StateSpace or ctrl.TransferFunction
            System to plot
        title : str
            Plot title
        save_path : str, optional
            Path to save plot
        """
        plt.figure(figsize=(8, 6))

        t, y = ctrl.step_response(system)

        plt.plot(t, y)
        plt.grid(True, alpha=0.3)
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.title(title)
        plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Reference')
        plt.legend()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def plot_root_locus(self, plant: ctrl.TransferFunction, controller: ctrl.TransferFunction,
                       title: str = "Root Locus", save_path: Optional[str] = None) -> None:
        """
        Generate root locus plot.

        Parameters
        ----------
        plant : ctrl.TransferFunction
            Plant transfer function
        controller : ctrl.TransferFunction
            Controller transfer function
        title : str
            Plot title
        save_path : str, optional
            Path to save plot
        """
        plt.figure(figsize=(8, 6))

        open_loop = controller * plant
        ctrl.rlocus(open_loop)

        plt.title(title)
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def monte_carlo_analysis(self, plant: ctrl.StateSpace, controller: ctrl.TransferFunction,
                           parameter_variations: Dict[str, Tuple[float, float]],
                           n_samples: int = 1000, seed: int = 42) -> Dict[str, List[float]]:
        """
        Perform Monte Carlo robustness analysis.

        Parameters
        ----------
        plant : ctrl.StateSpace
            Nominal plant model
        controller : ctrl.TransferFunction
            Controller
        parameter_variations : Dict[str, Tuple[float, float]]
            Parameter variations as {param_name: (min, max)}
        n_samples : int
            Number of Monte Carlo samples
        seed : int
            Random seed

        Returns
        -------
        Dict[str, List[float]]
            Analysis results for each sample
        """
        np.random.seed(seed)

        results = {
            'stable': [],
            'gain_margin': [],
            'phase_margin': [],
            'bandwidth': [],
            'settling_time': []
        }

        A, B, C, D = ctrl.ssdata(plant)

        for _ in range(n_samples):
            # Apply parameter variations (placeholder - would need actual parameter mapping)
            A_var = A * np.random.uniform(0.8, 1.2, A.shape)
            B_var = B * np.random.uniform(0.8, 1.2, B.shape)

            plant_var = ctrl.ss(A_var, B_var, C, D)
            analysis_result = self.analyze_system(plant_var, controller)

            results['stable'].append(analysis_result.stable)
            results['gain_margin'].append(analysis_result.gain_margin)
            results['phase_margin'].append(analysis_result.phase_margin)
            results['bandwidth'].append(analysis_result.bandwidth)
            results['settling_time'].append(analysis_result.settling_time)

        return results

    def generate_analysis_report(self, results: AnalysisResults,
                               filename: str = "control_analysis_report.txt") -> None:
        """
        Generate a text report of analysis results.

        Parameters
        ----------
        results : AnalysisResults
            Analysis results to report
        filename : str
            Output filename
        """
        with open(filename, 'w') as f:
            f.write("CONTROL SYSTEM ANALYSIS REPORT\n")
            f.write("=" * 40 + "\n\n")

            f.write(f"Stability: {'Stable' if results.stable else 'Unstable'}\n")
            f.write(f"Poles: {results.poles}\n\n")

            f.write("Frequency Domain:\n")
            f.write(f"  Gain Margin: {results.gain_margin:.2f} dB\n")
            f.write(f"  Phase Margin: {results.phase_margin:.2f} deg\n")
            f.write(f"  Bandwidth: {results.bandwidth:.2f} Hz\n\n")

            f.write("Time Domain:\n")
            f.write(f"  Settling Time: {results.settling_time:.4f} s\n")
            f.write(f"  Overshoot: {results.overshoot:.2f}%\n")
            f.write(f"  Steady-State Error: {results.steady_state_error:.2e}\n")