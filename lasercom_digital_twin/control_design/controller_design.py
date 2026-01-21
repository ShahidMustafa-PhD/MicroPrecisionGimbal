"""
Controller Design and Synthesis Tools

This module implements controller synthesis algorithms for the lasercom
pointing system, including PID controllers, Linear Quadratic Gaussian (LQG)
controllers, and H-infinity controllers.

The design process follows a systematic approach:
1. Plant model identification/linearization
2. Controller requirements specification
3. Controller synthesis and tuning
4. Stability and performance analysis
5. Robustness evaluation

Integration with GimbalDynamics Linearization:
---------------------------------------------
The module provides utilities to obtain a linear state-space model from the
nonlinear GimbalDynamics class via numerical linearization, then synthesize
controllers based on frequency-domain or state-space methods.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List, Union
from dataclasses import dataclass, field
import control as ctrl
from abc import ABC, abstractmethod
import sys
from pathlib import Path

# Import gimbal dynamics for linearization
# Add parent directories to path for proper imports
_current_dir = Path(__file__).parent
_project_root = _current_dir.parent.parent
sys.path.insert(0, str(_project_root))

from lasercom_digital_twin.core.dynamics.gimbal_dynamics import GimbalDynamics


@dataclass
class ControllerGains:
    """Container for controller gain parameters."""
    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0
    kf: float = 0.0  # Feedforward gain


@dataclass
class ControllerSpecs:
    """Controller performance specifications."""
    bandwidth_hz: float = 10.0          # Closed-loop bandwidth [Hz]
    phase_margin_deg: float = 45.0      # Phase margin [deg]
    gain_margin_db: float = 6.0         # Gain margin [dB]
    settling_time_sec: float = 0.1      # Settling time [s]
    overshoot_percent: float = 5.0      # Maximum overshoot [%]
    steady_state_error: float = 1e-6    # Steady-state error tolerance


class BaseController(ABC):
    """Abstract base class for all controllers."""

    def __init__(self, specs: ControllerSpecs):
        self.specs = specs
        self.gains = ControllerGains()
        self.is_tuned = False

    @abstractmethod
    def design(self, plant_model: ctrl.StateSpace) -> None:
        """Design controller for given plant model."""
        pass

    @abstractmethod
    def get_transfer_function(self) -> ctrl.TransferFunction:
        """Return controller transfer function."""
        pass

    def validate_design(self, plant_model: ctrl.StateSpace) -> Dict[str, float]:
        """Validate controller design against specifications."""
        # Form closed-loop system
        controller_tf = self.get_transfer_function()
        closed_loop = ctrl.feedback(controller_tf * plant_model)

        # Analyze performance metrics
        results = {}

        # Bandwidth
        mag, phase, omega = ctrl.bode(closed_loop, plot=False)
        bw_idx = np.where(mag <= 0.707)[0]
        if len(bw_idx) > 0:
            results['bandwidth_hz'] = omega[bw_idx[0]] / (2 * np.pi)
        else:
            results['bandwidth_hz'] = float('inf')

        # Stability margins
        gm, pm, wg, wp = ctrl.margin(controller_tf * plant_model)
        results['gain_margin_db'] = 20 * np.log10(gm) if gm is not None else float('inf')
        results['phase_margin_deg'] = pm if pm is not None else 0.0

        # Step response characteristics
        t, y = ctrl.step_response(closed_loop)
        results['settling_time'] = self._calculate_settling_time(t, y)
        results['overshoot'] = self._calculate_overshoot(y)
        results['steady_state_error'] = abs(1.0 - y[-1])

        return results

    def _calculate_settling_time(self, t: np.ndarray, y: np.ndarray, tolerance: float = 0.02) -> float:
        """Calculate settling time for step response."""
        steady_state = y[-1]
        settling_band = tolerance * abs(steady_state)

        for i in range(len(y)):
            if all(abs(y[j] - steady_state) <= settling_band for j in range(i, len(y))):
                return t[i]
        return t[-1]

    def _calculate_overshoot(self, y: np.ndarray) -> float:
        """Calculate percentage overshoot."""
        steady_state = y[-1]
        max_value = np.max(y)
        if steady_state > 0:
            return 100 * (max_value - steady_state) / steady_state
        return 0.0


class PIDController(BaseController):
    """PID Controller with systematic tuning methods."""

    def __init__(self, specs: ControllerSpecs, tuning_method: str = "ziegler_nichols"):
        super().__init__(specs)
        self.tuning_method = tuning_method

    def design(self, plant_model: ctrl.StateSpace) -> None:
        """Design PID controller using specified tuning method."""
        if self.tuning_method == "ziegler_nichols":
            self._ziegler_nichols_tuning(plant_model)
        elif self.tuning_method == "manual":
            self._manual_tuning()
        else:
            raise ValueError(f"Unknown tuning method: {self.tuning_method}")

        self.is_tuned = True

    def _ziegler_nichols_tuning(self, plant_model: ctrl.StateSpace) -> None:
        """Ziegler-Nichols tuning for PID controller."""
        # Convert to transfer function for analysis
        plant_tf = ctrl.ss2tf(plant_model)

        # Find ultimate gain and period (simplified approach)
        # In practice, this would involve finding the stability boundary
        Ku = 1.0  # Ultimate gain (placeholder)
        Tu = 1.0  # Ultimate period (placeholder)

        # Ziegler-Nichols rules for PID
        self.gains.kp = 0.6 * Ku
        self.gains.ki = 1.2 * Ku / Tu
        self.gains.kd = 0.075 * Ku * Tu

    def _manual_tuning(self) -> None:
        """Manual tuning based on specifications."""
        # Placeholder for manual tuning logic
        self.gains.kp = 100.0
        self.gains.ki = 50.0
        self.gains.kd = 2.0

    def get_transfer_function(self) -> ctrl.TransferFunction:
        """Return PID controller transfer function."""
        s = ctrl.tf('s')
        return self.gains.kp + self.gains.ki/s + self.gains.kd*s


class LQGController(BaseController):
    """Linear Quadratic Gaussian (LQG) Controller."""

    def __init__(self, specs: ControllerSpecs, Q: np.ndarray = None, R: np.ndarray = None,
                 V: np.ndarray = None, W: np.ndarray = None):
        super().__init__(specs)
        self.Q = Q  # State weighting matrix
        self.R = R  # Control weighting matrix
        self.V = V  # Measurement noise covariance
        self.W = W  # Process noise covariance
        self.K = None  # Controller gain matrix
        self.L = None  # Kalman filter gain matrix

    def design(self, plant_model: ctrl.StateSpace) -> None:
        """Design LQG controller."""
        # Extract system matrices
        A, B, C, D = ctrl.ssdata(plant_model)

        # Default weighting matrices if not provided
        n_states = A.shape[0]
        n_outputs = C.shape[0]
        n_inputs = B.shape[1]

        if self.Q is None:
            self.Q = np.eye(n_states)
        if self.R is None:
            self.R = np.eye(n_inputs)
        if self.V is None:
            self.V = 0.01 * np.eye(n_outputs)
        if self.W is None:
            self.W = 0.01 * np.eye(n_states)

        # Design Kalman filter
        self.L, P, E = ctrl.lqe(A, np.eye(n_states), C, self.W, self.V)

        # Design LQR controller
        self.K, S, E = ctrl.lqr(A, B, self.Q, self.R)

        self.is_tuned = True

    def get_transfer_function(self) -> ctrl.TransferFunction:
        """Return LQG controller transfer function."""
        # For simplicity, return the LQR portion
        # Full LQG would require state-space representation
        return ctrl.ss([], [], [], self.K)


class ControllerDesigner:
    """Main controller design interface."""

    def __init__(self):
        self.controllers = {}

    def design_pid_controller(self, plant_model: ctrl.StateSpace,
                            specs: ControllerSpecs,
                            tuning_method: str = "ziegler_nichols") -> PIDController:
        """Design a PID controller."""
        controller = PIDController(specs, tuning_method)
        controller.design(plant_model)
        self.controllers['pid'] = controller
        return controller

    def design_lqg_controller(self, plant_model: ctrl.StateSpace,
                            specs: ControllerSpecs,
                            Q: np.ndarray = None, R: np.ndarray = None,
                            V: np.ndarray = None, W: np.ndarray = None) -> LQGController:
        """Design an LQG controller."""
        controller = LQGController(specs, Q, R, V, W)
        controller.design(plant_model)
        self.controllers['lqg'] = controller
        return controller

    def get_controller(self, name: str) -> Optional[BaseController]:
        """Retrieve a designed controller."""
        return self.controllers.get(name)

    def compare_controllers(self, plant_model: ctrl.StateSpace,
                          controllers: List[BaseController]) -> Dict[str, Dict]:
        """Compare multiple controllers on the same plant."""
        results = {}
        for i, controller in enumerate(controllers):
            name = f"controller_{i}"
            results[name] = controller.validate_design(plant_model)
        return results
    
    def design_coarse_pid_from_specs(
        self,
        gimbal: GimbalDynamics,
        q_op: np.ndarray = None,
        dq_op: np.ndarray = None,
        bandwidth_hz: float = 5.0,
        phase_margin_deg: float = 60.0,
        use_loop_shaping: bool = True,
        derivative_filter_N: float = 15.0,
        generate_plots: bool = True
    ) -> Dict[str, Union[ControllerGains, Dict]]:
        """
        Synthesize PID controller for Coarse Gimbal with comprehensive stability analysis.
        
        This method implements a complete frequency-domain controller design workflow:
        1. Linearize gimbal dynamics at operating point → State-Space (A, B, C, D)
        2. Extract SISO transfer functions for Pan and Tilt axes
        3. Design PID compensator C(s) to meet bandwidth and phase margin specs
        4. Generate four diagnostic plots per axis:
           - Time Domain Response (Step Response)
           - Root Locus (Uncompensated Plant)
           - Root Locus (Compensated Closed-Loop)
           - Frequency Response (Bode with Margins)
        
        Mathematical Framework:
        ----------------------
        **Plant Model (Linearized):**
        $$\\dot{x} = Ax + Bu, \\quad y = Cx + Du$$
        
        **PID Compensator with Filtered Derivative:**
        $$C(s) = K_p + \\frac{K_i}{s} + \\frac{K_d s}{1 + s/(N \\omega_c)}$$
        
        where $N$ is the derivative filter ratio (typically 10-20).
        
        **Open-Loop Transfer Function:**
        $$L(s) = C(s) G(s)$$
        
        **Closed-Loop Transfer Function:**
        $$T(s) = \\frac{L(s)}{1 + L(s)}$$
        
        **Loop Shaping Criteria:**
        - Crossover frequency: $\\omega_c = 2\\pi f_{BW}$
        - Phase margin: $PM = 180° + \\angle L(j\\omega_c) \\geq 60°$
        - Gain margin: $GM = 1/|L(j\\omega_{180})|$ (in dB) $\\geq 6$ dB
        
        Args:
            gimbal: Nonlinear gimbal dynamics model
            q_op: Operating point positions [rad] (default: [0, 0])
            dq_op: Operating point velocities [rad/s] (default: [0, 0])
            bandwidth_hz: Target closed-loop bandwidth [Hz]
            phase_margin_deg: Target phase margin [deg]
            use_loop_shaping: Use analytical loop shaping (True) vs numerical optimization
            derivative_filter_N: Derivative filter ratio (N = f_filter / f_bandwidth)
            generate_plots: Generate diagnostic plots (default: True)
        
        Returns:
            Dict containing:
                - 'gains_pan': ControllerGains for pan axis
                - 'gains_tilt': ControllerGains for tilt axis
                - 'plant_ss': Linearized state-space model
                - 'analysis_pan': Stability margins and performance metrics
                - 'analysis_tilt': Stability margins and performance metrics
                - 'plots_generated': Boolean indicating if plots were created
        
        Example:
            >>> designer = ControllerDesigner()
            >>> gimbal = GimbalDynamics(pan_mass=0.5, tilt_mass=0.25)
            >>> result = designer.design_coarse_pid_from_specs(
            ...     gimbal, bandwidth_hz=5.0, phase_margin_deg=60.0
            ... )
            >>> print(f"Pan Kp: {result['gains_pan'].kp:.3f}")
        """
        # Default operating point: upright position at rest
        if q_op is None:
            q_op = np.array([0.0, 0.0])
        if dq_op is None:
            dq_op = np.array([0.0, 0.0])
        
        print("=" * 80)
        print("AUTOMATED GIMBAL PID SYNTHESIS & MULTI-VIEW STABILITY ANALYSIS")
        print("=" * 80)
        print(f"Operating Point: q = {q_op}, dq = {dq_op}")
        print(f"Target Bandwidth: {bandwidth_hz:.1f} Hz")
        print(f"Target Phase Margin: {phase_margin_deg:.1f}°")
        print(f"Derivative Filter N: {derivative_filter_N:.1f}")
        print()
        
        # ====================================================================
        # STEP 1: LINEARIZATION
        # ====================================================================
        print("Step 1: Linearizing gimbal dynamics...")
        A, B, C, D = gimbal.linearize(q_op, dq_op)
        plant_ss = ctrl.StateSpace(A, B, C, D)
        plant_ss.state_labels = ['q_pan', 'q_tilt', 'dq_pan', 'dq_tilt']
        plant_ss.input_labels = ['tau_pan', 'tau_tilt']
        plant_ss.output_labels = ['y_pan', 'y_tilt']
        print(f"  State-space model: {plant_ss.nstates} states, {plant_ss.ninputs} inputs, {plant_ss.noutputs} outputs")
        print()
        
        # ====================================================================
        # STEP 2: SISO EXTRACTION
        # ====================================================================
        print("Step 2: Extracting SISO transfer functions...")
        tf_matrix = ctrl.ss2tf(plant_ss)
        plant_pan = tf_matrix[0, 0]   # q_pan / tau_pan
        plant_tilt = tf_matrix[1, 1]  # q_tilt / tau_tilt
        print(f"  Pan plant G_pan(s):  {plant_pan}")
        print(f"  Tilt plant G_tilt(s): {plant_tilt}")
        print()
        
        # ====================================================================
        # STEP 3: PID COMPENSATOR DESIGN
        # ====================================================================
        print("Step 3: Synthesizing PID compensators via loop shaping...")
        omega_c = 2 * np.pi * bandwidth_hz  # Crossover frequency [rad/s]
        omega_filter = derivative_filter_N * omega_c  # Derivative filter cutoff
        
        def design_pid_siso(plant_siso, axis_name: str) -> Tuple[ControllerGains, Dict]:
            """Design PID for single axis using loop shaping."""
            print(f"\n  {axis_name} Axis:")
            
            # Evaluate plant at crossover frequency
            mag_at_wc, phase_at_wc, _ = ctrl.bode(plant_siso, [omega_c], plot=False)
            mag_at_wc = mag_at_wc[0]
            phase_at_wc = phase_at_wc[0]
            
            # Proportional gain: |Kp * G(jωc)| = 1 (0 dB crossover)
            Kp = 1.0 / mag_at_wc
            
            # Integral gain: Place integral corner at ωc/10
            Ti = 10.0 / omega_c
            Ki = Kp / Ti
            
            # Derivative gain: Add phase lead, accounting for filter
            # Unfiltered derivative adds +90° at high freq
            # Filtered derivative: Td*s / (1 + Td*s/N) peaks at sqrt(N)*wc
            Td = 1.0 / (derivative_filter_N * omega_c)
            Kd = Kp * Td
            
            # Phase contribution at crossover
            phase_integral = np.rad2deg(-np.arctan(omega_c * Ti))
            phase_derivative = np.rad2deg(np.arctan(derivative_filter_N * omega_c * Td) - np.arctan(omega_c * Td / derivative_filter_N))
            
            print(f"    Kp = {Kp:.6f}")
            print(f"    Ki = {Ki:.6f} (Ti = {Ti:.4f} s, corner @ {1/(2*np.pi*Ti):.2f} Hz)")
            print(f"    Kd = {Kd:.6f} (Td = {Td:.6f} s, N = {derivative_filter_N:.1f})")
            print(f"    Phase contribution at wc: I={phase_integral:.1f}deg, D={phase_derivative:.1f}deg")
            
            gains = ControllerGains(kp=Kp, ki=Ki, kd=Kd)
            
            # Construct PID transfer function with filtered derivative
            s = ctrl.tf('s')
            C_pid = Kp + Ki/s + (Kd * s) / (1 + s/omega_filter)
            
            return gains, C_pid
        
        gains_pan, C_pan = design_pid_siso(plant_pan, "Pan")
        gains_tilt, C_tilt = design_pid_siso(plant_tilt, "Tilt")
        print()
        
        # ====================================================================
        # STEP 4: FEEDBACK ANALYSIS
        # ====================================================================
        print("Step 4: Analyzing closed-loop stability and performance...")
        
        def analyze_closed_loop(plant_siso, C_siso, gains, axis_name: str) -> Dict:
            """Analyze single-axis closed loop."""
            # Open-loop transfer function
            L = C_siso * plant_siso
            
            # Closed-loop transfer function
            T = ctrl.feedback(L, 1)
            
            # Stability margins
            gm, pm, wgc, wpc = ctrl.margin(L)
            
            # Bandwidth (frequency where |T| = -3dB)
            omega_bode = np.logspace(-2, 3, 500)
            mag_cl, _, omega_cl = ctrl.bode(T, omega_bode, plot=False)
            bw_idx = np.where(20*np.log10(mag_cl) <= -3.0)[0]
            bw_achieved = omega_cl[bw_idx[0]] / (2*np.pi) if len(bw_idx) > 0 else None
            
            # Step response characteristics
            t_step = np.linspace(0, 2.0/bandwidth_hz, 500)
            t_step, y_step = ctrl.step_response(T, t_step)
            
            # Rise time (10% to 90%)
            y_10 = 0.1 * y_step[-1]
            y_90 = 0.9 * y_step[-1]
            idx_10 = np.where(y_step >= y_10)[0][0] if np.any(y_step >= y_10) else 0
            idx_90 = np.where(y_step >= y_90)[0][0] if np.any(y_step >= y_90) else len(t_step)-1
            rise_time = t_step[idx_90] - t_step[idx_10]
            
            # Overshoot
            overshoot = 100.0 * (np.max(y_step) - y_step[-1]) / y_step[-1] if y_step[-1] > 0 else 0.0
            
            # Settling time (2% criterion)
            settling_criterion = 0.02 * abs(y_step[-1])
            settled_idx = np.where(np.abs(y_step - y_step[-1]) <= settling_criterion)[0]
            settling_time = t_step[settled_idx[0]] if len(settled_idx) > 0 else t_step[-1]
            
            print(f"\n  {axis_name} Closed-Loop Analysis:")
            print(f"    Gain Margin: {20*np.log10(gm) if gm and np.isfinite(gm) else float('inf'):.2f} dB @ {wgc/(2*np.pi) if wgc else 0:.2f} Hz")
            print(f"    Phase Margin: {pm if pm else 0:.1f}° @ {wpc/(2*np.pi) if wpc else 0:.2f} Hz")
            print(f"    Bandwidth (-3dB): {bw_achieved if bw_achieved else 'N/A':.2f} Hz")
            print(f"    Rise Time: {rise_time*1000:.2f} ms")
            print(f"    Overshoot: {overshoot:.2f}%")
            print(f"    Settling Time (2%): {settling_time*1000:.2f} ms")
            
            return {
                'L': L,
                'T': T,
                'C': C_siso,
                'gain_margin_db': 20*np.log10(gm) if gm and np.isfinite(gm) else float('inf'),
                'phase_margin_deg': pm if pm else 0.0,
                'bandwidth_hz': bw_achieved,
                'crossover_freq_hz': wpc / (2*np.pi) if wpc else None,
                'rise_time_ms': rise_time * 1000,
                'overshoot_percent': overshoot,
                'settling_time_ms': settling_time * 1000,
                't_step': t_step,
                'y_step': y_step
            }
        
        analysis_pan = analyze_closed_loop(plant_pan, C_pan, gains_pan, "Pan")
        analysis_tilt = analyze_closed_loop(plant_tilt, C_tilt, gains_tilt, "Tilt")
        print()
        
        # ====================================================================
        # STEP 5: MULTI-VIEW DIAGNOSTIC PLOTS
        # ====================================================================
        plots_generated = False
        if generate_plots:
            try:
                import matplotlib.pyplot as plt
                print("Step 5: Generating multi-view diagnostic plots...")
                self._generate_diagnostic_plots(
                    plant_pan, plant_tilt,
                    analysis_pan, analysis_tilt,
                    bandwidth_hz
                )
                plots_generated = True
                print("  ✓ All diagnostic plots generated successfully")
            except ImportError:
                print("  ⚠ matplotlib not available - skipping plots")
            except Exception as e:
                print(f"  ⚠ Plot generation failed: {e}")
        
        print()
        print("=" * 80)
        print("PID SYNTHESIS COMPLETE")
        print("=" * 80)
        
        return {
            'gains_pan': gains_pan,
            'gains_tilt': gains_tilt,
            'plant_ss': plant_ss,
            'plant_pan': plant_pan,
            'plant_tilt': plant_tilt,
            'C_pan': C_pan,
            'C_tilt': C_tilt,
            'analysis_pan': analysis_pan,
            'analysis_tilt': analysis_tilt,
            'plots_generated': plots_generated
        }
    
    def _generate_diagnostic_plots(
        self,
        plant_pan, plant_tilt,
        analysis_pan: Dict, analysis_tilt: Dict,
        bandwidth_hz: float
    ) -> None:
        """
        Generate comprehensive diagnostic plots for both axes.
        
        Creates 8 figures (4 per axis):
        - Figure 1-4: Pan axis analysis
        - Figure 5-8: Tilt axis analysis
        """
        import matplotlib.pyplot as plt
        
        def plot_axis_diagnostics(plant, analysis, axis_name: str, fig_offset: int):
            """Generate 4 diagnostic plots for one axis."""
            
            # ================================================================
            # FIGURE 1: TIME DOMAIN RESPONSE (Compensated vs Uncompensated)
            # ================================================================
            fig1 = plt.figure(figsize=(10, 6))
            fig1.suptitle(f'{axis_name} Axis: Time Domain Response', fontsize=14, fontweight='bold')
            
            # Step response of uncompensated plant
            t_uncomp = np.linspace(0, 5.0/bandwidth_hz, 1000)
            t_uncomp, y_uncomp = ctrl.step_response(plant, t_uncomp)
            
            # Step response of compensated closed-loop
            t_comp = analysis['t_step']
            y_comp = analysis['y_step']
            
            plt.subplot(1, 1, 1)
            plt.plot(t_uncomp*1000, y_uncomp, 'b--', linewidth=2, label='Uncompensated Plant')
            plt.plot(t_comp*1000, y_comp, 'r-', linewidth=2, label='PID Closed-Loop')
            plt.axhline(y=1.0, color='k', linestyle=':', alpha=0.5, label='Steady-State')
            plt.axhline(y=1.02, color='g', linestyle=':', alpha=0.3)
            plt.axhline(y=0.98, color='g', linestyle=':', alpha=0.3, label='±2% Band')
            
            # Annotations
            plt.text(0.95, 0.95, f"Rise Time: {analysis['rise_time_ms']:.1f} ms\n"
                                  f"Overshoot: {analysis['overshoot_percent']:.1f}%\n"
                                  f"Settling Time: {analysis['settling_time_ms']:.1f} ms",
                     transform=plt.gca().transAxes, fontsize=10,
                     verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.xlabel('Time [ms]', fontsize=12)
            plt.ylabel('Position [rad/rad]', fontsize=12)
            plt.title('Step Response Comparison', fontsize=11)
            plt.grid(True, alpha=0.3)
            plt.legend(loc='lower right', fontsize=10)
            plt.tight_layout()
            
            # ================================================================
            # FIGURE 2: ROOT LOCUS OF UNCOMPENSATED SYSTEM
            # ================================================================
            fig2 = plt.figure(figsize=(10, 6))
            fig2.suptitle(f'{axis_name} Axis: Root Locus (Uncompensated)', fontsize=14, fontweight='bold')
            
            plt.subplot(1, 1, 1)
            ctrl.root_locus(plant, plot=True, grid=True)
            plt.title('Open-Loop Plant Poles and Zeros', fontsize=11)
            plt.xlabel('Real Axis', fontsize=12)
            plt.ylabel('Imaginary Axis', fontsize=12)
            
            # Highlight poles
            poles = ctrl.poles(plant)
            zeros = ctrl.zeros(plant)
            plt.plot(np.real(poles), np.imag(poles), 'rx', markersize=12, markeredgewidth=3, label='Poles')
            plt.plot(np.real(zeros), np.imag(zeros), 'bo', markersize=12, markeredgewidth=2, label='Zeros')
            
            plt.text(0.05, 0.95, f"Poles: {len(poles)}\nZeros: {len(zeros)}",
                     transform=plt.gca().transAxes, fontsize=10,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            plt.legend(loc='upper right', fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # ================================================================
            # FIGURE 3: ROOT LOCUS OF COMPENSATED SYSTEM
            # ================================================================
            fig3 = plt.figure(figsize=(10, 6))
            fig3.suptitle(f'{axis_name} Axis: Root Locus (PID Compensated)', fontsize=14, fontweight='bold')
            
            L = analysis['L']  # Open-loop L(s) = C(s)G(s)
            
            plt.subplot(1, 1, 1)
            ctrl.root_locus(L, plot=True, grid=True)
            plt.title('Closed-Loop Pole Migration with PID', fontsize=11)
            plt.xlabel('Real Axis', fontsize=12)
            plt.ylabel('Imaginary Axis', fontsize=12)
            
            # Closed-loop poles at designed gain
            cl_poles = ctrl.poles(analysis['T'])
            plt.plot(np.real(cl_poles), np.imag(cl_poles), 'gs', markersize=14, 
                     markeredgewidth=3, label='Closed-Loop Poles')
            
            # Damping ratio contours (optional enhancement)
            plt.text(0.05, 0.95, f"Closed-Loop Poles: {len(cl_poles)}\n"
                                  f"All in LHP: {all(np.real(cl_poles) < 0)}",
                     transform=plt.gca().transAxes, fontsize=10,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            
            plt.legend(loc='upper right', fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # ================================================================
            # FIGURE 4: FREQUENCY RESPONSE (Bode Plot with Margins)
            # ================================================================
            fig4 = plt.figure(figsize=(10, 8))
            fig4.suptitle(f'{axis_name} Axis: Frequency Response (Bode)', fontsize=14, fontweight='bold')
            
            omega = np.logspace(-2, 3, 1000)
            mag, phase, omega = ctrl.bode(L, omega, plot=False)
            mag_db = 20 * np.log10(mag)
            phase_deg = np.rad2deg(phase)
            
            # Magnitude plot
            plt.subplot(2, 1, 1)
            plt.semilogx(omega/(2*np.pi), mag_db, 'b-', linewidth=2, label='|L(jω)|')
            plt.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='0 dB (Crossover)')
            
            # Mark crossover frequency
            wpc = analysis['crossover_freq_hz']
            if wpc:
                plt.axvline(x=wpc, color='g', linestyle=':', alpha=0.7, label=f'ωc = {wpc:.2f} Hz')
            
            plt.ylabel('Magnitude [dB]', fontsize=12)
            plt.title('Open-Loop Frequency Response L(s) = C(s)G(s)', fontsize=11)
            plt.grid(True, which='both', alpha=0.3)
            plt.legend(loc='upper right', fontsize=9)
            
            # Phase plot
            plt.subplot(2, 1, 2)
            plt.semilogx(omega/(2*np.pi), phase_deg, 'b-', linewidth=2, label='∠L(jω)')
            plt.axhline(y=-180, color='r', linestyle='--', alpha=0.7, label='-180° (Instability)')
            
            # Mark phase margin
            pm = analysis['phase_margin_deg']
            if wpc:
                plt.axvline(x=wpc, color='g', linestyle=':', alpha=0.7)
                # Find phase at crossover
                idx_wpc = np.argmin(np.abs(omega/(2*np.pi) - wpc))
                phase_at_wpc = phase_deg[idx_wpc]
                plt.plot(wpc, phase_at_wpc, 'ro', markersize=10, label=f'PM = {pm:.1f}°')
            
            plt.xlabel('Frequency [Hz]', fontsize=12)
            plt.ylabel('Phase [deg]', fontsize=12)
            plt.grid(True, which='both', alpha=0.3)
            plt.legend(loc='lower left', fontsize=9)
            
            # Margin annotations
            gm = analysis['gain_margin_db']
            textstr = f"Gain Margin: {gm:.2f} dB\nPhase Margin: {pm:.1f}°\nBandwidth: {analysis['bandwidth_hz']:.2f} Hz"
            plt.text(0.98, 0.95, textstr,
                     transform=fig4.transFigure, fontsize=10,
                     verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Generate plots for both axes
        plot_axis_diagnostics(plant_pan, analysis_pan, "Pan", 0)
        plot_axis_diagnostics(plant_tilt, analysis_tilt, "Tilt", 4)
        
        print(f"  ✓ Generated 8 diagnostic figures (4 per axis)")


# ============================================================================
# GIMBAL LINEARIZATION AND CONTROLLER SYNTHESIS
# ============================================================================

def linearize_gimbal_at_operating_point(
    gimbal: GimbalDynamics,
    q_op: np.ndarray,
    dq_op: np.ndarray
) -> ctrl.StateSpace:
    """
    Linearize the GimbalDynamics model at a specified operating point.
    
    This function serves as a bridge between the nonlinear gimbal dynamics
    and the linear controller design tools.
    
    Mathematical Background:
    -----------------------
    The nonlinear gimbal dynamics are:
    
    $$M(q) \\ddot{q} + C(q, \\dot{q}) \\dot{q} + G(q) = \\tau$$
    
    Linearization yields the state-space form:
    
    $$\\dot{x} = A x + B u$$
    $$y = C x + D u$$
    
    where $x = [q_1, q_2, \\dot{q}_1, \\dot{q}_2]^T$ and $u = [\\tau_1, \\tau_2]^T$.
    
    Args:
        gimbal (GimbalDynamics): The nonlinear gimbal dynamics object
        q_op (np.ndarray): Operating point joint positions [rad] (2,)
        dq_op (np.ndarray): Operating point joint velocities [rad/s] (2,)
    
    Returns:
        ctrl.StateSpace: Linear state-space model (4 states, 2 inputs, 2 outputs)
    
    Example:
        >>> gimbal = GimbalDynamics()
        >>> q0 = np.array([0.0, 0.0])  # Upright position
        >>> dq0 = np.array([0.0, 0.0])  # At rest
        >>> plant_ss = linearize_gimbal_at_operating_point(gimbal, q0, dq0)
        >>> print(f"Plant has {plant_ss.nstates} states, {plant_ss.ninputs} inputs")
    """
    # Call the linearize method from GimbalDynamics
    A, B, C, D = gimbal.linearize(q_op, dq_op)
    
    # Create state-space object using python-control library
    plant_ss = ctrl.StateSpace(A, B, C, D)
    
    # Set state and input/output names for clarity
    plant_ss.state_labels = ['q_pan', 'q_tilt', 'dq_pan', 'dq_tilt']
    plant_ss.input_labels = ['tau_pan', 'tau_tilt']
    plant_ss.output_labels = ['y_pan', 'y_tilt']
    
    return plant_ss


def design_coarse_pid_from_specs(
    gimbal: GimbalDynamics,
    q_op: np.ndarray = None,
    dq_op: np.ndarray = None,
    bandwidth_hz: float = 5.0,
    phase_margin_deg: float = 60.0,
    use_loop_shaping: bool = True
) -> Dict[str, Union[ControllerGains, Dict]]:
    """
    Design a precision PID controller for the coarse gimbal stage.
    
    This function encapsulates the complete workflow:
    1. Linearize gimbal dynamics at operating point
    2. Analyze open-loop frequency response
    3. Synthesize PID gains to meet specifications
    4. Validate closed-loop performance
    
    Design Methodology:
    ------------------
    For SISO (Single-Input Single-Output) control of each axis:
    
    **Loop Shaping Approach:**
    - Target crossover frequency: $\\omega_c = 2\\pi f_{BW}$
    - Phase margin requirement: $PM \\geq 60°$ for robustness
    - Gain margin requirement: $GM \\geq 6$ dB
    
    **PID Transfer Function:**
    
    $$C(s) = K_p + \\frac{K_i}{s} + K_d s$$
    
    or in standard form with derivative filtering:
    
    $$C(s) = K_p \\left(1 + \\frac{1}{T_i s} + T_d s\\right)$$
    
    **Frequency-Domain Tuning:**
    - $K_p$: Set crossover frequency
    - $K_i$: Improve steady-state error (integral corner below $\\omega_c/10$)
    - $K_d$: Add phase lead for stability (derivative corner near $\\omega_c$)
    
    Args:
        gimbal (GimbalDynamics): Nonlinear gimbal dynamics model
        q_op (np.ndarray, optional): Operating point positions. Defaults to [0, 0].
        dq_op (np.ndarray, optional): Operating point velocities. Defaults to [0, 0].
        bandwidth_hz (float): Target closed-loop bandwidth [Hz]. Default 5 Hz.
        phase_margin_deg (float): Target phase margin [deg]. Default 60°.
        use_loop_shaping (bool): Use analytical loop shaping vs. numerical optimization.
    
    Returns:
        Dict containing:
            - 'gains_pan': ControllerGains for pan axis
            - 'gains_tilt': ControllerGains for tilt axis
            - 'plant_ss': Linearized plant model
            - 'analysis': Frequency response analysis results
    
    Example:
        >>> gimbal = GimbalDynamics(pan_mass=0.5, tilt_mass=0.25)
        >>> result = design_coarse_pid_from_specs(gimbal, bandwidth_hz=5.0)
        >>> print(f"Pan Kp: {result['gains_pan'].kp:.3f}")
    """
    # Default operating point: upright position at rest
    if q_op is None:
        q_op = np.array([0.0, 0.0])
    if dq_op is None:
        dq_op = np.array([0.0, 0.0])
    
    print("=" * 80)
    print("COARSE GIMBAL PID CONTROLLER DESIGN")
    print("=" * 80)
    print(f"Operating Point: q = {q_op}, dq = {dq_op}")
    print(f"Target Bandwidth: {bandwidth_hz:.1f} Hz")
    print(f"Target Phase Margin: {phase_margin_deg:.1f}°")
    print()
    
    # ========================================================================
    # STEP 1: LINEARIZE GIMBAL DYNAMICS
    # ========================================================================
    print("Step 1: Linearizing gimbal dynamics...")
    plant_ss = linearize_gimbal_at_operating_point(gimbal, q_op, dq_op)
    
    # Extract SISO models for each axis (diagonal approximation)
    # Plant output is position, so we have: [q_pan, q_tilt] = C*x + D*u
    # For Pan axis: Transfer from tau_pan to q_pan
    # For Tilt axis: Transfer from tau_tilt to q_tilt
    
    # Get transfer function matrix
    tf_matrix = ctrl.ss2tf(plant_ss)
    
    # Extract diagonal elements (SISO plants)
    plant_pan = tf_matrix[0, 0]   # q_pan / tau_pan
    plant_tilt = tf_matrix[1, 1]  # q_tilt / tau_tilt
    
    print(f"  Pan plant: {plant_pan}")
    print(f"  Tilt plant: {plant_tilt}")
    print()
    
    # ========================================================================
    # STEP 2: ANALYZE OPEN-LOOP FREQUENCY RESPONSE
    # ========================================================================
    print("Step 2: Analyzing open-loop frequency response...")
    
    omega = np.logspace(-2, 3, 500)  # 0.01 to 1000 rad/s
    
    # Pan axis
    mag_pan, phase_pan, _ = ctrl.bode(plant_pan, omega, plot=False)
    
    # Tilt axis
    mag_tilt, phase_tilt, _ = ctrl.bode(plant_tilt, omega, plot=False)
    
    # Find DC gain (low frequency magnitude)
    dc_gain_pan = mag_pan[0]
    dc_gain_tilt = mag_tilt[0]
    
    print(f"  Pan DC gain: {20*np.log10(dc_gain_pan):.2f} dB")
    print(f"  Tilt DC gain: {20*np.log10(dc_gain_tilt):.2f} dB")
    print()
    
    # ========================================================================
    # STEP 3: SYNTHESIZE PID GAINS (LOOP SHAPING)
    # ========================================================================
    print("Step 3: Synthesizing PID gains via loop shaping...")
    
    omega_c = 2 * np.pi * bandwidth_hz  # Crossover frequency [rad/s]
    
    # For a Type 0 plant (position output), add integral action for Type 1 system
    # PID structure: C(s) = Kp * (1 + 1/(Ti*s) + Td*s)
    
    # Simplified tuning rules (analytical approximation):
    # 1. Set Kp to achieve crossover at omega_c
    # 2. Place integral corner at omega_c / 10 for good tracking
    # 3. Add derivative to boost phase margin
    
    def tune_pid_axis(plant_siso, name: str) -> ControllerGains:
        """Tune PID for a single axis."""
        print(f"\n  Tuning {name} axis:")
        
        # Evaluate plant at crossover frequency
        mag_at_wc, phase_at_wc, _ = ctrl.bode(plant_siso, [omega_c], plot=False)
        mag_at_wc = mag_at_wc[0]
        phase_at_wc = phase_at_wc[0]
        
        # Proportional gain: compensate plant gain at crossover
        # We want |Kp * G(jωc)| = 1 (0 dB)
        Kp = 1.0 / mag_at_wc
        
        # Integral gain: Ti = 10 / omega_c (integral corner frequency)
        Ti = 10.0 / omega_c
        Ki = Kp / Ti
        
        # Derivative gain: Add phase lead
        # Rule of thumb: Td = 0.1 * Ti for moderate phase boost
        Td = 0.1 * Ti
        Kd = Kp * Td
        
        # Phase contribution at crossover (estimate)
        # Integral: -90° at high freq
        # Derivative: +arctan(Td * omega_c) ≈ +6° for Td*omega_c = 0.1
        phase_boost = np.rad2deg(np.arctan(Td * omega_c))
        
        print(f"    Kp = {Kp:.3f}")
        print(f"    Ki = {Ki:.3f} (Ti = {Ti:.3f} s)")
        print(f"    Kd = {Kd:.6f} (Td = {Td:.6f} s)")
        print(f"    Estimated phase boost: {phase_boost:.1f}°")
        
        return ControllerGains(kp=Kp, ki=Ki, kd=Kd)
    
    gains_pan = tune_pid_axis(plant_pan, "Pan")
    gains_tilt = tune_pid_axis(plant_tilt, "Tilt")
    
    print()
    
    # ========================================================================
    # STEP 4: VALIDATE CLOSED-LOOP PERFORMANCE
    # ========================================================================
    print("Step 4: Validating closed-loop performance...")
    
    def validate_axis(plant_siso, gains: ControllerGains, name: str) -> Dict:
        """Validate single-axis closed loop."""
        # Construct PID transfer function
        s = ctrl.tf('s')
        C = gains.kp + gains.ki / s + gains.kd * s
        
        # Open-loop transfer function
        L = C * plant_siso
        
        # Closed-loop transfer function
        T = ctrl.feedback(L, 1)
        
        # Stability margins
        gm, pm, wgc, wpc = ctrl.margin(L)
        
        # Bandwidth (frequency where |T| = -3dB)
        mag_cl, _, omega_cl = ctrl.bode(T, omega, plot=False)
        bw_idx = np.where(20*np.log10(mag_cl) <= -3.0)[0]
        if len(bw_idx) > 0:
            bw_achieved = omega_cl[bw_idx[0]] / (2 * np.pi)
        else:
            bw_achieved = None
        
        print(f"\n  {name} axis validation:")
        print(f"    Gain Margin: {20*np.log10(gm) if gm else 'inf':.2f} dB at {wgc/(2*np.pi) if wgc else 0:.2f} Hz")
        print(f"    Phase Margin: {pm if pm else 0:.1f}° at {wpc/(2*np.pi) if wpc else 0:.2f} Hz")
        print(f"    Achieved Bandwidth: {bw_achieved if bw_achieved else 'N/A':.2f} Hz")
        
        return {
            'gain_margin_db': 20*np.log10(gm) if gm else float('inf'),
            'phase_margin_deg': pm if pm else 0.0,
            'bandwidth_hz': bw_achieved,
            'crossover_freq_hz': wpc / (2 * np.pi) if wpc else None
        }
    
    analysis_pan = validate_axis(plant_pan, gains_pan, "Pan")
    analysis_tilt = validate_axis(plant_tilt, gains_tilt, "Tilt")
    
    print()
    print("=" * 80)
    print("DESIGN COMPLETE")
    print("=" * 80)
    
    return {
        'gains_pan': gains_pan,
        'gains_tilt': gains_tilt,
        'plant_ss': plant_ss,
        'plant_pan': plant_pan,
        'plant_tilt': plant_tilt,
        'analysis_pan': analysis_pan,
        'analysis_tilt': analysis_tilt
    }


# ============================================================================
# COMMAND-LINE INTERFACE FOR STANDALONE EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Standalone execution: Design coarse PID controller with comprehensive diagnostics.
    """
    print("\n" + "=" * 80)
    print("GIMBAL CONTROLLER DESIGN TOOL")
    print("=" * 80)
    print("\nThis script demonstrates the complete workflow:")
    print("  1. Instantiate GimbalDynamics")
    print("  2. Linearize at operating point")
    print("  3. Design PID controllers for Pan and Tilt axes")
    print("  4. Validate stability margins and bandwidth")
    print("  5. Generate multi-view diagnostic plots")
    print("\n" + "=" * 80 + "\n")
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    # Physical parameters (matching typical lasercom terminal)
    gimbal = GimbalDynamics(
        pan_mass=0.5,      # 500g pan assembly
        tilt_mass=0.25,    # 250g tilt/payload
        cm_r=0.002,        # 2mm longitudinal offset
        cm_h=0.0005,       # 0.5mm lateral offset
        gravity=9.81
    )
    
    # Operating point: upright "hold" position
    q_nominal = np.array([0.0, 0.0])    # [pan, tilt] = [0°, 0°]
    dq_nominal = np.array([0.0, 0.0])   # At rest
    
    # Performance specifications
    target_bandwidth = 5.0      # 5 Hz for coarse stage (conservative)
    target_phase_margin = 60.0  # 60° for robustness
    
    # ========================================================================
    # CONTROLLER SYNTHESIS WITH DIAGNOSTIC PLOTS
    # ========================================================================
    designer = ControllerDesigner()
    
    result = designer.design_coarse_pid_from_specs(
        gimbal=gimbal,
        q_op=q_nominal,
        dq_op=dq_nominal,
        bandwidth_hz=target_bandwidth,
        phase_margin_deg=target_phase_margin,
        use_loop_shaping=True,
        derivative_filter_N=15.0,
        generate_plots=True
    )
    
    # ========================================================================
    # OUTPUT SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("FINAL CONTROLLER GAINS")
    print("=" * 80)
    print(f"\nPan Axis PID:")
    print(f"  Kp = {result['gains_pan'].kp:.6f}")
    print(f"  Ki = {result['gains_pan'].ki:.6f}")
    print(f"  Kd = {result['gains_pan'].kd:.6f}")
    
    print(f"\nTilt Axis PID:")
    print(f"  Kp = {result['gains_tilt'].kp:.6f}")
    print(f"  Ki = {result['gains_tilt'].ki:.6f}")
    print(f"  Kd = {result['gains_tilt'].kd:.6f}")
    
    print("\n" + "=" * 80)
    print("STABILITY ANALYSIS")
    print("=" * 80)
    print(f"\nPan Axis:")
    print(f"  Gain Margin: {result['analysis_pan']['gain_margin_db']:.2f} dB")
    print(f"  Phase Margin: {result['analysis_pan']['phase_margin_deg']:.1f}°")
    print(f"  Bandwidth: {result['analysis_pan']['bandwidth_hz']:.2f} Hz")
    print(f"  Rise Time: {result['analysis_pan']['rise_time_ms']:.2f} ms")
    print(f"  Overshoot: {result['analysis_pan']['overshoot_percent']:.2f}%")
    
    print(f"\nTilt Axis:")
    print(f"  Gain Margin: {result['analysis_tilt']['gain_margin_db']:.2f} dB")
    print(f"  Phase Margin: {result['analysis_tilt']['phase_margin_deg']:.1f}°")
    print(f"  Bandwidth: {result['analysis_tilt']['bandwidth_hz']:.2f} Hz")
    print(f"  Rise Time: {result['analysis_tilt']['rise_time_ms']:.2f} ms")
    print(f"  Overshoot: {result['analysis_tilt']['overshoot_percent']:.2f}%")
    
    print("\n" + "=" * 80)
    print("USE IN SIMULATION")
    print("=" * 80)
    print("\nTo use these gains in the simulation, add to SimulationConfig:")
    print(f"""
coarse_controller_config={{
    'kp': {result['gains_pan'].kp:.3f},
    'ki': {result['gains_pan'].ki:.3f},
    'kd': {result['gains_pan'].kd:.6f},
    'anti_windup_gain': 1.0,
    'tau_rate_limit': 50.0
}}
""")
    
    if result['plots_generated']:
        print("=" * 80)
        print("DIAGNOSTIC PLOTS")
        print("=" * 80)
        print("\n8 diagnostic figures generated:")
        print("  Pan Axis:")
        print("    • Figure 1: Time Domain Response (Step)")
        print("    • Figure 2: Root Locus (Uncompensated)")
        print("    • Figure 3: Root Locus (Compensated)")
        print("    • Figure 4: Bode Plot with Margins")
        print("  Tilt Axis:")
        print("    • Figure 5: Time Domain Response (Step)")
        print("    • Figure 6: Root Locus (Uncompensated)")
        print("    • Figure 7: Root Locus (Compensated)")
        print("    • Figure 8: Bode Plot with Margins")
        print("\nClose all plot windows to exit...")
        
        import matplotlib.pyplot as plt
        plt.show()
    
    print("\n" + "=" * 80 + "\n")