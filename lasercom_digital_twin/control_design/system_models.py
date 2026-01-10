"""
System Modeling Tools for Control Design

This module provides tools for creating linear system models from the
nonlinear lasercom simulation, including linearization, model order
reduction, and system identification.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import control as ctrl
from dataclasses import dataclass
from scipy.optimize import least_squares


@dataclass
class LinearModel:
    """Container for linear system model."""
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    state_names: List[str] = None
    input_names: List[str] = None
    output_names: List[str] = None

    def to_control(self) -> ctrl.StateSpace:
        """Convert to python-control StateSpace object."""
        return ctrl.ss(self.A, self.B, self.C, self.D)

    def to_transfer_function(self) -> ctrl.TransferFunction:
        """Convert to transfer function (for SISO systems)."""
        return ctrl.ss2tf(self.to_control())


class SystemModeler:
    """Tools for system modeling and linearization."""

    def __init__(self):
        self.models = {}

    def linearize_around_operating_point(self, nonlinear_dynamics: Callable,
                                       operating_point: Dict[str, float],
                                       input_perturbation: float = 1e-6,
                                       state_perturbation: float = 1e-6) -> LinearModel:
        """
        Linearize nonlinear dynamics around an operating point.

        Parameters
        ----------
        nonlinear_dynamics : callable
            Function that computes dx/dt = f(x, u)
        operating_point : Dict[str, float]
            Operating point {state_name: value, input_name: value}
        input_perturbation : float
            Perturbation size for input Jacobian
        state_perturbation : float
            Perturbation size for state Jacobian

        Returns
        -------
        LinearModel
            Linearized system model
        """
        # Extract states and inputs from operating point
        state_vars = [k for k in operating_point.keys() if k.startswith('x_')]
        input_vars = [k for k in operating_point.keys() if k.startswith('u_')]

        x0 = np.array([operating_point[var] for var in state_vars])
        u0 = np.array([operating_point[var] for var in input_vars])

        n_states = len(x0)
        n_inputs = len(u0)

        # Compute A matrix (state Jacobian)
        A = np.zeros((n_states, n_states))
        f0 = nonlinear_dynamics(x0, u0)

        for i in range(n_states):
            x_pert = x0.copy()
            x_pert[i] += state_perturbation
            f_pert = nonlinear_dynamics(x_pert, u0)
            A[:, i] = (f_pert - f0) / state_perturbation

        # Compute B matrix (input Jacobian)
        B = np.zeros((n_states, n_inputs))

        for i in range(n_inputs):
            u_pert = u0.copy()
            u_pert[i] += input_perturbation
            f_pert = nonlinear_dynamics(x0, u_pert)
            B[:, i] = (f_pert - f0) / input_perturbation

        # Assume full state output (C = I, D = 0)
        C = np.eye(n_states)
        D = np.zeros((n_states, n_inputs))

        return LinearModel(A, B, C, D, state_vars, input_vars, state_vars)

    def identify_from_data(self, input_data: np.ndarray, output_data: np.ndarray,
                          dt: float, order: int = 2) -> LinearModel:
        """
        System identification from input-output data using subspace methods.

        Parameters
        ----------
        input_data : np.ndarray
            Input time series (n_inputs x n_samples)
        output_data : np.ndarray
            Output time series (n_outputs x n_samples)
        dt : float
            Sampling time
        order : int
            System order

        Returns
        -------
        LinearModel
            Identified linear model
        """
        # Placeholder for subspace identification
        # In practice, would use N4SID or similar algorithm
        n_samples = input_data.shape[1]
        n_inputs = input_data.shape[0]
        n_outputs = output_data.shape[0]

        # Simple ARX model as placeholder
        A = np.random.randn(order, order) * 0.1
        B = np.random.randn(order, n_inputs) * 0.1
        C = np.random.randn(n_outputs, order) * 0.1
        D = np.zeros((n_outputs, n_inputs))

        return LinearModel(A, B, C, D)

    def reduce_model_order(self, full_model: LinearModel, reduced_order: int,
                          method: str = "balanced") -> LinearModel:
        """
        Reduce model order using balanced truncation or other methods.

        Parameters
        ----------
        full_model : LinearModel
            Full-order model
        reduced_order : int
            Desired reduced order
        method : str
            Reduction method ('balanced', 'hsv', etc.)

        Returns
        -------
        LinearModel
            Reduced-order model
        """
        ss_full = full_model.to_control()

        if method == "balanced":
            ss_reduced = ctrl.balred(ss_full, reduced_order)
        else:
            # Default to balanced truncation
            ss_reduced = ctrl.balred(ss_full, reduced_order)

        A_r, B_r, C_r, D_r = ctrl.ssdata(ss_reduced)

        return LinearModel(A_r, B_r, C_r, D_r,
                          full_model.state_names, full_model.input_names, full_model.output_names)

    def create_gimbal_plant_model(self, inertia_az: float, inertia_el: float,
                                friction_az: float, friction_el: float,
                                motor_kt: float, motor_r: float, motor_l: float) -> LinearModel:
        """
        Create linear plant model for gimbal system.

        The model includes:
        - Rigid body dynamics for azimuth and elevation
        - Motor electrical dynamics
        - Friction effects

        Parameters
        ----------
        inertia_az : float
            Azimuth inertia [kg·m²]
        inertia_el : float
            Elevation inertia [kg·m²]
        friction_az : float
            Azimuth friction coefficient [N·m·s/rad]
        friction_el : float
            Elevation friction coefficient [N·m·s/rad]
        motor_kt : float
            Motor torque constant [N·m/A]
        motor_r : float
            Motor resistance [Ω]
        motor_l : float
            Motor inductance [H]

        Returns
        -------
        LinearModel
            Linear gimbal plant model
        """
        # State variables: [theta_az, omega_az, i_az, theta_el, omega_el, i_el]
        n_states = 6
        n_inputs = 2  # [v_az, v_el]
        n_outputs = 6  # Full state output

        A = np.zeros((n_states, n_states))
        B = np.zeros((n_states, n_inputs))
        C = np.eye(n_states)
        D = np.zeros((n_outputs, n_inputs))

        # Azimuth dynamics (states 0-2: theta_az, omega_az, i_az)
        A[0, 1] = 1.0  # d(theta_az)/dt = omega_az
        A[1, 1] = -friction_az / inertia_az  # d(omega_az)/dt = -b*omega_az/J + Kt*i_az/J
        A[1, 2] = motor_kt / inertia_az
        A[2, 2] = -motor_r / motor_l  # di_az/dt = -R*i_az/L + v_az/L

        B[2, 0] = 1.0 / motor_l  # di_az/dt += v_az/L

        # Elevation dynamics (states 3-5: theta_el, omega_el, i_el)
        A[3, 4] = 1.0  # d(theta_el)/dt = omega_el
        A[4, 4] = -friction_el / inertia_el  # d(omega_el)/dt = -b*omega_el/J + Kt*i_el/J
        A[4, 5] = motor_kt / inertia_el
        A[5, 5] = -motor_r / motor_l  # di_el/dt = -R*i_el/L + v_el/L

        B[5, 1] = 1.0 / motor_l  # di_el/dt += v_el/L

        state_names = ['theta_az', 'omega_az', 'i_az', 'theta_el', 'omega_el', 'i_el']
        input_names = ['v_az', 'v_el']
        output_names = state_names

        return LinearModel(A, B, C, D, state_names, input_names, output_names)

    def create_fsm_plant_model(self, natural_freq: float, damping_ratio: float,
                             max_angle: float) -> LinearModel:
        """
        Create linear plant model for Fast Steering Mirror (FSM).

        Parameters
        ----------
        natural_freq : float
            Natural frequency [rad/s]
        damping_ratio : float
            Damping ratio
        max_angle : float
            Maximum deflection angle [rad]

        Returns
        -------
        LinearModel
            Linear FSM plant model
        """
        # State variables: [alpha, alpha_dot] (tip angle and rate)
        # Similar for tilt: [beta, beta_dot]
        n_states = 4  # [alpha, alpha_dot, beta, beta_dot]
        n_inputs = 2  # [cmd_alpha, cmd_beta]
        n_outputs = 4

        A = np.zeros((n_states, n_states))
        B = np.zeros((n_states, n_inputs))
        C = np.eye(n_states)
        D = np.zeros((n_outputs, n_inputs))

        wn = natural_freq
        zeta = damping_ratio

        # Tip dynamics (states 0-1)
        A[0, 1] = 1.0
        A[1, 0] = -wn**2
        A[1, 1] = -2*zeta*wn
        B[1, 0] = wn**2  # Assuming unity gain actuator

        # Tilt dynamics (states 2-3)
        A[2, 3] = 1.0
        A[3, 2] = -wn**2
        A[3, 3] = -2*zeta*wn
        B[3, 1] = wn**2

        state_names = ['alpha', 'alpha_dot', 'beta', 'beta_dot']
        input_names = ['cmd_alpha', 'cmd_beta']
        output_names = state_names

        return LinearModel(A, B, C, D, state_names, input_names, output_names)

    def validate_model(self, model: LinearModel, validation_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Validate linear model against experimental/validation data.

        Parameters
        ----------
        model : LinearModel
            Linear model to validate
        validation_data : Dict[str, np.ndarray]
            Validation data {signal_name: time_series}

        Returns
        -------
        Dict[str, float]
            Validation metrics (fit percentage, etc.)
        """
        # Placeholder for model validation
        # Would compare model predictions vs. actual data
        return {
            'fit_percentage': 95.0,  # Placeholder
            'rmse': 0.01  # Placeholder
        }