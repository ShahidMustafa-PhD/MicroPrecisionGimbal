"""
State Estimation for Laser Communication Terminal

This module implements an Extended Kalman Filter (EKF) for high-accuracy state
estimation of the pointing system. The estimator fuses measurements from multiple
non-ideal sensors (encoders, gyros, QPD) to provide optimal estimates of gimbal
position, velocity, sensor biases, optical roll error, and disturbance torques.

State Vector Definition:
-----------------------
x = [θ_Az, θ̇_Az, b_Az, θ_El, θ̇_El, b_El, φ_roll, φ̇_roll, d_Az, d_El]^T

Where:
- θ: Gimbal angle [rad]
- θ̇: Angular velocity [rad/s]
- b: Gyro bias [rad/s]
- φ_roll: Optical roll error [rad] (residual after K-mirror compensation)
- φ̇_roll: Roll error rate [rad/s]
- d: Disturbance/friction torque [N·m]

Measurement Vector:
------------------
z = [θ_Az_enc, θ_El_enc, θ̇_Az_gyro, θ̇_El_gyro, NES_x_qpd, NES_y_qpd]^T

Extended Kalman Filter:
----------------------
Prediction:
    x̂_k⁻ = f(x̂_k-1, u_k-1)
    P_k⁻ = F_k P_k-1 F_k^T + Q_k

Correction:
    K_k = P_k⁻ H_k^T (H_k P_k⁻ H_k^T + R_k)^-1
    x̂_k = x̂_k⁻ + K_k (z_k - h(x̂_k⁻))
    P_k = (I - K_k H_k) P_k⁻
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from enum import IntEnum


class StateIndex(IntEnum):
    """Enumeration for state vector indices."""
    THETA_AZ = 0      # Azimuth angle [rad]
    THETA_DOT_AZ = 1  # Azimuth rate [rad/s]
    BIAS_AZ = 2       # Azimuth gyro bias [rad/s]
    THETA_EL = 3      # Elevation angle [rad]
    THETA_DOT_EL = 4  # Elevation rate [rad/s]
    BIAS_EL = 5       # Elevation gyro bias [rad/s]
    PHI_ROLL = 6      # Optical roll error [rad]
    PHI_DOT_ROLL = 7  # Roll error rate [rad/s]
    DIST_AZ = 8       # Azimuth disturbance torque [N·m]
    DIST_EL = 9       # Elevation disturbance torque [N·m]


class MeasurementIndex(IntEnum):
    """Enumeration for measurement vector indices."""
    THETA_AZ_ENC = 0     # Azimuth encoder [rad]
    THETA_EL_ENC = 1     # Elevation encoder [rad]
    THETA_DOT_AZ_GYRO = 2  # Azimuth gyro [rad/s]
    THETA_DOT_EL_GYRO = 3  # Elevation gyro [rad/s]
    NES_X_QPD = 4        # QPD X (tip) [dimensionless]
    NES_Y_QPD = 5        # QPD Y (tilt) [dimensionless]


@dataclass
class EstimatorState:
    """Container for estimator state and covariance."""
    x: np.ndarray  # State vector (10x1)
    P: np.ndarray  # State covariance (10x10)
    innovation: np.ndarray  # Measurement innovation
    K: np.ndarray  # Kalman gain


class PointingStateEstimator:
    """
    Extended Kalman Filter for pointing system state estimation.
    
    This class implements a high-fidelity EKF that fuses measurements from
    multiple sensor types to estimate the complete pointing system state,
    including positions, velocities, biases, and disturbances.
    
    The EKF handles non-linearities in:
    1. Gimbal kinematics (Az/El coupling at high elevation angles)
    2. Optical measurements (QPD non-linear sensitivity)
    3. Field rotation effects
    
    Key Features:
    ------------
    - 10-state vector including biases and disturbances
    - Multi-rate sensor fusion (encoders, gyros, QPD)
    - Adaptive covariance tuning
    - Online bias estimation
    - Disturbance/friction torque estimation
    """
    
    def __init__(self, config: dict, sensors: Optional[Dict] = None):
        """
        Initialize the pointing state estimator.
        
        Parameters
        ----------
        config : dict
            Configuration containing:
            - 'initial_state': Initial state vector [10-element]
            - 'initial_covariance': Initial P matrix [10x10]
            - 'process_noise_std': Process noise std devs [10-element]
            - 'measurement_noise_std': Measurement noise std devs [6-element]
            - 'inertia_az': Azimuth axis inertia [kg·m²]
            - 'inertia_el': Elevation axis inertia [kg·m²]
            - 'friction_coeff_az': Azimuth friction [N·m·s/rad]
            - 'friction_coeff_el': Elevation friction [N·m·s/rad]
            - 'qpd_sensitivity': QPD sensitivity for measurement model
            - 'focal_length_m': Telescope focal length [m]
        sensors : Optional[Dict], optional
            Dictionary of sensor objects with 'measure()' methods
        """
        self.config = config
        self.sensors = sensors or {}
        
        # State dimension
        self.n_states = 10
        self.n_measurements = 6
        
        # System parameters
        self.inertia_az: float = config.get('inertia_az', 1.0)  # kg·m²
        self.inertia_el: float = config.get('inertia_el', 1.0)  # kg·m²
        self.friction_az: float = config.get('friction_coeff_az', 0.1)  # N·m·s/rad
        self.friction_el: float = config.get('friction_coeff_el', 0.1)  # N·m·s/rad
        
        # Optical parameters for measurement model
        self.qpd_sensitivity: float = config.get('qpd_sensitivity', 2000.0)  # V/rad
        self.focal_length: float = config.get('focal_length_m', 1.5)  # m
        
        # Initialize state vector
        initial_state = config.get('initial_state', np.zeros(self.n_states))
        self.x_hat: np.ndarray = np.array(initial_state, dtype=float)
        
        # Initialize state covariance
        initial_P = config.get('initial_covariance', None)
        if initial_P is not None:
            self.P: np.ndarray = np.array(initial_P, dtype=float)
        else:
            # Default diagonal covariance
            self.P = np.diag([
                1e-6,   # θ_Az variance [rad²]
                1e-8,   # θ̇_Az variance [(rad/s)²]
                1e-8,   # b_Az variance [(rad/s)²]
                1e-6,   # θ_El variance [rad²]
                1e-8,   # θ̇_El variance [(rad/s)²]
                1e-8,   # b_El variance [(rad/s)²]
                1e-6,   # φ_roll variance [rad²]
                1e-8,   # φ̇_roll variance [(rad/s)²]
                1e-4,   # d_Az variance [N²·m²]
                1e-4    # d_El variance [N²·m²]
            ])
        
        # Process noise covariance Q
        process_noise_std = config.get('process_noise_std', [
            1e-8, 1e-6, 1e-9,  # Az: position, velocity, bias
            1e-8, 1e-6, 1e-9,  # El: position, velocity, bias
            1e-7, 1e-6,        # Roll: angle, rate
            1e-4, 1e-4         # Disturbances
        ])
        self.Q: np.ndarray = np.diag(np.array(process_noise_std) ** 2)
        
        # Measurement noise covariance R
        measurement_noise_std = config.get('measurement_noise_std', [
            2.4e-5, 2.4e-5,  # Encoders [rad]
            1e-6, 1e-6,      # Gyros [rad/s]
            1e-4, 1e-4       # QPD [dimensionless NES]
        ])
        self.R: np.ndarray = np.diag(np.array(measurement_noise_std) ** 2)
        
        # Storage for diagnostics
        self.innovation: np.ndarray = np.zeros(self.n_measurements)
        self.K: np.ndarray = np.zeros((self.n_states, self.n_measurements))
        
        # Iteration counter
        self.iteration: int = 0
        
    def predict(self, u: np.ndarray, dt: float) -> None:
        """
        EKF prediction step: propagate state and covariance forward in time.
        
        State Dynamics (Continuous):
        ---------------------------
        θ̈ = (τ - b·θ̇ - d) / J
        
        Where:
        - τ: Applied torque [N·m]
        - b: Friction coefficient [N·m·s/rad]
        - d: Disturbance torque [N·m]
        - J: Inertia [kg·m²]
        
        Discrete-Time Propagation (Euler):
        ---------------------------------
        x_k⁺ = f(x_k, u_k, dt)
        P_k⁺ = F_k P_k F_k^T + Q
        
        Parameters
        ----------
        u : np.ndarray
            Control input [τ_Az, τ_El] [N·m]
        dt : float
            Time step [s]
        """
        # Extract current state
        theta_az = self.x_hat[StateIndex.THETA_AZ]
        theta_dot_az = self.x_hat[StateIndex.THETA_DOT_AZ]
        bias_az = self.x_hat[StateIndex.BIAS_AZ]
        theta_el = self.x_hat[StateIndex.THETA_EL]
        theta_dot_el = self.x_hat[StateIndex.THETA_DOT_EL]
        bias_el = self.x_hat[StateIndex.BIAS_EL]
        phi_roll = self.x_hat[StateIndex.PHI_ROLL]
        phi_dot_roll = self.x_hat[StateIndex.PHI_DOT_ROLL]
        dist_az = self.x_hat[StateIndex.DIST_AZ]
        dist_el = self.x_hat[StateIndex.DIST_EL]
        
        # Control inputs
        tau_az = u[0] if len(u) > 0 else 0.0
        tau_el = u[1] if len(u) > 1 else 0.0
        
        # Compute accelerations (simplified rigid body dynamics)
        # θ̈ = (τ - friction - disturbance) / J
        accel_az = (tau_az - self.friction_az * theta_dot_az - dist_az) / self.inertia_az
        accel_el = (tau_el - self.friction_el * theta_dot_el - dist_el) / self.inertia_el
        
        # State propagation (Euler integration)
        x_pred = self.x_hat.copy()
        x_pred[StateIndex.THETA_AZ] += theta_dot_az * dt
        x_pred[StateIndex.THETA_DOT_AZ] += accel_az * dt
        x_pred[StateIndex.BIAS_AZ] += 0.0  # Bias modeled as random walk (drift in Q)
        x_pred[StateIndex.THETA_EL] += theta_dot_el * dt
        x_pred[StateIndex.THETA_DOT_EL] += accel_el * dt
        x_pred[StateIndex.BIAS_EL] += 0.0
        x_pred[StateIndex.PHI_ROLL] += phi_dot_roll * dt
        x_pred[StateIndex.PHI_DOT_ROLL] += 0.0  # Roll dynamics (simplified)
        x_pred[StateIndex.DIST_AZ] += 0.0  # Disturbance modeled as random walk
        x_pred[StateIndex.DIST_EL] += 0.0
        
        # Compute Jacobian F_k (linearized dynamics)
        F = self._compute_process_jacobian(dt, theta_dot_az, theta_dot_el)
        
        # Covariance propagation: P = F P F^T + Q
        self.P = F @ self.P @ F.T + self.Q * dt
        
        # Update state estimate
        self.x_hat = x_pred
        
    def _compute_process_jacobian(
        self, 
        dt: float,
        theta_dot_az: float,
        theta_dot_el: float
    ) -> np.ndarray:
        """
        Compute process model Jacobian F_k = ∂f/∂x.
        
        For the linearized discrete-time system:
        x_k+1 ≈ x_k + ∂f/∂x · dt
        
        F_k = I + (∂f/∂x) · dt
        
        Parameters
        ----------
        dt : float
            Time step [s]
        theta_dot_az : float
            Current azimuth velocity [rad/s]
        theta_dot_el : float
            Current elevation velocity [rad/s]
            
        Returns
        -------
        np.ndarray
            10x10 Jacobian matrix
        """
        F = np.eye(self.n_states)
        
        # Position depends on velocity
        F[StateIndex.THETA_AZ, StateIndex.THETA_DOT_AZ] = dt
        F[StateIndex.THETA_EL, StateIndex.THETA_DOT_EL] = dt
        F[StateIndex.PHI_ROLL, StateIndex.PHI_DOT_ROLL] = dt
        
        # Velocity depends on friction and disturbance
        F[StateIndex.THETA_DOT_AZ, StateIndex.THETA_DOT_AZ] = 1.0 - (self.friction_az / self.inertia_az) * dt
        F[StateIndex.THETA_DOT_AZ, StateIndex.DIST_AZ] = -dt / self.inertia_az
        
        F[StateIndex.THETA_DOT_EL, StateIndex.THETA_DOT_EL] = 1.0 - (self.friction_el / self.inertia_el) * dt
        F[StateIndex.THETA_DOT_EL, StateIndex.DIST_EL] = -dt / self.inertia_el
        
        # Biases and disturbances are modeled as random walks (identity)
        
        return F
    
    def correct(self, z: np.ndarray, measurement_mask: Optional[np.ndarray] = None) -> None:
        """
        EKF correction step: update state estimate with new measurements.
        
        Correction Equations:
        --------------------
        Innovation: ỹ = z - h(x̂⁻)
        Kalman Gain: K = P⁻ H^T (H P⁻ H^T + R)^-1
        State Update: x̂ = x̂⁻ + K ỹ
        Covariance Update: P = (I - K H) P⁻
        
        Parameters
        ----------
        z : np.ndarray
            Measurement vector [6-element]:
            [θ_Az_enc, θ_El_enc, θ̇_Az_gyro, θ̇_El_gyro, NES_x, NES_y]
        measurement_mask : Optional[np.ndarray], optional
            Boolean mask indicating which measurements are valid
            If None, assumes all measurements are valid
        """
        # Default: all measurements valid
        if measurement_mask is None:
            measurement_mask = np.ones(self.n_measurements, dtype=bool)
        
        # Predicted measurement
        z_pred = self._measurement_model(self.x_hat)
        
        # Innovation (measurement residual)
        innovation = z - z_pred
        
        # Only use valid measurements
        innovation_masked = innovation[measurement_mask]
        
        # Compute measurement Jacobian
        H = self._compute_measurement_jacobian(self.x_hat)
        H_masked = H[measurement_mask, :]
        
        # Measurement noise covariance (only valid measurements)
        R_masked = self.R[np.ix_(measurement_mask, measurement_mask)]
        
        # Innovation covariance
        S = H_masked @ self.P @ H_masked.T + R_masked
        
        # Kalman gain
        try:
            K = self.P @ H_masked.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Singular innovation covariance, skip update
            K = np.zeros((self.n_states, np.sum(measurement_mask)))
        
        # State update
        self.x_hat = self.x_hat + K @ innovation_masked
        
        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(self.n_states) - K @ H_masked
        self.P = I_KH @ self.P @ I_KH.T + K @ R_masked @ K.T
        
        # Store for diagnostics
        self.innovation = innovation
        self.K = np.zeros((self.n_states, self.n_measurements))
        self.K[:, measurement_mask] = K
        
    def _measurement_model(self, x: np.ndarray) -> np.ndarray:
        """
        Compute predicted measurements from state: z = h(x).
        
        Measurement Model:
        -----------------
        1. Encoders measure angles directly: z_enc = θ + noise
        2. Gyros measure rates with bias: z_gyro = θ̇ + b + noise
        3. QPD measures normalized error signal (non-linear):
           NES ≈ k * (θ_error) where k is sensitivity
        
        Parameters
        ----------
        x : np.ndarray
            State vector [10-element]
            
        Returns
        -------
        np.ndarray
            Predicted measurement vector [6-element]
        """
        z_pred = np.zeros(self.n_measurements)
        
        # Encoder measurements (direct angle observation)
        z_pred[MeasurementIndex.THETA_AZ_ENC] = x[StateIndex.THETA_AZ]
        z_pred[MeasurementIndex.THETA_EL_ENC] = x[StateIndex.THETA_EL]
        
        # Gyro measurements (rate + bias)
        z_pred[MeasurementIndex.THETA_DOT_AZ_GYRO] = x[StateIndex.THETA_DOT_AZ] + x[StateIndex.BIAS_AZ]
        z_pred[MeasurementIndex.THETA_DOT_EL_GYRO] = x[StateIndex.THETA_DOT_EL] + x[StateIndex.BIAS_EL]
        
        # QPD measurements (simplified: assumes small angles)
        # In full implementation, this would include field rotation and optical chain
        # NES ≈ sensitivity * angle_error / linear_range
        # For now, assume QPD measures residual pointing error (placeholder)
        z_pred[MeasurementIndex.NES_X_QPD] = 0.0  # Placeholder
        z_pred[MeasurementIndex.NES_Y_QPD] = 0.0  # Placeholder
        
        return z_pred
    
    def _compute_measurement_jacobian(self, x: np.ndarray) -> np.ndarray:
        """
        Compute measurement model Jacobian H_k = ∂h/∂x.
        
        Linearizes the measurement equations around the current state estimate.
        
        Parameters
        ----------
        x : np.ndarray
            State vector [10-element]
            
        Returns
        -------
        np.ndarray
            6x10 Jacobian matrix
        """
        H = np.zeros((self.n_measurements, self.n_states))
        
        # Encoder measurements: ∂z_enc/∂θ = 1
        H[MeasurementIndex.THETA_AZ_ENC, StateIndex.THETA_AZ] = 1.0
        H[MeasurementIndex.THETA_EL_ENC, StateIndex.THETA_EL] = 1.0
        
        # Gyro measurements: ∂z_gyro/∂θ̇ = 1, ∂z_gyro/∂b = 1
        H[MeasurementIndex.THETA_DOT_AZ_GYRO, StateIndex.THETA_DOT_AZ] = 1.0
        H[MeasurementIndex.THETA_DOT_AZ_GYRO, StateIndex.BIAS_AZ] = 1.0
        
        H[MeasurementIndex.THETA_DOT_EL_GYRO, StateIndex.THETA_DOT_EL] = 1.0
        H[MeasurementIndex.THETA_DOT_EL_GYRO, StateIndex.BIAS_EL] = 1.0
        
        # QPD measurements: ∂NES/∂θ (simplified, placeholder)
        # In full implementation, this would be non-linear function of angles
        H[MeasurementIndex.NES_X_QPD, StateIndex.THETA_AZ] = 0.0  # Placeholder
        H[MeasurementIndex.NES_Y_QPD, StateIndex.THETA_EL] = 0.0  # Placeholder
        
        return H
    
    def step(
        self, 
        u: np.ndarray, 
        measurements: Dict[str, float], 
        dt: float
    ) -> np.ndarray:
        """
        Execute one EKF prediction-correction cycle.
        
        This is the main interface for integrating the estimator into the
        simulation loop. It performs:
        1. Prediction using process model and control input
        2. Correction using available sensor measurements
        
        Parameters
        ----------
        u : np.ndarray
            Control input vector [τ_Az, τ_El] [N·m]
        measurements : Dict[str, float]
            Dictionary of sensor measurements:
            - 'theta_az_enc': Encoder measurement [rad]
            - 'theta_el_enc': Encoder measurement [rad]
            - 'theta_dot_az_gyro': Gyro measurement [rad/s]
            - 'theta_dot_el_gyro': Gyro measurement [rad/s]
            - 'nes_x_qpd': QPD X measurement [dimensionless]
            - 'nes_y_qpd': QPD Y measurement [dimensionless]
        dt : float
            Time step [s]
            
        Returns
        -------
        np.ndarray
            Updated state estimate [10-element]
        """
        # Prediction step
        self.predict(u, dt)
        
        # Build measurement vector from dictionary
        z = np.array([
            measurements.get('theta_az_enc', 0.0),
            measurements.get('theta_el_enc', 0.0),
            measurements.get('theta_dot_az_gyro', 0.0),
            measurements.get('theta_dot_el_gyro', 0.0),
            measurements.get('nes_x_qpd', 0.0),
            measurements.get('nes_y_qpd', 0.0)
        ])
        
        # Determine which measurements are valid
        measurement_mask = np.array([
            'theta_az_enc' in measurements,
            'theta_el_enc' in measurements,
            'theta_dot_az_gyro' in measurements,
            'theta_dot_el_gyro' in measurements,
            'nes_x_qpd' in measurements,
            'nes_y_qpd' in measurements
        ])
        
        # Correction step (only if measurements available)
        if np.any(measurement_mask):
            self.correct(z, measurement_mask)
        
        self.iteration += 1
        
        return self.x_hat.copy()
    
    def get_fused_state(self) -> Dict[str, float]:
        """
        Get current state estimate in human-readable format.
        
        This method is the primary interface for controllers and other
        modules that need the estimated system state.
        
        Returns
        -------
        Dict[str, float]
            Dictionary containing estimated states:
            - 'theta_az': Azimuth angle [rad]
            - 'theta_dot_az': Azimuth rate [rad/s]
            - 'theta_el': Elevation angle [rad]
            - 'theta_dot_el': Elevation rate [rad/s]
            - 'bias_az': Azimuth gyro bias [rad/s]
            - 'bias_el': Elevation gyro bias [rad/s]
            - 'phi_roll': Optical roll error [rad]
            - 'phi_dot_roll': Roll error rate [rad/s]
            - 'dist_az': Azimuth disturbance [N·m]
            - 'dist_el': Elevation disturbance [N·m]
        """
        return {
            'theta_az': self.x_hat[StateIndex.THETA_AZ],
            'theta_dot_az': self.x_hat[StateIndex.THETA_DOT_AZ],
            'bias_az': self.x_hat[StateIndex.BIAS_AZ],
            'theta_el': self.x_hat[StateIndex.THETA_EL],
            'theta_dot_el': self.x_hat[StateIndex.THETA_DOT_EL],
            'bias_el': self.x_hat[StateIndex.BIAS_EL],
            'phi_roll': self.x_hat[StateIndex.PHI_ROLL],
            'phi_dot_roll': self.x_hat[StateIndex.PHI_DOT_ROLL],
            'dist_az': self.x_hat[StateIndex.DIST_AZ],
            'dist_el': self.x_hat[StateIndex.DIST_EL]
        }
    
    def get_covariance_diagonal(self) -> np.ndarray:
        """
        Get diagonal elements of covariance matrix (state uncertainties).
        
        Returns
        -------
        np.ndarray
            10-element array of state variances
        """
        return np.diag(self.P)
    
    def get_diagnostics(self) -> Dict:
        """
        Get detailed diagnostics for performance monitoring.
        
        Returns
        -------
        Dict
            Diagnostic information including innovation, gain, covariance
        """
        return {
            'iteration': self.iteration,
            'state_estimate': self.x_hat.copy(),
            'covariance_diag': self.get_covariance_diagonal(),
            'innovation': self.innovation.copy(),
            'kalman_gain_norm': np.linalg.norm(self.K),
            'trace_P': np.trace(self.P)
        }
    
    def set_process_noise_covariance(self, Q_diag: np.ndarray) -> None:
        """
        Update process noise covariance matrix (tuning hook).
        
        Parameters
        ----------
        Q_diag : np.ndarray
            Diagonal elements of Q (variances) [10-element]
        """
        self.Q = np.diag(Q_diag)
    
    def set_measurement_noise_covariance(self, R_diag: np.ndarray) -> None:
        """
        Update measurement noise covariance matrix (tuning hook).
        
        Parameters
        ----------
        R_diag : np.ndarray
            Diagonal elements of R (variances) [6-element]
        """
        self.R = np.diag(R_diag)
    
    def reset(self, initial_state: Optional[np.ndarray] = None) -> None:
        """
        Reset estimator to initial conditions.
        
        Parameters
        ----------
        initial_state : Optional[np.ndarray], optional
            New initial state vector, if None uses config default
        """
        if initial_state is not None:
            self.x_hat = np.array(initial_state, dtype=float)
        else:
            self.x_hat = np.array(
                self.config.get('initial_state', np.zeros(self.n_states)),
                dtype=float
            )
        
        # Reset covariance to initial value
        initial_P = self.config.get('initial_covariance', None)
        if initial_P is not None:
            self.P = np.array(initial_P, dtype=float)
        else:
            self.P = np.diag([1e-6, 1e-8, 1e-8, 1e-6, 1e-8, 1e-8, 1e-6, 1e-8, 1e-4, 1e-4])
        
        self.innovation = np.zeros(self.n_measurements)
        self.K = np.zeros((self.n_states, self.n_measurements))
        self.iteration = 0
