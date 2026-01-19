"""
Integrated Simulation Runner for Laser Communication Terminal Digital Twin

This module implements the complete closed-loop simulation framework, integrating:
- MuJoCo physics engine for gimbal dynamics
- Custom actuator models (motor + FSM)
- Non-ideal sensor models (encoders, gyros, QPD)
- Extended Kalman Filter for state estimation
- Hierarchical control architecture (coarse + fine)

The simulation uses a fixed-step, multi-rate execution loop for deterministic
behavior and realistic timing separation between subsystems.

Multi-Rate Architecture:
-----------------------
- MuJoCo Dynamics: Δt_sim = 1 ms (fastest)
- Fine Control (FSM): Δt_fine = 1 ms (high-bandwidth)
- Coarse Control: Δt_coarse = 10 ms (low-bandwidth)
- Sensors: Various rates depending on type

Data Flow:
---------
MuJoCo State → Sensors → Estimator → Controllers → Actuators → MuJoCo Forces
"""


import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
import time
from collections import defaultdict
import threading

# Import all subsystem components
from lasercom_digital_twin.core.actuators.motor_models import GimbalMotorModel
from lasercom_digital_twin.core.actuators.fsm_actuator import FSMActuatorModel
from lasercom_digital_twin.core.sensors.sensor_models import AbsoluteEncoder, RateGyro
from lasercom_digital_twin.core.sensors.quadrant_detector import QuadrantDetector
from lasercom_digital_twin.core.estimators.state_estimator import PointingStateEstimator
from lasercom_digital_twin.core.controllers.control_laws import (
    CoarseGimbalController, 
    FeedbackLinearizationController
)
from lasercom_digital_twin.core.controllers.fsm_controller import FSMController
from lasercom_digital_twin.core.optics.optical_chain import TelescopeOptics, FocalPlanePosition
from lasercom_digital_twin.core.coordinate_frames.transformations import OpticalFrameCompensator
from lasercom_digital_twin.core.dynamics.gimbal_dynamics import GimbalDynamics


@dataclass
class SimulationConfig:
    """Configuration for simulation runner."""
    
    # Timing
    dt_sim: float = 0.001          # MuJoCo timestep [s] (1 ms)
    dt_fine: float = 0.001         # FSM control rate [s] (1 ms)
    dt_coarse: float = 0.010       # Coarse control rate [s] (10 ms)
    dt_encoder: float = 0.001      # Encoder sample rate [s] (1 ms)
    dt_gyro: float = 0.001         # Gyro sample rate [s] (1 ms)
    dt_qpd: float = 0.001          # QPD sample rate [s] (1 ms)
    log_period: float = 0.001      # Data logging rate [s] (1 ms)
    
    # Visualization and execution
    enable_visualization: bool = False  # Enable MuJoCo viewer
    real_time_factor: float = 0.0       # 0.0 = fast-as-possible, 1.0 = real-time
    viewer_fps: float = 30.0            # Viewer refresh rate [Hz]
    
    # Deterministic execution
    seed: int = 42
    
    # Target tracking (for testing)
    target_az: float = 0.0         # Target azimuth [rad]
    target_el: float = 0.5         # Target elevation [rad]
    target_enabled: bool = True
    
    # Controller selection
    use_feedback_linearization: bool = False  # Use FL controller instead of PID
    
    # Component configs
    dynamics_config: Dict = field(default_factory=dict)
    motor_config: Dict = field(default_factory=dict)
    fsm_config: Dict = field(default_factory=dict)
    encoder_config: Dict = field(default_factory=dict)
    gyro_config: Dict = field(default_factory=dict)
    qpd_config: Dict = field(default_factory=dict)
    estimator_config: Dict = field(default_factory=dict)
    coarse_controller_config: Dict = field(default_factory=dict)
    feedback_linearization_config: Dict = field(default_factory=dict)
    fsm_controller_config: Dict = field(default_factory=dict)
    optics_config: Dict = field(default_factory=dict)
    frame_config: Dict = field(default_factory=dict)


@dataclass
class SimulationState:
    """Container for current simulation state."""
    
    # Time
    time: float = 0.0
    
    # MuJoCo state (or placeholder dynamics)
    q_az: float = 0.0              # Azimuth angle [rad]
    q_el: float = 0.0              # Elevation angle [rad]
    qd_az: float = 0.0             # Azimuth velocity [rad/s]
    qd_el: float = 0.0             # Elevation velocity [rad/s]
    
    # FSM state
    fsm_tip: float = 0.0           # FSM tip angle [rad]
    fsm_tilt: float = 0.0          # FSM tilt angle [rad]
    
    # Control commands
    torque_az: float = 0.0         # Azimuth motor torque [N·m]
    torque_el: float = 0.0         # Elevation motor torque [N·m]
    fsm_cmd_tip: float = 0.0       # FSM tip command [rad]
    fsm_cmd_tilt: float = 0.0      # FSM tilt command [rad]
    
    # Sensor measurements
    z_enc_az: float = 0.0          # Encoder azimuth [rad]
    z_enc_el: float = 0.0          # Encoder elevation [rad]
    z_gyro_az: float = 0.0         # Gyro azimuth [rad/s]
    z_gyro_el: float = 0.0         # Gyro elevation [rad/s]
    z_qpd_nes_x: float = 0.0       # QPD NES X
    z_qpd_nes_y: float = 0.0       # QPD NES Y
    
    # Estimated state
    est_az: float = 0.0            # Estimated azimuth [rad]
    est_el: float = 0.0            # Estimated elevation [rad]
    est_az_dot: float = 0.0        # Estimated azimuth rate [rad/s]
    est_el_dot: float = 0.0        # Estimated elevation rate [rad/s]
    
    # Performance metrics
    los_error_x: float = 0.0       # True LOS error X (tip) [rad]
    los_error_y: float = 0.0       # True LOS error Y (tilt) [rad]
    fsm_saturated: bool = False    # FSM saturation flag


class DigitalTwinRunner:
    """
    Complete closed-loop simulation runner for laser comm terminal.
    
    This class integrates all subsystems into a deterministic, multi-rate
    simulation loop. It handles:
    1. Physics simulation (MuJoCo or simplified dynamics)
    2. Multi-rate sensor sampling
    3. State estimation (EKF)
    4. Hierarchical control (coarse + fine)
    5. Actuator dynamics
    6. Data logging and telemetry
    
    The simulation enforces strict timing separation between control loops
    to accurately model real-world system behavior.
    
    Usage:
    ------
    >>> config = SimulationConfig(seed=42)
    >>> runner = DigitalTwinRunner(config)
    >>> results = runner.run_simulation(duration=5.0)
    >>> print(f"Final LOS error: {results['los_error_rms']:.2e} rad")
    """
    
    def __init__(self, config: SimulationConfig):
        """
        Initialize the digital twin simulation runner.
        
        Parameters
        ----------
        config : SimulationConfig
            Complete configuration for all subsystems and timing
        """
        self.config = config
        
        # Set deterministic seed
        self.rng = np.random.default_rng(config.seed)
        
        # Threading lock for MuJoCo data access
        self.mj_data_lock = threading.Lock()
        
        # Initialize timing counters
        self.time: float = 0.0
        self.iteration: int = 0
        self.last_coarse_update: float = 0.0
        self.last_fine_update: float = 0.0
        self.last_encoder_update: float = 0.0
        self.last_gyro_update: float = 0.0
        self.last_qpd_update: float = 0.0
        self.last_log_time: float = 0.0
        
        # Set physical parameters from config
        self.pan_mass = getattr(config, 'pan_mass', 0.5)
        self.tilt_mass = getattr(config, 'tilt_mass', 0.25)
        self.cm_r = getattr(config, 'cm_r', 0.002)
        self.cm_h = getattr(config, 'cm_h', 0.005)
        self.gravity = getattr(config, 'gravity', 9.81)
        self.friction_az = getattr(config, 'friction_az', 0.1)
        self.friction_el = getattr(config, 'friction_el', 0.1)
        
        # Initialize GimbalDynamics
        self.dynamics = GimbalDynamics(
            pan_mass=self.pan_mass,
            tilt_mass=self.tilt_mass,
            cm_r=self.cm_r,
            cm_h=self.cm_h,
            gravity=self.gravity
        )
        
        # Compute inertias from mass matrix at zero position (q=0)
        # For a 2-DOF gimbal, inertia_az = M[0,0] and inertia_el = M[1,1] at q=0
        M_at_zero = self.dynamics.get_mass_matrix(np.zeros(2))  # Mass matrix at [0, 0]
        self.inertia_az = M_at_zero[0, 0]  # Azimuth inertia [kg·m²]
        self.inertia_el = M_at_zero[1, 1]  # Elevation inertia [kg·m²]

        # Initialize MuJoCo or placeholder dynamics
        self._init_dynamics()
        
        # Initialize actuators
        self._init_actuators()
        
        # Initialize sensors
        self._init_sensors()
        
        # Initialize optical chain
        self._init_optics()
        
        # Initialize estimator
        self._init_estimator()
        
        # Initialize controllers
        self._init_controllers()
        
        # Initialize data logging
        self._init_logging()
        
        # Current state
        self.state = SimulationState()
        
    def _init_dynamics(self) -> None:
        """
        Initialize dynamics model (MuJoCo or simplified).
        
        In production, this would load gimbal_cpa_model.xml into MuJoCo.
        For now, implements simplified rigid-body dynamics as placeholder.
        """
        # MuJoCo interface (placeholder - would use mujoco library)
        self.use_mujoco = False  # Set to True when MuJoCo model is available
        
        if self.use_mujoco:
            try:
                import mujoco
                from pathlib import Path
                
                # Robust path resolution relative to package root
                # Assumes 'models' directory is at package root level
                current_file = Path(__file__).resolve()
                package_root = current_file.parent.parent.parent
                model_path = package_root / "models" / "gimbal_cpa_model.xml"
                
                if not model_path.exists():
                    raise FileNotFoundError(
                        f"CRITICAL: MuJoCo model not found at expected path: {model_path}. "
                        "Ensure the model description file exists."
                    )
                
                # Load physics model and create data structure
                self.mj_model = mujoco.MjModel.from_xml_path(str(model_path))
                self.mj_data = mujoco.MjData(self.mj_model)
                
                # Create separate data structure for visualization to avoid threading conflicts
                self.mj_data_vis = mujoco.MjData(self.mj_model)
                
                # Initialize viewer-related attributes
                self.viewer = None
                
            except ImportError:
                print("Warning: 'mujoco' library not installed. Falling back to simplified dynamics.")
                self.use_mujoco = False
            except Exception as e:
                raise RuntimeError(f"Failed to initialize MuJoCo dynamics: {str(e)}") from e
        
        if not self.use_mujoco:
            # Simplified dynamics model
            self.inertia_az = 0.20  # kg·m²
            self.inertia_el = 0.15  # kg·m²
            self.friction_az = 0.05  # N·m·s/rad
            self.friction_el = 0.05  # N·m·s/rad
            
            # State variables
            self.q_az = 0.0
            self.q_el = 0.0
            self.qd_az = 0.0
            self.qd_el = 0.0
    
    def _init_actuators(self) -> None:
        """Initialize motor and FSM actuator models."""
        # Gimbal motors (Az/El)
        motor_config = self.config.motor_config or {
            'R': 2.0,
            'L': 0.05,  # Increased from 0.005 to prevent numerical instability
            'K_t': 0.5,
            'K_e': 0.5,
            'tau_max': 10.0,
            'tau_min': -10.0,
            'cogging_amplitude': 0.05,
            'cogging_frequency': 10,
            'seed': self.config.seed
        }
        
        self.motor_az = GimbalMotorModel(motor_config)
        self.motor_el = GimbalMotorModel(motor_config)
        
        # Fast Steering Mirror
        fsm_config = self.config.fsm_config or {
            'natural_frequency': 500.0,
            'damping_ratio': 0.7,
            'max_angle': 0.01,
            'hysteresis_amplitude': 5e-5,
            'max_rate': 1.0,
            'seed': self.config.seed + 1
        }
        
        self.fsm = FSMActuatorModel(fsm_config)
    
    def _init_sensors(self) -> None:
        """Initialize all sensor models."""
        # Absolute encoders (Az/El)
        encoder_config = self.config.encoder_config or {
            'resolution_bits': 20,
            'noise_std': 1e-6,
            'bias': 0.0,
            'seed': self.config.seed + 2
        }
        
        self.encoder_az = AbsoluteEncoder(encoder_config)
        self.encoder_el = AbsoluteEncoder({**encoder_config, 'seed': self.config.seed + 3})
        
        # Rate gyros (Az/El)
        gyro_config = self.config.gyro_config or {
            'arw_std': 1e-5,
            'bias_std': 1e-6,
            'bias_instability': 1e-7,
            'latency': 0.002,
            'seed': self.config.seed + 4
        }
        
        self.gyro_az = RateGyro(gyro_config)
        self.gyro_el = RateGyro({**gyro_config, 'seed': self.config.seed + 5})
        
        # Quadrant Photo Detector
        qpd_config = self.config.qpd_config or {
            'sensitivity': 2000.0,
            'linear_range': 0.001,
            'noise_std': 1e-4,
            'bias_x': 0.0,
            'bias_y': 0.0,
            'nonlinearity_coeff': 100.0,
            'spot_size_um': 50.0,
            'seed': self.config.seed + 6
        }
        
        self.qpd = QuadrantDetector(qpd_config)
    
    def _init_optics(self) -> None:
        """Initialize optical chain and coordinate transformations."""
        # Telescope optics
        optics_config = self.config.optics_config or {
            'focal_length_m': 1.5,
            'aperture_diameter_m': 0.3,
            'wavelength_m': 1550e-9,
            'detector_size_um': 1000.0
        }
        
        self.optics = TelescopeOptics(optics_config)
        
        # Optical frame compensator
        frame_config = self.config.frame_config or {
            'site_latitude_deg': 35.0,
            'epsilon_az': 0.0,
            'epsilon_el': 0.0
        }
        
        self.frame_compensator = OpticalFrameCompensator(frame_config)
    
    def _init_estimator(self) -> None:
        """Initialize Extended Kalman Filter."""
        estimator_config = self.config.estimator_config or {
            'initial_state': np.zeros(10),
            'inertia_az': self.inertia_az,
            'inertia_el': self.inertia_el,
            'friction_coeff_az': self.friction_az,
            'friction_coeff_el': self.friction_el,
            'process_noise_std': [1e-8, 1e-6, 1e-9, 1e-8, 1e-6, 1e-9, 1e-7, 1e-6, 1e-4, 1e-4],
            'measurement_noise_std': [2.4e-5, 2.4e-5, 1e-6, 1e-6, 1e-4, 1e-4]
        }
        
        self.estimator = PointingStateEstimator(estimator_config)
    
    def _init_controllers(self) -> None:
        """Initialize hierarchical control system."""
        # Initialize gimbal dynamics model (needed for feedback linearization)
        dynamics_config = self.config.dynamics_config or {
            'pan_mass': 0.5,
            'tilt_mass': 0.25,
            'cm_r': 0.02,
            'cm_h': 0.005,
            'gravity': 9.81
        }
        self.gimbal_dynamics = GimbalDynamics(**dynamics_config)
        
        # Select coarse controller type
        if self.config.use_feedback_linearization:
            # Feedback Linearization Controller
            fl_config = self.config.feedback_linearization_config or {
                'kp': [100.0, 100.0],
                'kd': [20.0, 20.0],
                'tau_max': [10.0, 10.0],
                'tau_min': [-10.0, -10.0]
            }
            self.coarse_controller = FeedbackLinearizationController(
                fl_config, 
                self.gimbal_dynamics
            )
            print("INFO: Using Feedback Linearization Controller")
        else:
            # Coarse gimbal controller (Level 1) - Standard PID
            coarse_config = self.config.coarse_controller_config or {
                'kp': 50.0,
                'ki': 5.0,
                'kd': 2.0,
                'anti_windup_gain': 1.0,
                'tau_rate_limit': 50.0,
                'setpoint_filter_wn': 10.0,
                'setpoint_filter_zeta': 0.7
            }
            self.coarse_controller = CoarseGimbalController(coarse_config)
            print("INFO: Using Standard PID Controller")
        
        # Fine FSM controller (Level 2)
        fsm_controller_config = self.config.fsm_controller_config or {
            'kp_tip': 100.0,
            'kp_tilt': 100.0,
            'ki_tip': 50.0,
            'ki_tilt': 50.0,
            'max_angle': 0.01,
            'enable_feedforward': True,
            'enable_high_pass_filter': False
        }
        
        self.fsm_controller = FSMController(fsm_controller_config)
    
    def _init_logging(self) -> None:
        """Initialize data logging infrastructure."""
        self.log_data: Dict[str, List] = defaultdict(list)
        
        # Define signals to log
        self.log_signals = [
            'time',
            'q_az', 'q_el', 'qd_az', 'qd_el',
            'est_az', 'est_el', 'est_az_dot', 'est_el_dot',
            'torque_az', 'torque_el',
            'fsm_tip', 'fsm_tilt', 'fsm_cmd_tip', 'fsm_cmd_tilt',
            'fsm_saturated',
            'z_enc_az', 'z_enc_el', 'z_gyro_az', 'z_gyro_el',
            'z_qpd_nes_x', 'z_qpd_nes_y',
            'los_error_x', 'los_error_y',
            'target_az', 'target_el'
        ]
    
    def _step_dynamics(self, dt: float) -> None:
        """
        Step the dynamics forward (MuJoCo or simplified).
        
        Applies actuator torques and integrates equations of motion.
        
        Parameters
        ----------
        dt : float
            Timestep [s]
        """
        if self.use_mujoco:
            with self.mj_data_lock:
                # MuJoCo integration
                self.mj_data.ctrl[0] = self.state.torque_az
                self.mj_data.ctrl[1] = self.state.torque_el
                
                # Step physics forward
                import mujoco
                mujoco.mj_step(self.mj_model, self.mj_data)
                
                # Extract state
                self.q_az = self.mj_data.qpos[0]
                self.q_el = self.mj_data.qpos[1] 
                self.qd_az = self.mj_data.qvel[0]
                self.qd_el = self.mj_data.qvel[1]
                
                # Update state container
                self.state.q_az = self.q_az
                self.state.q_el = self.q_el
                self.state.qd_az = self.qd_az
                self.state.qd_el = self.qd_el
        else:
            # -----------------------------------------------------------
            # High-Fidelity Coupled Dynamics (Lagrangian Formulation)
            # -----------------------------------------------------------
            # Solves: M(q)q̈ + C(q,q̇)q̇ + G(q) = τ_net
            # where τ_net = τ_motor - D·q̇ (Friction)

            # 1. State Vectors
            q = np.array([self.q_az, self.q_el])
            dq = np.array([self.qd_az, self.qd_el])
            
            # 2. Input Torques (with Friction Compensation)
            # GimbalDynamics expects the net torque (or handles G internally? No, G is in LHS)
            # compute_forward_dynamics solves M*qdd = tau - C*dq - G
            # We supply tau = tau_motor - Friction
            tau_motor = np.array([self.state.torque_az, self.state.torque_el])
            tau_friction = np.array([
                self.friction_az * self.qd_az,
                self.friction_el * self.qd_el
            ])
            tau_net = tau_motor - tau_friction

            # 3. Compute Accelerations via Inverted Mass Matrix
            # q̈ = M⁻¹(τ_net - C·q̇ - G)
            q_dd = self.dynamics.compute_forward_dynamics(q, dq, tau_net)

            # 4. Numerical Integration (Semi-Implicit / Symplectic Euler)
            # Preferred for Hamiltonian systems/mechanical stability
            
            # Update Velocities first
            self.qd_az += q_dd[0] * dt
            self.qd_el += q_dd[1] * dt
            
            # Update Positions using new Velocities
            self.q_az += self.qd_az * dt
            self.q_el += self.qd_el * dt
            
            # 5. Update Simulation State
            self.state.q_az = self.q_az
            self.state.q_el = self.q_el
            self.state.qd_az = self.qd_az
            self.state.qd_el = self.qd_el
    
    def _sample_sensors(self) -> None:
        """
        Sample all sensors at their respective rates.
        
        Checks timing and updates sensor measurements when appropriate.
        """
        # Encoder sampling
        if (self.time - self.last_encoder_update) >= self.config.dt_encoder:
            self.state.z_enc_az = self.encoder_az.measure(self.state.q_az)
            self.state.z_enc_el = self.encoder_el.measure(self.state.q_el)
            self.last_encoder_update = self.time
        
        # Gyro sampling
        if (self.time - self.last_gyro_update) >= self.config.dt_gyro:
            self.state.z_gyro_az = self.gyro_az.measure(self.state.qd_az)
            self.state.z_gyro_el = self.gyro_el.measure(self.state.qd_el)
            self.gyro_az.update(self.config.dt_gyro)
            self.gyro_el.update(self.config.dt_gyro)
            self.last_gyro_update = self.time
        
        # QPD sampling (requires optical chain calculation)
        if (self.time - self.last_qpd_update) >= self.config.dt_qpd:
            self._update_qpd_measurement()
            self.last_qpd_update = self.time
    
    def _update_qpd_measurement(self) -> None:
        """
        Compute QPD measurement through optical chain.
        
        Flow: Gimbal angles → LOS error → FSM correction → Focal plane → QPD
        """
        # Compute pointing error
        if self.config.target_enabled:
            error_az = self.config.target_az - self.state.q_az
            error_el = self.config.target_el - self.state.q_el
        else:
            error_az = 0.0
            error_el = 0.0
        
        # Store true LOS error
        self.state.los_error_x = error_az
        self.state.los_error_y = error_el
        
        # Apply FSM correction to beam pointing
        # FSM deflects beam by 2× mirror angle
        corrected_error_az = error_az - 2.0 * self.state.fsm_tip
        corrected_error_el = error_el - 2.0 * self.state.fsm_tilt
        
        # Map to focal plane
        focal_pos = self.optics.compute_spot_position(
            corrected_error_az,
            corrected_error_el,
            fsm_tip_rad=0.0,  # Already applied above
            fsm_tilt_rad=0.0
        )
        
        # QPD measures from focal plane position
        nes_x, nes_y = self.qpd.measure_from_focal_plane(focal_pos.x_um, focal_pos.y_um)
        self.state.z_qpd_nes_x = nes_x
        self.state.z_qpd_nes_y = nes_y
    
    def _update_estimator(self) -> None:
        """
        Update state estimator (EKF prediction + correction).
        
        Runs at coarse control rate.
        """
        # Build control input vector
        u = np.array([self.state.torque_az, self.state.torque_el])
        
        # Build measurement dictionary
        measurements = {
            'theta_az_enc': self.state.z_enc_az,
            'theta_el_enc': self.state.z_enc_el,
            'theta_dot_az_gyro': self.state.z_gyro_az,
            'theta_dot_el_gyro': self.state.z_gyro_el,
            'nes_x_qpd': self.state.z_qpd_nes_x,
            'nes_y_qpd': self.state.z_qpd_nes_y
        }
        
        # EKF step
        self.estimator.step(u, measurements, self.config.dt_coarse)
        
        # Extract fused state
        fused_state = self.estimator.get_fused_state()
        self.state.est_az = fused_state['theta_az']
        self.state.est_el = fused_state['theta_el']
        self.state.est_az_dot = fused_state['theta_dot_az']
        self.state.est_el_dot = fused_state['theta_dot_el']
    
    def _update_coarse_controller(self) -> None:
        """
        Update coarse gimbal controller (Level 1).
        
        Implements the signal flow architecture:
        1. Get filtered state from EKF
        2. Compute control (PID or Feedback Linearization)
        3. Send torque commands to motors
        
        Runs at coarse control rate (~10 Hz).
        """
        # Setpoints (target tracking)
        setpoint = np.array([self.config.target_az, self.config.target_el])
        setpoint_vel = np.zeros(2)  # Assume stationary target
        
        # Estimated state from EKF (already updated)
        position_est = np.array([self.state.est_az, self.state.est_el])
        velocity_est = np.array([self.state.est_az_dot, self.state.est_el_dot])
        
        # Compute control based on controller type
        if self.config.use_feedback_linearization:
            # Feedback Linearization Controller
            # Get full state estimate dictionary from EKF
            fused_state = self.estimator.get_fused_state()
            
            tau_cmd, meta = self.coarse_controller.compute_control(
                q_ref=setpoint,
                dq_ref=setpoint_vel,
                state_estimate=fused_state,
                dt=self.config.dt_coarse,
                ddq_ref=None  # Assume zero acceleration reference
            )
            
            # For FSM feedforward, use tracking error from metadata
            residual_error = meta.get('error', setpoint - position_est)
        else:
            # Standard PID Controller
            tau_cmd, meta = self.coarse_controller.compute_control(
                setpoint,
                position_est,
                self.config.dt_coarse,
                velocity_estimate=velocity_est
            )
            
            # Get residual error for FSM feedforward
            residual_error = self.coarse_controller.get_residual_error_for_fsm(
                setpoint, position_est
            )
        
        # Extract Az/El torques
        tau_az = tau_cmd[0]
        tau_el = tau_cmd[1]
        
        # Store for FSM controller
        self.coarse_residual = residual_error
        
        # Update motor models with voltage commands
        # Convert torque command to voltage (simplified: τ = K_t * I, V = R * I)
        # V ≈ τ / K_t * R (ignoring inductance for steady-state)
        voltage_az = tau_az
        voltage_el = tau_el
        
        # Step motor models
        actual_tau_az = self.motor_az.step(
            voltage_az,
            self.state.qd_az,
            self.state.q_az,
            self.config.dt_coarse
        )
        actual_tau_el = self.motor_el.step(
            voltage_el,
            self.state.qd_el,
            self.state.q_el,
            self.config.dt_coarse
        )
        
        # Update state
        self.state.torque_az = actual_tau_az
        self.state.torque_el = actual_tau_el
    
    def _update_fine_controller(self) -> None:
        """
        Update fine FSM controller (Level 2).
        
        Runs at fine control rate (~1 kHz).
        """
        # QPD error signal (already in tip/tilt)
        qpd_error = np.array([self.state.z_qpd_nes_x, self.state.z_qpd_nes_y])
        
        # Compute FSM command with feedforward (returns tuple: command, metadata)
        fsm_cmd, fsm_meta = self.fsm_controller.compute_control(
            qpd_error,
            self.config.dt_fine,
            coarse_residual=self.coarse_residual if hasattr(self, 'coarse_residual') else None
        )
        
        # Store commands (extract scalars)
        tip_val = fsm_cmd[0].item() if hasattr(fsm_cmd[0], 'item') else float(fsm_cmd[0])
        tilt_val = fsm_cmd[1].item() if hasattr(fsm_cmd[1], 'item') else float(fsm_cmd[1])
        
        self.state.fsm_cmd_tip = tip_val
        self.state.fsm_cmd_tilt = tilt_val
        
        # Check saturation
        self.state.fsm_saturated = self.fsm_controller.is_saturated()
        
        # Update FSM actuator model
        self.fsm.step(tip_val, tilt_val, self.config.dt_fine)
        
        # Get actual FSM position
        fsm_state = self.fsm.get_state()
        self.state.fsm_tip = fsm_state['alpha']  # Tip = alpha
        self.state.fsm_tilt = fsm_state['beta']  # Tilt = beta
    
    def _log_data(self) -> None:
        """
        Log current state to telemetry buffer.
        
        Logs at specified log_period rate.
        """
        if (self.time - self.last_log_time) >= self.config.log_period:
            # Log all signals
            self.log_data['time'].append(self.time)
            self.log_data['q_az'].append(self.state.q_az)
            self.log_data['q_el'].append(self.state.q_el)
            self.log_data['qd_az'].append(self.state.qd_az)
            self.log_data['qd_el'].append(self.state.qd_el)
            self.log_data['est_az'].append(self.state.est_az)
            self.log_data['est_el'].append(self.state.est_el)
            self.log_data['est_az_dot'].append(self.state.est_az_dot)
            self.log_data['est_el_dot'].append(self.state.est_el_dot)
            self.log_data['torque_az'].append(self.state.torque_az)
            self.log_data['torque_el'].append(self.state.torque_el)
            self.log_data['fsm_tip'].append(self.state.fsm_tip)
            self.log_data['fsm_tilt'].append(self.state.fsm_tilt)
            self.log_data['fsm_cmd_tip'].append(self.state.fsm_cmd_tip)
            self.log_data['fsm_cmd_tilt'].append(self.state.fsm_cmd_tilt)
            self.log_data['fsm_saturated'].append(self.state.fsm_saturated)
            self.log_data['z_enc_az'].append(self.state.z_enc_az)
            self.log_data['z_enc_el'].append(self.state.z_enc_el)
            self.log_data['z_gyro_az'].append(self.state.z_gyro_az)
            self.log_data['z_gyro_el'].append(self.state.z_gyro_el)
            self.log_data['z_qpd_nes_x'].append(self.state.z_qpd_nes_x)
            self.log_data['z_qpd_nes_y'].append(self.state.z_qpd_nes_y)
            self.log_data['los_error_x'].append(self.state.los_error_x)
            self.log_data['los_error_y'].append(self.state.los_error_y)
            self.log_data['target_az'].append(self.config.target_az)
            self.log_data['target_el'].append(self.config.target_el)
            
            self.last_log_time = self.time
    
    def run_single_step(self) -> SimulationState:
        """
        Execute a single simulation step across all subsystems.
        
        This method handles multi-rate execution, dynamics integration,
        sensor sampling, estimation, and control updates for one dt_sim step.
        """
        # 1. Step dynamics forward
        self._step_dynamics(self.config.dt_sim)
        
        # 2. Sample sensors (multi-rate)
        self._sample_sensors()
        
        # 3. Update estimator and coarse controller (at coarse rate)
        # Using epsilon for floating point comparison robustness
        if (self.time - self.last_coarse_update) >= (self.config.dt_coarse - 1e-9):
            self._update_estimator()
            self._update_coarse_controller()
            self.last_coarse_update = self.time
        
        # 4. Update fine controller (at fine rate)
        if (self.time - self.last_fine_update) >= (self.config.dt_fine - 1e-9):
            self._update_fine_controller()
            self.last_fine_update = self.time
        
        # 5. Log data
        self._log_data()
        
        # Increment internal time and iteration
        self.iteration += 1
        self.time = self.iteration * self.config.dt_sim
        
        return self.state

    def run_simulation(self, duration: float) -> Dict:
        """
        Execute the complete multi-rate closed-loop simulation.
        
        This is the main entry point for running the digital twin. It orchestrates
        all subsystems in a fixed-step loop with proper timing separation.
        
        CRITICAL: Ensures deterministic termination based on simulated time,
        not viewer status or wall-clock time.
        
        Parameters
        ----------
        duration : float
            Total simulation time [s]
            
        Returns
        -------
        Dict
            Complete logged telemetry and performance summary
        """
        print(f"Starting simulation for {duration:.2f} seconds...")
        print(f"  dt_sim: {self.config.dt_sim*1e3:.2f} ms")
        print(f"  dt_coarse: {self.config.dt_coarse*1e3:.2f} ms")
        print(f"  dt_fine: {self.config.dt_fine*1e3:.2f} ms")
        print(f"  Target: Az={np.rad2deg(self.config.target_az):.2f}°, El={np.rad2deg(self.config.target_el):.2f}°")
        print(f"  Visualization: {'Enabled' if self.config.enable_visualization else 'Disabled'}")
        
        # Initialize visualization if requested (OUTSIDE main loop)
        viewer = None
        if self.config.enable_visualization and self.use_mujoco:
            try:
                import mujoco.viewer
                # Use the visualization data structure for the viewer
                viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data_vis)
                print(f"  Viewer initialized at {self.config.viewer_fps:.0f} FPS")
            except Exception as e:
                print(f"Warning: Failed to initialize viewer: {e}")
                viewer = None
        
        # Timing setup
        start_time = time.perf_counter()
        last_viewer_update = 0.0
        last_realtime_check = 0.0
        viewer_dt = 1.0 / self.config.viewer_fps if self.config.viewer_fps > 0 else 0.1
        
        # Reset internal time for fresh run
        self.time = 0.0
        self.iteration = 0
        if self.use_mujoco:
            self.mj_data.time = 0.0
            # Reset visualization data to match
            if hasattr(self, 'mj_data_vis'):
                self.mj_data_vis.time = 0.0
        
        try:
            # CRITICAL: Main simulation loop with DETERMINISTIC TERMINATION
            # Condition: simulated time < duration (NOT viewer status)
            while True:
                # Check termination condition: SIMULATED TIME ONLY
                current_sim_time = self.mj_data.time if self.use_mujoco else self.time
                if current_sim_time >= duration:
                    break
                
                # Execute single simulation step
                self.run_single_step()
                #lets work with mojoco
                
                # NON-BLOCKING viewer update (if enabled)
                if viewer is not None and (current_sim_time - last_viewer_update) >= viewer_dt:
                    try:
                        # Thread-safe copy of simulation data to visualization data
                        with self.mj_data_lock:
                            # Manual copy of key data fields since mj_copyData may not be available
                            self.mj_data_vis.qpos[:] = self.mj_data.qpos[:]
                            self.mj_data_vis.qvel[:] = self.mj_data.qvel[:]
                            if hasattr(self.mj_data, 'act') and hasattr(self.mj_data_vis, 'act'):
                                self.mj_data_vis.act[:] = self.mj_data.act[:]
                            self.mj_data_vis.time = self.mj_data.time
                        
                        # Update viewer with copied state (thread-safe)
                        viewer.sync()
                        last_viewer_update = current_sim_time
                        
                        # Check if user closed viewer window (but don't terminate on this)
                        # This is informational only - simulation continues regardless
                        if not viewer.is_running():
                            print("  Viewer closed by user (simulation continuing...)")
                            viewer = None
                    except Exception as e:
                        # Viewer error - continue simulation without visualization
                        print(f"  Viewer error: {e} (continuing without visualization)")
                        viewer = None
                
                # Optional real-time constraint (for debugging, disabled for CI/MC)
                if self.config.real_time_factor > 0 and (current_sim_time - last_realtime_check) >= 0.1:
                    elapsed_wall = time.perf_counter() - start_time
                    target_wall = current_sim_time / self.config.real_time_factor
                    if elapsed_wall < target_wall:
                        time.sleep(min(0.001, target_wall - elapsed_wall))  # Max 1ms sleep
                    last_realtime_check = current_sim_time
                
                # Progress indicator
                if self.iteration % max(1, int(duration / (self.config.dt_sim * 10))) == 0 and self.iteration > 0:
                    progress = 100.0 * current_sim_time / duration
                    print(f"  Progress: {progress:.0f}% (t={current_sim_time:.2f}s)")
        
        finally:
            # CRITICAL: Cleanup - close viewer only after user closes it
            if viewer is not None:
                print("  Simulation complete. Close the viewer window to continue.")
                try:
                    while viewer.is_running():
                        time.sleep(0.1)
                    viewer.close()
                    print("  Viewer closed by user")
                except Exception:
                    pass  # Ignore cleanup errors
        
        # Final metrics
        elapsed_time = time.perf_counter() - start_time
        final_sim_time = self.mj_data.time if self.use_mujoco else self.time
        print(f"Simulation complete: {final_sim_time:.3f} simulated seconds")
        print(f"  Wall-clock time: {elapsed_time:.2f} seconds")
        print(f"  Real-time factor: {final_sim_time/elapsed_time:.1f}x")
        print(f"  Total iterations: {self.iteration}")
        
        # Compute performance summary
        results = self._compute_summary()
        self.plot_results()
        return results
    
    def _compute_summary(self) -> Dict:
        """
        Compute performance metrics from logged data.
        
        Returns
        -------
        Dict
            Performance summary with RMS errors, statistics
        """
        # Convert lists to arrays
        log_arrays = {key: np.array(val) for key, val in self.log_data.items()}
        
        # Compute RMS errors
        los_error_rms_x = np.sqrt(np.mean(log_arrays['los_error_x']**2))
        los_error_rms_y = np.sqrt(np.mean(log_arrays['los_error_y']**2))
        los_error_rms = np.sqrt(los_error_rms_x**2 + los_error_rms_y**2)
        
        # Estimation error
        est_error_az = log_arrays['q_az'] - log_arrays['est_az']
        est_error_el = log_arrays['q_el'] - log_arrays['est_el']
        est_error_rms = np.sqrt(np.mean(est_error_az**2 + est_error_el**2))
        
        # FSM saturation statistics
        fsm_saturation_percent = 100.0 * np.sum(log_arrays['fsm_saturated']) / len(log_arrays['fsm_saturated'])
        
        # Control effort
        torque_rms_az = np.sqrt(np.mean(log_arrays['torque_az']**2))
        torque_rms_el = np.sqrt(np.mean(log_arrays['torque_el']**2))
        
        summary = {
            'log_data': self.log_data,
            'log_arrays': log_arrays,
            'n_samples': len(log_arrays['time']),
            'duration': log_arrays['time'][-1] if len(log_arrays['time']) > 0 else 0.0,
            'los_error_rms': los_error_rms,
            'los_error_rms_x': los_error_rms_x,
            'los_error_rms_y': los_error_rms_y,
            'los_error_final': np.sqrt(log_arrays['los_error_x'][-1]**2 + log_arrays['los_error_y'][-1]**2),
            'est_error_rms': est_error_rms,
            'fsm_saturation_percent': fsm_saturation_percent,
            'torque_rms_az': torque_rms_az,
            'torque_rms_el': torque_rms_el,
            'final_az': log_arrays['q_az'][-1],
            'final_el': log_arrays['q_el'][-1]
        }
        
        return summary
    
    def plot_results(self) -> None:
        """
        Generate high-fidelity diagnostic plots for closed-loop simulation telemetry.
        
        Creates separate figures for each subsystem:
        - Figure 1: Gimbal Position (q_az, q_el with commands)
        - Figure 2: Gimbal Velocity (qd_az, qd_el)
        - Figure 3: Control Torques (torque_az, torque_el)
        - Figure 4: FSM State (fsm_tip, fsm_tilt, fsm_cmd_tip, fsm_cmd_tilt)
        - Figure 5: Sensor Measurements (encoders, gyros)
        - Figure 6: QPD Measurements (z_qpd_nes_x, z_qpd_nes_y)
        - Figure 7: LOS Errors (los_error_x, los_error_y, total)
        - Figure 8: Estimated State (est_az, est_el, est_az_dot, est_el_dot)
        
        Suitable for design reviews, performance validation, and control tuning.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Warning: matplotlib not installed. Cannot generate plots.")
            return
        
        # Check if data exists
        if not self.log_data or len(self.log_data.get('time', [])) == 0:
            print("Warning: No simulation data to plot. Run simulation first.")
            return
        
        # Extract time-series data (vectorized for efficiency)
        t = np.array(self.log_data['time'])
        
        # Color scheme
        color_az = '#1f77b4'    # Blue
        color_el = '#d62728'    # Red
        color_cmd = '#2ca02c'   # Green
        color_x = '#ff7f0e'     # Orange
        color_y = '#9467bd'     # Purple
        
        # ===================================================================
        # FIGURE 1: Gimbal Position (q_az, q_el with commands)
        # ===================================================================
        fig1, (ax1a, ax1b) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Azimuth Position
        ax1a.plot(t, np.rad2deg(np.array(self.log_data['q_az'])), 
                  color=color_az, linewidth=2, label='q_az Actual', alpha=0.9)
        ax1a.plot(t, np.rad2deg(np.array(self.log_data['target_az'])), 
                  color=color_cmd, linewidth=1.5, linestyle='--', label='q_az Command', alpha=0.7)
        ax1a.set_ylabel('Azimuth Angle [deg]', fontsize=11, fontweight='bold')
        ax1a.set_title('Gimbal Azimuth Position', fontsize=12, fontweight='bold')
        ax1a.legend(loc='best', fontsize=9)
        ax1a.grid(True, alpha=0.3, linestyle=':')
        
        # Elevation Position
        ax1b.plot(t, np.rad2deg(np.array(self.log_data['q_el'])), 
                  color=color_el, linewidth=2, label='q_el Actual', alpha=0.9)
        ax1b.plot(t, np.rad2deg(np.array(self.log_data['target_el'])), 
                  color=color_cmd, linewidth=1.5, linestyle='--', label='q_el Command', alpha=0.7)
        ax1b.set_ylabel('Elevation Angle [deg]', fontsize=11, fontweight='bold')
        ax1b.set_xlabel('Time [s]', fontsize=11, fontweight='bold')
        ax1b.set_title('Gimbal Elevation Position', fontsize=12, fontweight='bold')
        ax1b.legend(loc='best', fontsize=9)
        ax1b.grid(True, alpha=0.3, linestyle=':')
        
        fig1.suptitle('Gimbal Position Tracking', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # ===================================================================
        # FIGURE 2: Gimbal Velocity (qd_az, qd_el)
        # ===================================================================
        fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Azimuth Velocity
        ax2a.plot(t, np.rad2deg(np.array(self.log_data['qd_az'])), 
                  color=color_az, linewidth=1.5, label='qd_az', alpha=0.9)
        ax2a.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
        ax2a.set_ylabel('Azimuth Rate [deg/s]', fontsize=11, fontweight='bold')
        ax2a.set_title('Gimbal Azimuth Velocity', fontsize=12, fontweight='bold')
        ax2a.legend(loc='best', fontsize=9)
        ax2a.grid(True, alpha=0.3, linestyle=':')
        
        # Elevation Velocity
        ax2b.plot(t, np.rad2deg(np.array(self.log_data['qd_el'])), 
                  color=color_el, linewidth=1.5, label='qd_el', alpha=0.9)
        ax2b.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
        ax2b.set_ylabel('Elevation Rate [deg/s]', fontsize=11, fontweight='bold')
        ax2b.set_xlabel('Time [s]', fontsize=11, fontweight='bold')
        ax2b.set_title('Gimbal Elevation Velocity', fontsize=12, fontweight='bold')
        ax2b.legend(loc='best', fontsize=9)
        ax2b.grid(True, alpha=0.3, linestyle=':')
        
        fig2.suptitle('Gimbal Angular Velocities', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # ===================================================================
        # FIGURE 3: Control Torques (torque_az, torque_el)
        # ===================================================================
        fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        tau_max = 10.0  # From motor config
        
        # Azimuth Torque
        ax3a.plot(t, np.array(self.log_data['torque_az']), 
                  color=color_az, linewidth=1.5, label='torque_az', alpha=0.9)
        ax3a.axhline(tau_max, color='red', linewidth=1.0, linestyle=':', 
                     alpha=0.6, label='Saturation Limit')
        ax3a.axhline(-tau_max, color='red', linewidth=1.0, linestyle=':', alpha=0.6)
        ax3a.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
        ax3a.set_ylabel('Azimuth Torque [N·m]', fontsize=11, fontweight='bold')
        ax3a.set_title('Azimuth Motor Control Effort', fontsize=12, fontweight='bold')
        ax3a.legend(loc='best', fontsize=9)
        ax3a.grid(True, alpha=0.3, linestyle=':')
        
        # Elevation Torque
        ax3b.plot(t, np.array(self.log_data['torque_el']), 
                  color=color_el, linewidth=1.5, label='torque_el', alpha=0.9)
        ax3b.axhline(tau_max, color='red', linewidth=1.0, linestyle=':', 
                     alpha=0.6, label='Saturation Limit')
        ax3b.axhline(-tau_max, color='red', linewidth=1.0, linestyle=':', alpha=0.6)
        ax3b.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
        ax3b.set_ylabel('Elevation Torque [N·m]', fontsize=11, fontweight='bold')
        ax3b.set_xlabel('Time [s]', fontsize=11, fontweight='bold')
        ax3b.set_title('Elevation Motor Control Effort', fontsize=12, fontweight='bold')
        ax3b.legend(loc='best', fontsize=9)
        ax3b.grid(True, alpha=0.3, linestyle=':')
        
        fig3.suptitle('Motor Control Torques', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # ===================================================================
        # FIGURE 4: FSM State (fsm_tip, fsm_tilt, commands, saturation)
        # ===================================================================
        fig4, ((ax4a, ax4b), (ax4c, ax4d)) = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
        
        fsm_max = 10000  # 10 mrad = 10000 µrad
        
        # FSM Tip Position
        ax4a.plot(t, np.array(self.log_data['fsm_tip']) * 1e6, 
                  color=color_az, linewidth=1.5, label='fsm_tip', alpha=0.9)
        ax4a.axhline(fsm_max, color='orange', linewidth=1.0, linestyle=':', 
                     alpha=0.6, label='FSM Limit')
        ax4a.axhline(-fsm_max, color='orange', linewidth=1.0, linestyle=':', alpha=0.6)
        ax4a.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
        ax4a.set_ylabel('Tip Angle [µrad]', fontsize=10, fontweight='bold')
        ax4a.set_title('FSM Tip Position', fontsize=11, fontweight='bold')
        ax4a.legend(loc='best', fontsize=8)
        ax4a.grid(True, alpha=0.3, linestyle=':')
        
        # FSM Tilt Position
        ax4b.plot(t, np.array(self.log_data['fsm_tilt']) * 1e6, 
                  color=color_el, linewidth=1.5, label='fsm_tilt', alpha=0.9)
        ax4b.axhline(fsm_max, color='orange', linewidth=1.0, linestyle=':', 
                     alpha=0.6, label='FSM Limit')
        ax4b.axhline(-fsm_max, color='orange', linewidth=1.0, linestyle=':', alpha=0.6)
        ax4b.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
        ax4b.set_ylabel('Tilt Angle [µrad]', fontsize=10, fontweight='bold')
        ax4b.set_title('FSM Tilt Position', fontsize=11, fontweight='bold')
        ax4b.legend(loc='best', fontsize=8)
        ax4b.grid(True, alpha=0.3, linestyle=':')
        
        # FSM Tip Command
        ax4c.plot(t, np.array(self.log_data['fsm_cmd_tip']) * 1e6, 
                  color=color_cmd, linewidth=1.5, label='fsm_cmd_tip', alpha=0.9)
        ax4c.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
        ax4c.set_ylabel('Tip Command [µrad]', fontsize=10, fontweight='bold')
        ax4c.set_xlabel('Time [s]', fontsize=10, fontweight='bold')
        ax4c.set_title('FSM Tip Command', fontsize=11, fontweight='bold')
        ax4c.legend(loc='best', fontsize=8)
        ax4c.grid(True, alpha=0.3, linestyle=':')
        
        # FSM Tilt Command
        ax4d.plot(t, np.array(self.log_data['fsm_cmd_tilt']) * 1e6, 
                  color=color_cmd, linewidth=1.5, label='fsm_cmd_tilt', alpha=0.9)
        ax4d.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
        ax4d.set_ylabel('Tilt Command [µrad]', fontsize=10, fontweight='bold')
        ax4d.set_xlabel('Time [s]', fontsize=10, fontweight='bold')
        ax4d.set_title('FSM Tilt Command', fontsize=11, fontweight='bold')
        ax4d.legend(loc='best', fontsize=8)
        ax4d.grid(True, alpha=0.3, linestyle=':')
        
        fig4.suptitle('Fast Steering Mirror State and Commands', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # ===================================================================
        # FIGURE 5: Sensor Measurements - Encoders and Gyros
        # ===================================================================
        fig5, ((ax5a, ax5b), (ax5c, ax5d)) = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
        
        # Encoder Azimuth
        ax5a.plot(t, np.rad2deg(np.array(self.log_data['z_enc_az'])), 
                  color=color_az, linewidth=1.5, label='z_enc_az', alpha=0.9)
        ax5a.set_ylabel('Azimuth [deg]', fontsize=10, fontweight='bold')
        ax5a.set_title('Encoder Azimuth Measurement', fontsize=11, fontweight='bold')
        ax5a.legend(loc='best', fontsize=8)
        ax5a.grid(True, alpha=0.3, linestyle=':')
        
        # Encoder Elevation
        ax5b.plot(t, np.rad2deg(np.array(self.log_data['z_enc_el'])), 
                  color=color_el, linewidth=1.5, label='z_enc_el', alpha=0.9)
        ax5b.set_ylabel('Elevation [deg]', fontsize=10, fontweight='bold')
        ax5b.set_title('Encoder Elevation Measurement', fontsize=11, fontweight='bold')
        ax5b.legend(loc='best', fontsize=8)
        ax5b.grid(True, alpha=0.3, linestyle=':')
        
        # Gyro Azimuth
        ax5c.plot(t, np.rad2deg(np.array(self.log_data['z_gyro_az'])), 
                  color=color_az, linewidth=1.5, label='z_gyro_az', alpha=0.9)
        ax5c.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
        ax5c.set_ylabel('Azimuth Rate [deg/s]', fontsize=10, fontweight='bold')
        ax5c.set_xlabel('Time [s]', fontsize=10, fontweight='bold')
        ax5c.set_title('Gyro Azimuth Measurement', fontsize=11, fontweight='bold')
        ax5c.legend(loc='best', fontsize=8)
        ax5c.grid(True, alpha=0.3, linestyle=':')
        
        # Gyro Elevation
        ax5d.plot(t, np.rad2deg(np.array(self.log_data['z_gyro_el'])), 
                  color=color_el, linewidth=1.5, label='z_gyro_el', alpha=0.9)
        ax5d.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
        ax5d.set_ylabel('Elevation Rate [deg/s]', fontsize=10, fontweight='bold')
        ax5d.set_xlabel('Time [s]', fontsize=10, fontweight='bold')
        ax5d.set_title('Gyro Elevation Measurement', fontsize=11, fontweight='bold')
        ax5d.legend(loc='best', fontsize=8)
        ax5d.grid(True, alpha=0.3, linestyle=':')
        
        fig5.suptitle('Encoder and Gyro Sensor Measurements', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # ===================================================================
        # FIGURE 6: QPD Measurements (z_qpd_nes_x, z_qpd_nes_y)
        # ===================================================================
        fig6, (ax6a, ax6b) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # QPD X (NES)
        ax6a.plot(t, np.array(self.log_data['z_qpd_nes_x']), 
                  color=color_x, linewidth=1.5, label='z_qpd_nes_x', alpha=0.9)
        ax6a.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
        ax6a.set_ylabel('NES X', fontsize=11, fontweight='bold')
        ax6a.set_title('QPD X-Axis Measurement', fontsize=12, fontweight='bold')
        ax6a.legend(loc='best', fontsize=9)
        ax6a.grid(True, alpha=0.3, linestyle=':')
        
        # QPD Y (NES)
        ax6b.plot(t, np.array(self.log_data['z_qpd_nes_y']), 
                  color=color_y, linewidth=1.5, label='z_qpd_nes_y', alpha=0.9)
        ax6b.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
        ax6b.set_ylabel('NES Y', fontsize=11, fontweight='bold')
        ax6b.set_xlabel('Time [s]', fontsize=11, fontweight='bold')
        ax6b.set_title('QPD Y-Axis Measurement', fontsize=12, fontweight='bold')
        ax6b.legend(loc='best', fontsize=9)
        ax6b.grid(True, alpha=0.3, linestyle=':')
        
        fig6.suptitle('Quadrant Photo Detector (QPD) Measurements', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # ===================================================================
        # FIGURE 7: LOS Errors (los_error_x, los_error_y, total)
        # ===================================================================
        fig7, (ax7a, ax7b, ax7c) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        los_err_x = np.array(self.log_data['los_error_x'])
        los_err_y = np.array(self.log_data['los_error_y'])
        los_err_total = np.sqrt(los_err_x**2 + los_err_y**2)
        
        # LOS Error X
        ax7a.plot(t, los_err_x * 1e6, 
                  color=color_x, linewidth=1.5, label='los_error_x', alpha=0.9)
        ax7a.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
        ax7a.set_ylabel('LOS Error X [µrad]', fontsize=11, fontweight='bold')
        ax7a.set_title('Line-of-Sight Error X-Axis', fontsize=12, fontweight='bold')
        ax7a.legend(loc='best', fontsize=9)
        ax7a.grid(True, alpha=0.3, linestyle=':')
        
        # LOS Error Y
        ax7b.plot(t, los_err_y * 1e6, 
                  color=color_y, linewidth=1.5, label='los_error_y', alpha=0.9)
        ax7b.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
        ax7b.set_ylabel('LOS Error Y [µrad]', fontsize=11, fontweight='bold')
        ax7b.set_title('Line-of-Sight Error Y-Axis', fontsize=12, fontweight='bold')
        ax7b.legend(loc='best', fontsize=9)
        ax7b.grid(True, alpha=0.3, linestyle=':')
        
        # Total LOS Error
        ax7c.plot(t, los_err_total * 1e6, 
                  color='black', linewidth=2, label='Total LOS Error', alpha=0.9)
        ax7c.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
        ax7c.set_ylabel('Total LOS Error [µrad]', fontsize=11, fontweight='bold')
        ax7c.set_xlabel('Time [s]', fontsize=11, fontweight='bold')
        ax7c.set_title('Total Line-of-Sight Error Magnitude', fontsize=12, fontweight='bold')
        ax7c.legend(loc='best', fontsize=9)
        ax7c.grid(True, alpha=0.3, linestyle=':')
        
        # Add RMS metric to title
        rms_los = np.sqrt(np.mean(los_err_x**2 + los_err_y**2)) * 1e6
        fig7.suptitle(f'Line-of-Sight Pointing Errors (RMS: {rms_los:.2f} µrad)', 
                      fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # ===================================================================
        # FIGURE 8: Estimated State (est_az, est_el, est_az_dot, est_el_dot)
        # ===================================================================
        fig8, ((ax8a, ax8b), (ax8c, ax8d)) = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
        
        # Estimated Azimuth Position
        ax8a.plot(t, np.rad2deg(np.array(self.log_data['est_az'])), 
                  color=color_az, linewidth=1.5, label='est_az', alpha=0.9)
        ax8a.plot(t, np.rad2deg(np.array(self.log_data['q_az'])), 
                  color=color_az, linewidth=1.0, linestyle='--', label='q_az (true)', alpha=0.5)
        ax8a.set_ylabel('Azimuth [deg]', fontsize=10, fontweight='bold')
        ax8a.set_title('Estimated Azimuth Position', fontsize=11, fontweight='bold')
        ax8a.legend(loc='best', fontsize=8)
        ax8a.grid(True, alpha=0.3, linestyle=':')
        
        # Estimated Elevation Position
        ax8b.plot(t, np.rad2deg(np.array(self.log_data['est_el'])), 
                  color=color_el, linewidth=1.5, label='est_el', alpha=0.9)
        ax8b.plot(t, np.rad2deg(np.array(self.log_data['q_el'])), 
                  color=color_el, linewidth=1.0, linestyle='--', label='q_el (true)', alpha=0.5)
        ax8b.set_ylabel('Elevation [deg]', fontsize=10, fontweight='bold')
        ax8b.set_title('Estimated Elevation Position', fontsize=11, fontweight='bold')
        ax8b.legend(loc='best', fontsize=8)
        ax8b.grid(True, alpha=0.3, linestyle=':')
        
        # Estimated Azimuth Velocity
        ax8c.plot(t, np.rad2deg(np.array(self.log_data['est_az_dot'])), 
                  color=color_az, linewidth=1.5, label='est_az_dot', alpha=0.9)
        ax8c.plot(t, np.rad2deg(np.array(self.log_data['qd_az'])), 
                  color=color_az, linewidth=1.0, linestyle='--', label='qd_az (true)', alpha=0.5)
        ax8c.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
        ax8c.set_ylabel('Azimuth Rate [deg/s]', fontsize=10, fontweight='bold')
        ax8c.set_xlabel('Time [s]', fontsize=10, fontweight='bold')
        ax8c.set_title('Estimated Azimuth Velocity', fontsize=11, fontweight='bold')
        ax8c.legend(loc='best', fontsize=8)
        ax8c.grid(True, alpha=0.3, linestyle=':')
        
        # Estimated Elevation Velocity
        ax8d.plot(t, np.rad2deg(np.array(self.log_data['est_el_dot'])), 
                  color=color_el, linewidth=1.5, label='est_el_dot', alpha=0.9)
        ax8d.plot(t, np.rad2deg(np.array(self.log_data['qd_el'])), 
                  color=color_el, linewidth=1.0, linestyle='--', label='qd_el (true)', alpha=0.5)
        ax8d.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
        ax8d.set_ylabel('Elevation Rate [deg/s]', fontsize=10, fontweight='bold')
        ax8d.set_xlabel('Time [s]', fontsize=10, fontweight='bold')
        ax8d.set_title('Estimated Elevation Velocity', fontsize=11, fontweight='bold')
        ax8d.legend(loc='best', fontsize=8)
        ax8d.grid(True, alpha=0.3, linestyle=':')
        
        fig8.suptitle('EKF State Estimates vs. True State', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Show all figures
        plt.show()
        
        print("=" * 70)
        print("PLOTS GENERATED SUCCESSFULLY")
        print("=" * 70)
        print(f"Total Figures Created: 8")
        print(f"  - Figure 1: Gimbal Position (q_az, q_el)")
        print(f"  - Figure 2: Gimbal Velocity (qd_az, qd_el)")
        print(f"  - Figure 3: Control Torques (torque_az, torque_el)")
        print(f"  - Figure 4: FSM State (tip, tilt, commands)")
        print(f"  - Figure 5: Sensor Measurements (encoders, gyros)")
        print(f"  - Figure 6: QPD Measurements (NES X, Y)")
        print(f"  - Figure 7: LOS Errors (X, Y, Total)")
        print(f"  - Figure 8: EKF State Estimates")
        print(f"\nTotal Simulation Time: {t[-1]:.3f} s")
        print(f"Number of Samples: {len(t)}")
        print(f"RMS LOS Error: {rms_los:.2f} µrad")
        print("=" * 70)
    
    def reset(self) -> None:
        """Reset simulation to initial conditions."""
        self.time = 0.0
        self.iteration = 0
        self.last_coarse_update = 0.0
        self.last_fine_update = 0.0
        self.last_encoder_update = 0.0
        self.last_gyro_update = 0.0
        self.last_qpd_update = 0.0
        self.last_log_time = 0.0
        
        # Reset dynamics
        self.q_az = 0.0
        self.q_el = 0.0
        self.qd_az = 0.0
        self.qd_el = 0.0
        
        # Reset all components
        self.motor_az.reset()
        self.motor_el.reset()
        self.fsm.reset()
        self.encoder_az.reset()
        self.encoder_el.reset()
        self.gyro_az.reset()
        self.gyro_el.reset()
        self.qpd.reset()
        self.estimator.reset()
        self.coarse_controller.reset()
        self.fsm_controller.reset()
        
        # Reset state
        self.state = SimulationState()
        
        # Clear logs
        self.log_data.clear()


def main():
    """
    Demonstration of digital twin simulation.
    
    Runs a 5-second closed-loop simulation with target tracking.
    """
    print("=" * 70)
    print("Laser Communication Terminal Digital Twin")
    print("Integrated Closed-Loop Simulation")
    print("=" * 70)
    print()
    
    # Configure simulation
    config = SimulationConfig(
        dt_sim=0.001,        # 1 ms physics timestep
        dt_coarse=0.010,     # 10 ms coarse control
        dt_fine=0.001,       # 1 ms fine control
        log_period=0.001,    # 1 ms logging
        seed=42,
        target_az=np.deg2rad(5.0),   # 5° azimuth target
        target_el=np.deg2rad(30.0),  # 30° elevation target
        target_enabled=True,
        enable_visualization=False,  # Set to True to enable MuJoCo viewer
        real_time_factor=0.0,        # 0.0 = fast-as-possible
        viewer_fps=30.0              # 30 FPS when visualization enabled
    )
    
    # Create runner
    runner = DigitalTwinRunner(config)
    
    # Run simulation
    results = runner.run_simulation(duration=5.0)
    
    # Print summary
    print()
    print("=" * 70)
    print("SIMULATION RESULTS")
    print("=" * 70)
    print(f"Duration:          {results['duration']:.3f} s")
    print(f"Samples logged:    {results['n_samples']}")
    print()
    print("POINTING PERFORMANCE:")
    print(f"  LOS Error RMS:   {results['los_error_rms']*1e6:.2f} µrad")
    print(f"  LOS Error Final: {results['los_error_final']*1e6:.2f} µrad")
    print(f"  LOS Error X RMS: {results['los_error_rms_x']*1e6:.2f} µrad")
    print(f"  LOS Error Y RMS: {results['los_error_rms_y']*1e6:.2f} µrad")
    print()
    print("ESTIMATION PERFORMANCE:")
    print(f"  State Est. RMS:  {results['est_error_rms']*1e6:.2f} µrad")
    print()
    print("CONTROL EFFORT:")
    print(f"  Torque Az RMS:   {results['torque_rms_az']:.3f} N·m")
    print(f"  Torque El RMS:   {results['torque_rms_el']:.3f} N·m")
    print(f"  FSM Saturation:  {results['fsm_saturation_percent']:.1f}%")
    print()
    print("FINAL STATE:")
    print(f"  Azimuth:         {np.rad2deg(results['final_az']):.3f}°")
    print(f"  Elevation:       {np.rad2deg(results['final_el']):.3f}°")
    print("=" * 70)
    
    return results


def main_feedback_linearization():
    """
    Demonstration of Feedback Linearization Controller.
    
    This demo showcases the complete signal flow architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    SENSOR LAYER                              │
    │  (sensors/encoder.py, sensors/gyro.py, sensors/qpd.py)     │
    └────────────────┬────────────────────────────────────────────┘
                     │ Raw Measurements:
                     │ - Encoder: θ_az, θ_el (noisy position)
                     │ - Gyro: ω_az, ω_el (noisy angular velocity)
                     │ - QPD: Δθ_fine (optical pointing error)
                     ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                   ESTIMATOR LAYER                            │
    │           (estimators/state_estimator.py)                   │
    │  Extended Kalman Filter fuses all sensor data               │
    └────────────────┬────────────────────────────────────────────┘
                     │ Filtered State Dictionary
                     ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                  CONTROLLER LAYER                            │
    │  FeedbackLinearizationController receives state_estimate    │
    │  Cancels M(q), C(q,dq), G(q), compensates disturbances     │
    └────────────────┬────────────────────────────────────────────┘
                     │ Torque Commands [N·m]
                     ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                   ACTUATOR LAYER                             │
    │              (actuators/motor_model.py)                      │
    └─────────────────────────────────────────────────────────────┘
    
    Runs a 10-second closed-loop simulation with aggressive target tracking.
    """
    print("=" * 70)
    print("Laser Communication Terminal Digital Twin")
    print("FEEDBACK LINEARIZATION CONTROLLER DEMONSTRATION")
    print("=" * 70)
    print()
    print("Signal Flow Architecture:")
    print("  Sensors → Estimator (EKF) → FL Controller → Actuators")
    print()
    
    # Configure simulation with Feedback Linearization
    config = SimulationConfig(
        dt_sim=0.001,        # 1 ms physics timestep
        dt_coarse=0.010,     # 10 ms coarse control
        dt_fine=0.001,       # 1 ms fine control
        log_period=0.001,    # 1 ms logging
        seed=42,
        target_az=np.deg2rad(10.0),   # 10° azimuth target (aggressive)
        target_el=np.deg2rad(45.0),   # 45° elevation target (aggressive)
        target_enabled=True,
        use_feedback_linearization=True,  # Enable FL controller
        enable_visualization=False,
        real_time_factor=0.0,
        viewer_fps=30.0,
        
        # Feedback Linearization gains
        feedback_linearization_config={
            'kp': [150.0, 150.0],      # Higher gains possible due to linearization
            'kd': [30.0, 30.0],
            'tau_max': [10.0, 10.0],
            'tau_min': [-10.0, -10.0]
        },
        
        # Gimbal dynamics parameters
        dynamics_config={
            'pan_mass': 0.5,
            'tilt_mass': 0.25,
            'cm_r': 0.02,
            'cm_h': 0.005,
            'gravity': 9.81
        }
    )
    
    # Create runner
    print("Initializing simulation...")
    runner = DigitalTwinRunner(config)
    print("  ✓ Dynamics model initialized")
    print("  ✓ Sensors initialized (encoders, gyros, QPD)")
    print("  ✓ EKF state estimator initialized")
    print("  ✓ Feedback Linearization controller initialized")
    print("  ✓ Motor and FSM actuators initialized")
    print()
    
    # Run simulation
    print("Running closed-loop simulation (10 seconds)...")
    results = runner.run_simulation(duration=10.0)
    
    # Print detailed summary
    print()
    print("=" * 70)
    print("FEEDBACK LINEARIZATION CONTROLLER RESULTS")
    print("=" * 70)
    print(f"Duration:          {results['duration']:.3f} s")
    print(f"Samples logged:    {results['n_samples']}")
    print(f"Controller:        Feedback Linearization")
    print()
    print("POINTING PERFORMANCE:")
    print(f"  LOS Error RMS:   {results['los_error_rms']*1e6:.2f} µrad")
    print(f"  LOS Error Final: {results['los_error_final']*1e6:.2f} µrad")
    print(f"  LOS Error X RMS: {results['los_error_rms_x']*1e6:.2f} µrad")
    print(f"  LOS Error Y RMS: {results['los_error_rms_y']*1e6:.2f} µrad")
    print()
    print("ESTIMATION PERFORMANCE:")
    print(f"  State Est. RMS:  {results['est_error_rms']*1e6:.2f} µrad")
    print()
    print("CONTROL EFFORT:")
    print(f"  Torque Az RMS:   {results['torque_rms_az']:.3f} N·m")
    print(f"  Torque El RMS:   {results['torque_rms_el']:.3f} N·m")
    print(f"  FSM Saturation:  {results['fsm_saturation_percent']:.1f}%")
    print()
    print("FINAL STATE:")
    print(f"  Target Az:       {np.rad2deg(config.target_az):.3f}°")
    print(f"  Target El:       {np.rad2deg(config.target_el):.3f}°")
    print(f"  Actual Az:       {np.rad2deg(results['final_az']):.3f}°")
    print(f"  Actual El:       {np.rad2deg(results['final_el']):.3f}°")
    print(f"  Az Error:        {np.rad2deg(config.target_az - results['final_az'])*3600:.2f} arcsec")
    print(f"  El Error:        {np.rad2deg(config.target_el - results['final_el'])*3600:.2f} arcsec")
    print()
    print("NOTES:")
    print("  - Feedback Linearization cancels nonlinear dynamics (M, C, G)")
    print("  - EKF estimates and compensates disturbances")
    print("  - Higher control gains are stable due to linearization")
    print("  - See log_data in results for time-series plots")
    print("=" * 70)
    
    return results


if __name__ == '__main__':
    # Run standard PID demonstration
    print("\n### Running Standard PID Controller Demo ###\n")
    results_pid = main()
    
    print("\n\n")
    
    # Run Feedback Linearization demonstration
    print("### Running Feedback Linearization Controller Demo ###\n")
    results_fl = main_feedback_linearization()
    
    # Comparison
    print("\n\n")
    print("=" * 70)
    print("CONTROLLER COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<30} {'PID':<15} {'FL':<15}")
    print("-" * 70)
    print(f"{'LOS Error RMS (µrad)':<30} {results_pid['los_error_rms']*1e6:<15.2f} {results_fl['los_error_rms']*1e6:<15.2f}")
    print(f"{'Torque Az RMS (N·m)':<30} {results_pid['torque_rms_az']:<15.3f} {results_fl['torque_rms_az']:<15.3f}")
    print(f"{'Torque El RMS (N·m)':<30} {results_pid['torque_rms_el']:<15.3f} {results_fl['torque_rms_el']:<15.3f}")
    print(f"{'FSM Saturation (%)':<30} {results_pid['fsm_saturation_percent']:<15.1f} {results_fl['fsm_saturation_percent']:<15.1f}")
    print("=" * 70)
