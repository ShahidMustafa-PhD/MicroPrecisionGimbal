# AI Coding Agent Instructions - MicroPrecisionGimbal Digital Twin

## Project Overview
Aerospace-grade simulation framework for satellite laser communication (lasercom) pointing systems with **sub-microradian accuracy** requirements. Two-stage hierarchical control: Coarse Pointing Assembly (CPA) gimbal + Fine Steering Mirror (FSM). MuJoCo physics engine with Extended Kalman Filter state estimation. DO-178C Level B compliance.

## Architecture: Signal Flow & Multi-Rate Execution

**Critical Concept:** This is a multi-rate, fixed-step deterministic simulation. Each subsystem operates at different rates:
```
MuJoCo Physics → Sensors → EKF → Controllers → Actuators → Physics (1ms loop)
                            ↓
                    Coarse Gimbal: 10ms rate (10 Hz)
                    FSM Controller: 1ms rate (1 kHz)
```

**Data Flow Pattern:**
1. **Sensors** (`core/sensors/`) measure noisy gimbal state → produce individual measurements
2. **Estimator** (`core/estimators/state_estimator.py`) fuses sensors via EKF → returns `state_estimate` dict
3. **Controllers** (`core/controllers/`) compute torque commands from state estimates
4. **Actuators** (`core/actuators/`) apply motor dynamics (friction, backlash)
5. **Dynamics** (`core/dynamics/`) or MuJoCo integrates motion equations

**Key Files:**
- [lasercom_digital_twin/core/simulation/simulation_runner.py](lasercom_digital_twin/core/simulation/simulation_runner.py) - Master orchestrator, read this FIRST
- [lasercom_digital_twin/core/controllers/control_laws.py](lasercom_digital_twin/core/controllers/control_laws.py) - Coarse controllers (PID, Feedback Linearization)
- [lasercom_digital_twin/core/controllers/fsm_pid_control.py](lasercom_digital_twin/core/controllers/fsm_pid_control.py) - FSM 4th-order state-space controller
- [lasercom_digital_twin/core/dynamics/fsm_dynamics.py](lasercom_digital_twin/core/dynamics/fsm_dynamics.py) - FSM plant model (RK4 integration)

## Configuration-Driven Design

**Never hardcode parameters.** All configs use `@dataclass` pattern and JSON files.

### Fidelity Levels (L1-L4)
Located in [config/fidelity_levels.json](config/fidelity_levels.json):
- **L1**: Quick test (10s, linear models, no noise) - unit testing
- **L2**: Integration (20s, moderate fidelity) 
- **L3**: System validation (30s, high fidelity)
- **L4**: Production (60s, max fidelity, < 2 µrad RMS requirement)

**Usage:**
```bash
# Run with specific fidelity
python -m lasercom_digital_twin.runner --fidelity L4 --duration 60
```

### Config Structure Pattern
```python
@dataclass
class ComponentConfig:
    """All configs follow this pattern."""
    param1: float = default_value
    param2: bool = False
```

See [lasercom_digital_twin/core/simulation/simulation_runner.py#L63](lasercom_digital_twin/core/simulation/simulation_runner.py#L63) for `SimulationConfig` example.

## Testing Strategy

### Regression Tests Are NOT Unit Tests
[lasercom_digital_twin/core/ci_tests/test_regression.py](lasercom_digital_twin/core/ci_tests/test_regression.py) contains **system-level end-to-end validation**. Tests **MUST FAIL** if performance degrades.

**Mandatory Performance Gates:**
- RMS Pointing Error < 2.0 µrad
- Peak Error < 30.0 µrad  
- FSM Saturation < 1.0%
- No NaN/Inf in telemetry

**Running Tests:**
```bash
# All regression tests with L4 fidelity
pytest lasercom_digital_twin/core/ci_tests/test_regression.py -v

# Unit tests for specific components
pytest lasercom_digital_twin/tests/test_controllers.py -v

# With markers
pytest -m "not slow"  # Skip long-duration tests
```

Config: [pytest.ini](pytest.ini#L9) - testpaths point to `tests/` and `core/ci_tests/`

### Docker CI/CD Workflow
```bash
# Build (multi-stage optimized)
docker build -t lasercom-digital-twin:latest .

# Run headless regression
docker run --rm lasercom-digital-twin:latest

# Interactive debugging
docker run --rm -it lasercom-digital-twin:latest bash
```

See [docs/CI_CD_Pipeline.md](docs/CI_CD_Pipeline.md) for complete deployment guide.

## Control Design Module

Located in [lasercom_digital_twin/control_design/](lasercom_digital_twin/control_design/) - **separate from simulation execution** to ensure reproducible controller synthesis.

**Key Tools:**
- `ControllerDesigner` - Synthesis algorithms (PID, LQG, H-infinity)
- `ControlAnalyzer` - Stability margins, frequency/time domain analysis
- `SystemModeler` - Plant linearization and model reduction
- `DesignRequirements` - Performance specs validation

**Example Pattern:**
```python
from lasercom_digital_twin.control_design import ControllerDesigner, SystemModeler

# 1. Create plant model
modeler = SystemModeler()
plant = modeler.create_gimbal_plant_model(inertia_az=2.0, ...)

# 2. Design controller
designer = ControllerDesigner()
controller = designer.design_pid_controller(plant, specs)

# 3. Validate against requirements
analyzer = ControlAnalyzer()
results = analyzer.analyze_system(plant, controller)
```

Controllers are exported to JSON (e.g., [fsm_controller_gains.json](fsm_controller_gains.json)) then loaded by simulation runner.

## Critical Conventions

### Coordinate Frames
- **B-frame**: Body (gimbal mechanical axes)
- **O-frame**: Optical (detector aligned, field rotation compensated)
- **N-frame**: Navigation (ECI or target reference)

FSM operates in O-frame. Transformations in [lasercom_digital_twin/core/coordinate_frames/transformations.py](lasercom_digital_twin/core/coordinate_frames/transformations.py).

### State Estimation Pattern
EKF returns a dictionary, not an array:
```python
state_estimate = estimator.get_fused_state()
# {'theta_az': float, 'theta_el': float, 'theta_dot_az': float, ...}
```

**Never access raw sensor values directly in controllers.** Always use EKF fused state.

### Controller Interface
All controllers implement:
```python
def compute_control(self, state: Dict, target: Dict, dt: float) -> np.ndarray:
    """Returns torque/angle command."""
```

Coarse controllers output torque [N·m], FSM controllers output angle [rad].

## GimbalDynamics API (Common Error Source)

When implementing feedback linearization or model-based control:
```python
# CORRECT API (as of latest refactor)
M = dynamics.get_mass_matrix()        # NOT compute_inertia_matrix()
C = dynamics.get_coriolis_matrix()    # NOT compute_coriolis_matrix()  
G = dynamics.get_gravity_vector()     # NOT compute_gravity_vector()
```

See [FEEDBACK_LINEARIZATION_GUIDE.md](FEEDBACK_LINEARIZATION_GUIDE.md) for complete example.

## Running Demonstrations

```bash
# Feedback linearization comparison
python demo_feedback_linearization.py

# Monte Carlo analysis
python lasercom_digital_twin/examples/demo_monte_carlo_analysis.py

# Disturbances and faults
python lasercom_digital_twin/examples/demo_disturbances_faults.py
```

## Python Environment Setup

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux

# Install dependencies
pip install -r requirements.txt

# Run simulation
python -m lasercom_digital_twin.runner --fidelity L3 --visualize
```

**Key Dependencies:** numpy, scipy, mujoco>=3.2.0, control, pytest

## Common Pitfalls

1. **Indentation Errors**: Controllers must be module-level classes, not nested inside other classes
2. **Mutable Defaults**: Use `Optional[np.ndarray] = None` not `np.ndarray = np.zeros(2)` in function signatures
3. **Hardcoded Parameters**: Always load from config files, never embed magic numbers
4. **Unit Confusion**: Angles are **radians** internally, degrees only in user-facing outputs
5. **Integration Method**: FSM uses RK4 for stiff dynamics, gimbal uses forward Euler (configurable)
6. **PID Design for Gimbal**: **CRITICAL** - Gimbal is Type-2 plant (double integrator), NOT Type-0
   - Linearization returns position output but dynamics are acceleration-based
   - Correct gains: `Kp = M*ωc²`, `Kd = 2*ζ*sqrt(M*Kp)` for bandwidth ωc
   - Always add gravity feedforward: `tau = tau_pid + G(q)`
   - Derivative term must use `error_derivative = ref_velocity - meas_velocity`

## Documentation Quick Reference

- [README.md](README.md) - System architecture, compliance standards
- [DEPLOYMENT_SUMMARY.md](DEPLOYMENT_SUMMARY.md) - CI/CD deliverables, Docker usage
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Signal flow, recent refactors
- [docs/CI_CD_Pipeline.md](docs/CI_CD_Pipeline.md) - Complete deployment guide
- [lasercom_digital_twin/control_design/README.md](lasercom_digital_twin/control_design/README.md) - Controller synthesis workflow

## When Making Changes

1. **Read [lasercom_digital_twin/README.md](lasercom_digital_twin/README.md) first** - design principles (deterministic execution, parameter-driven config, swappable fidelity)
2. **Update fidelity configs** if adding new parameters
3. **Add regression tests** for any performance-critical features
4. **Maintain DO-178C traceability** - reference requirements in docstrings
5. **Test with L4 fidelity** before committing - must pass all thresholds
