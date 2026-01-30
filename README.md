<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/MuJoCo-3.2.0+-green.svg" alt="MuJoCo 3.2+">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
  <img src="https://img.shields.io/badge/DO--178C-Level%20B-red.svg" alt="DO-178C Level B">
  <img src="https://img.shields.io/badge/Accuracy-<2%20μrad%20RMS-brightgreen.svg" alt="Sub-2 μrad RMS">
</p>

# MicroPrecisionGimbal Digital Twin

**Aerospace-Grade Simulation Framework for Satellite Laser Communication Pointing Systems**

A high-fidelity digital twin for precision pointing systems achieving **sub-microradian accuracy** for optical inter-satellite links (OISL) and ground-to-space laser communication terminals. Features a two-stage hierarchical control architecture with Coarse Pointing Assembly (CPA) and Fine Steering Mirror (FSM).

---

## 🎯 Key Features

| Feature | Description |
|---------|-------------|
| **Sub-2 µrad RMS Pointing** | Production-validated pointing accuracy for laser communication |
| **Hierarchical Control** | Two-stage CPA + FSM architecture with seamless handover |
| **Multi-Fidelity Simulation** | Four fidelity levels (L1-L4) for rapid iteration to production validation |
| **Advanced Controllers** | PID, Feedback Linearization (FBL), FBL+NDOB implementations |
| **Extended Kalman Filter** | Sensor fusion for optimal state estimation |
| **Frequency Response Analysis** | Empirical SIDF-based Bode plot generation |
| **DO-178C Compliant** | Level B aerospace software development standards |
| **Publication-Quality Outputs** | 300 DPI figures with LaTeX typography for IEEE/AIAA papers |

---

## 📐 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MicroPrecisionGimbal Digital Twin                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Target      ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐     │
│  Command ───►│ Coarse   │──►│   EKF    │──►│  Coarse  │──►│  Gimbal  │───┐ │
│              │ Pointing │   │ Estimator│   │Controller│   │ Dynamics │   │ │
│              │ Assembly │   └──────────┘   └──────────┘   └──────────┘   │ │
│              └──────────┘                                                │ │
│                   │                                                      │ │
│                   │ Handover (<0.8°)                                     │ │
│                   ▼                                                      │ │
│              ┌──────────┐   ┌──────────┐   ┌──────────┐                  │ │
│              │   FSM    │──►│   FSM    │──►│   FSM    │                  │ │
│              │Controller│   │ Dynamics │   │ Actuator │                  │ │
│              └──────────┘   └──────────┘   └──────────┘                  │ │
│                   │                                                      │ │
│                   └──────────────────────────────────────────────────────┘ │
│                                      │                                     │
│                                      ▼                                     │
│                              Line-of-Sight Output                          │
│                              (< 2 µrad RMS)                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Control Hierarchy

| Stage | Component | Bandwidth | Range | Accuracy |
|-------|-----------|-----------|-------|----------|
| **Coarse** | 2-Axis Gimbal (CPA) | 10 Hz | ±90° | ±0.8° |
| **Fine** | Fast Steering Mirror (FSM) | 1 kHz | ±400 µrad | <2 µrad RMS |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- 8 GB RAM minimum (16 GB recommended for L4 fidelity)
- Windows 10/11, Linux, or macOS

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/MicroPrecisionGimbal.git
cd MicroPrecisionGimbal

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run Your First Simulation

```bash
# Three-way controller comparison (PID vs FBL vs FBL+NDOB)
python demo_feedback_linearization.py

# Frequency response analysis
python demo_frequency_response.py
```

---

## 📊 Simulation Fidelity Levels

The simulation supports four fidelity levels for different use cases:

| Level | Name | Duration | Timestep | Use Case |
|-------|------|----------|----------|----------|
| **L1** | Quick Test | 10s | 10 ms | Unit testing, rapid iteration |
| **L2** | Integration | 20s | 5 ms | Controller tuning, integration tests |
| **L3** | System Test | 30s | 2 ms | Performance validation, disturbance analysis |
| **L4** | Production | 60s | 1 ms | Acceptance testing, flight readiness |

### Selecting Fidelity Level

```python
from lasercom_digital_twin.core.simulation.simulation_runner import SimulationConfig

# L1: Rapid development testing
config_l1 = SimulationConfig(
    duration=10.0,
    dt=0.01,
    fidelity_level='L1'
)

# L4: Production validation (CI/CD regression tests)
config_l4 = SimulationConfig(
    duration=60.0,
    dt=0.001,
    fidelity_level='L4'
)
```

### Performance Thresholds by Fidelity

| Metric | L1 | L2 | L3 | L4 (Production) |
|--------|----|----|----|----|
| RMS Pointing Error | <50 µrad | <20 µrad | <10 µrad | **<2 µrad** |
| Peak Pointing Error | <200 µrad | <100 µrad | <50 µrad | **<30 µrad** |
| FSM Saturation | <10% | <5% | <3% | **<1%** |

---

## 🎮 Controller Selection

### Available Controllers

#### 1. PID Controller (Baseline)
```python
config = SimulationConfig(
    controller_type='pid',
    controller_gains={
        'kp': [3.514, 1.320],    # Azimuth, Elevation
        'ki': [15.464, 4.148],
        'kd': [0.293, 0.059]
    }
)
```

#### 2. Feedback Linearization (FBL)
Model-based nonlinear control with gravity compensation and inertia decoupling.

```python
config = SimulationConfig(
    controller_type='fbl',
    controller_gains={
        'kp': [400.0, 400.0],
        'kd': [40.0, 40.0]
    }
)
```

#### 3. FBL + Nonlinear Disturbance Observer (NDOB)
Advanced control with real-time disturbance estimation and rejection.

```python
from lasercom_digital_twin.core.n_dist_observer import NDOBConfig

ndob_config = NDOBConfig(
    lambda_az=50.0,    # Observer bandwidth (rad/s)
    lambda_el=50.0,
    d_max=0.5,         # Maximum disturbance estimate (N·m)
    enable=True
)

config = SimulationConfig(
    controller_type='fbl_ndob',
    ndob_config=ndob_config
)
```

---

## ⚙️ Configuration Parameters

### Simulation Configuration

```python
config = SimulationConfig(
    # Time parameters
    duration=30.0,              # Simulation duration (seconds)
    dt=0.001,                   # Timestep (seconds)
    
    # Initial conditions
    theta_init=[0.0, 0.0],      # Initial position [az, el] (rad)
    theta_dot_init=[0.0, 0.0],  # Initial velocity [az, el] (rad/s)
    
    # Target trajectory
    target_type='step',         # 'step', 'sine', 'square', 'ramp'
    target_amplitude=[0.1, 0.05], # Target amplitude [az, el] (rad)
    target_frequency=0.5,       # For periodic targets (Hz)
    
    # Sensor configuration
    sensor_noise_enabled=True,
    qpd_noise_std=1.0,          # QPD noise (µrad)
    imu_gyro_noise_std=0.01,    # IMU gyro noise (rad/s)
    
    # Disturbances
    disturbances_enabled=True,
    base_motion_amplitude=0.01, # Platform vibration (rad)
    pointing_disturbance_rms=15.0  # Disturbance torque (mN·m)
)
```

### Gimbal Dynamics Parameters

```python
from lasercom_digital_twin.core.dynamics.gimbal_dynamics import GimbalDynamics

dynamics = GimbalDynamics(
    pan_mass=1.0,       # Azimuth axis mass (kg)
    tilt_mass=0.5,      # Elevation axis mass (kg)
    cm_r=0.0,           # Center of mass radial offset (m)
    cm_h=0.0,           # Center of mass height offset (m)
    gravity=9.81,       # Gravitational acceleration (m/s²)
    friction_az=0.1,    # Azimuth friction coefficient
    friction_el=0.1     # Elevation friction coefficient
)
```

### FSM Configuration

```python
fsm_config = {
    'bandwidth': 500.0,         # Control bandwidth (Hz)
    'damping': 0.707,           # Damping ratio
    'angle_limit_urad': 400.0,  # Maximum deflection (µrad)
    'nonlinearity': True,       # Enable nonlinear effects
    'hysteresis_enabled': True  # Enable hysteresis model
}
```

---

## 📈 Output & Visualization

### Generated Figures

Running `demo_feedback_linearization.py` generates 13 publication-quality figures:

| Figure | Description |
|--------|-------------|
| `position_tracking_az/el` | Position response comparison |
| `tracking_error_az/el` | Error time history with RMS annotations |
| `control_torque_az/el` | Control effort comparison |
| `phase_portrait_az/el` | State-space trajectory |
| `los_error` | Combined line-of-sight error |
| `fsm_utilization` | Fine steering mirror usage |
| `metrics_summary` | Bar chart of performance metrics |
| `handover_region` | CPA→FSM handover analysis |

### Output Directories

```
figures_comparative/    # Controller comparison plots
figures_bode/          # Frequency response plots  
figures_diagrams/      # Block diagrams
frequency_response_data/ # Raw frequency sweep data
```

---

## 🧪 Testing

### Run All Tests

```bash
# Unit tests
pytest lasercom_digital_twin/tests/ -v

# CI/CD regression tests (L4 fidelity)
pytest lasercom_digital_twin/ci_tests/ -v

# With coverage
pytest --cov=lasercom_digital_twin --cov-report=html
```

### Performance Regression Tests

The CI/CD pipeline enforces strict performance gates:

```bash
pytest lasercom_digital_twin/ci_tests/test_regression.py -v
```

**Mandatory Pass Criteria:**
- ✅ RMS Pointing Error < 2.0 µrad
- ✅ Peak Error < 30.0 µrad
- ✅ FSM Saturation < 1.0%
- ✅ No NaN/Inf in telemetry

---

## 🐳 Docker Deployment

### Build Container

```bash
docker build -t lasercom-digital-twin:latest .
```

### Run Headless Regression

```bash
docker run --rm lasercom-digital-twin:latest
```

### Interactive Development

```bash
docker run --rm -it lasercom-digital-twin:latest bash
```

---

## 📁 Project Structure

```
MicroPrecisionGimbal/
├── demo_feedback_linearization.py  # Main controller comparison demo
├── demo_frequency_response.py      # Frequency response analysis
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Container deployment
├── pytest.ini                      # Test configuration
│
├── config/
│   └── fidelity_levels.json        # L1-L4 fidelity configurations
│
├── lasercom_digital_twin/          # Core simulation package
│   ├── core/
│   │   ├── controllers/            # PID, FBL, FSM controllers
│   │   ├── dynamics/               # Gimbal & FSM dynamics
│   │   ├── estimators/             # EKF state estimation
│   │   ├── sensors/                # QPD, IMU models
│   │   ├── actuators/              # Motor & FSM actuators
│   │   ├── frequency_response/     # Bode plot analysis
│   │   ├── plots/                  # Publication-quality plotting
│   │   └── simulation/             # Simulation orchestrator
│   ├── control_design/             # Controller synthesis tools
│   ├── ci_tests/                   # Regression test suite
│   └── tests/                      # Unit tests
│
├── demos/                          # Development & debug scripts
├── docs/                           # Technical documentation
├── scripts/                        # Utility scripts
└── figures_*/                      # Generated output (gitignored)
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [FEEDBACK_LINEARIZATION_GUIDE.md](docs/FEEDBACK_LINEARIZATION_GUIDE.md) | FBL controller theory & implementation |
| [IMPLEMENTATION_SUMMARY.md](docs/IMPLEMENTATION_SUMMARY.md) | Signal flow & architecture details |
| [CI_CD_Pipeline.md](docs/CI_CD_Pipeline.md) | Deployment & testing workflow |
| [EKF_ADAPTIVE_TUNING_SUMMARY.md](docs/EKF_ADAPTIVE_TUNING_SUMMARY.md) | Extended Kalman Filter tuning |
| [NDOB_FIX_SUMMARY.md](docs/NDOB_FIX_SUMMARY.md) | Disturbance observer details |

---

## 🔬 Control Design Workflow

For designing new controllers:

```python
from lasercom_digital_twin.control_design import (
    ControllerDesigner,
    SystemModeler,
    ControlAnalyzer
)

# 1. Create linearized plant model
modeler = SystemModeler()
plant = modeler.create_gimbal_plant_model(inertia_az=2.0, inertia_el=1.0)

# 2. Design controller
designer = ControllerDesigner()
controller = designer.design_pid_controller(
    plant,
    bandwidth=10.0,      # Closed-loop bandwidth (Hz)
    phase_margin=60.0    # Phase margin (degrees)
)

# 3. Analyze stability
analyzer = ControlAnalyzer()
results = analyzer.analyze_system(plant, controller)
print(f"Gain Margin: {results.gain_margin_db:.1f} dB")
print(f"Phase Margin: {results.phase_margin_deg:.1f}°")
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`pytest -v`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

### Code Standards

- Follow PEP 8 style guidelines
- Add docstrings to all public functions
- Include unit tests for new features
- Ensure L4 regression tests pass

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📖 Citation

If you use this software in your research, please cite:

```bibtex
@software{microprecisiongimbal2026,
  author = {Mustafa, S. Shahid},
  title = {MicroPrecisionGimbal: Digital Twin for Satellite Laser Communication Pointing Systems},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/yourusername/MicroPrecisionGimbal}
}
```

---

## 📧 Contact

**Dr. S. Shahid Mustafa**  
- GitHub: [@yourusername](https://github.com/yourusername)

---

<p align="center">
  <b>Built for precision. Designed for space.</b>
</p>
