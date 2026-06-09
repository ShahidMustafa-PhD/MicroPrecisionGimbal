<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/MuJoCo-3.2.0+-green.svg" alt="MuJoCo 3.2+">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
  <img src="https://img.shields.io/badge/DO--178C-Level%20B-red.svg" alt="DO-178C Level B">
  <img src="https://img.shields.io/badge/Pointing%20Accuracy-<2%20µrad%20RMS-brightgreen.svg" alt="Sub-2 µrad RMS">
  <img src="https://img.shields.io/badge/Target-IEEE%2FAIAA-blueviolet.svg" alt="IEEE/AIAA">
</p>

<h1 align="center">MicroPrecisionGimbal Digital Twin</h1>

<p align="center">
  <b>Aerospace-Grade Simulation Framework for Satellite Laser Communication Pointing Systems</b><br>
  PID &nbsp;·&nbsp; Feedback Linearization (FBL) &nbsp;·&nbsp; FBL + Nonlinear Disturbance Observer (NDOB)
</p>

---

## Simulation Results

> Full figure archive → **[📊 View Complete Results Gallery](docs/RESULTS.md)**

<table>
  <tr>
    <td width="50%">
      <img src="docs/figures/fig1_position_tracking.png" alt="Position Tracking Comparison" width="100%">
      <p align="center"><b>Fig. 1</b> — Position tracking: PID vs. FBL vs. FBL+NDOB.<br>FBL+NDOB achieves the fastest convergence with minimum overshoot on both axes.</p>
    </td>
    <td width="50%">
      <img src="docs/figures/fig2_tracking_error_handover.png" alt="Tracking Error and CPA-FSM Handover" width="100%">
      <p align="center"><b>Fig. 2</b> — Tracking error with CPA→FSM handover boundary.<br>FBL+NDOB: <b>1.34 µrad RMS</b> steady-state vs. 8.7 µrad (PID baseline).</p>
    </td>
  </tr>
</table>

<p align="center">
  <a href="docs/RESULTS.md"><b>→ See all 32 simulation figures including friction analysis, FSM performance, EKF tuning, and benchmark tables</b></a>
</p>

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Sub-2 µrad RMS Pointing** | Production-validated pointing accuracy for laser communication |
| **Hierarchical Control** | Two-stage CPA + FSM architecture with seamless handover |
| **Multi-Fidelity Simulation** | Four fidelity levels (L1–L4) for rapid iteration to production |
| **Three Control Laws** | PID, Feedback Linearization, FBL+NDOB implemented and benchmarked |
| **10-State EKF** | Optimal state estimation with adaptive noise covariance |
| **LuGre Friction Model** | Dynamic pre-sliding friction (vs. classical Tustin) |
| **DO-178C Compliant** | Level B aerospace software development standards |
| **Publication-Quality Outputs** | 300 DPI figures with LaTeX typography for IEEE/AIAA papers |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MicroPrecisionGimbal Digital Twin                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Target      ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐    │
│  Command ───►│  Coarse  │──►│   EKF    │──►│  Coarse  │──►│  Gimbal  │──┐ │
│              │ Pointing │   │Estimator │   │Controller│   │ Dynamics │  │ │
│              │ Assembly │   └──────────┘   └──────────┘   └──────────┘  │ │
│              └──────────┘                                                │ │
│                   │ Handover (<0.8°)                                     │ │
│                   ▼                                                      │ │
│              ┌──────────┐   ┌──────────┐   ┌──────────┐                 │ │
│              │   FSM    │──►│   FSM    │──►│   FSM    │                 │ │
│              │Controller│   │ Dynamics │   │ Actuator │                 │ │
│              └──────────┘   └──────────┘   └──────────┘                 │ │
│                   │                                                      │ │
│                   └──────────────────────────────────────────────────────┘ │
│                                      │                                     │
│                              Line-of-Sight Output                          │
│                              (< 2 µrad RMS)                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Control Hierarchy

| Stage | Component | Bandwidth | Range | Accuracy |
|-------|-----------|-----------|-------|----------|
| **Coarse** | 2-Axis Gimbal (CPA) | 10 Hz | ±90° | ±0.8° |
| **Fine** | Fast Steering Mirror (FSM) | 1 kHz | ±400 µrad | <2 µrad RMS |

### FBL + NDOB Control Law

```
τ = M(q)·v + C(q,q̇)·q̇ + G(q) − d̂
```

`d̂` is the real-time disturbance estimate (friction + vibration) from the Nonlinear Disturbance Observer.  
The same Lagrangian model is shared between simulation physics and the controller.

---

## Quick Start

### Prerequisites

- Python 3.10+
- 8 GB RAM minimum (16 GB recommended for L4 fidelity)
- Windows 10/11, Linux, or macOS

### Installation

```bash
git clone https://github.com/ShahidMustafa-PhD/MicroPrecisionGimbal.git
cd MicroPrecisionGimbal

python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### Run Your First Simulation

```bash
# Three-way controller comparison (PID vs FBL vs FBL+NDOB) — reproduces all figures
python demo_feedback_linearization.py

# Frequency response analysis
python demo_frequency_response.py
```

Figures are written to `figures_comparative/`. A pre-generated gallery is available at [docs/RESULTS.md](docs/RESULTS.md).

---

## Controllers

### 1. PID (Baseline)

```python
config = SimulationConfig(
    controller_type='pid',
    controller_gains={'kp': [3.514, 1.320], 'ki': [15.464, 4.148], 'kd': [0.293, 0.059]}
)
```

### 2. Feedback Linearization (FBL)

```python
config = SimulationConfig(
    controller_type='fbl',
    controller_gains={'kp': [400.0, 400.0], 'kd': [40.0, 40.0]}
)
```

### 3. FBL + NDOB *(best performance)*

```python
from lasercom_digital_twin.core.n_dist_observer import NDOBConfig

config = SimulationConfig(
    controller_type='fbl_ndob',
    ndob_config=NDOBConfig(lambda_az=50.0, lambda_el=50.0, d_max=0.5, enable=True)
)
```

---

## Simulation Fidelity Levels

| Level | Duration | Timestep | RMS Gate | Use Case |
|-------|----------|----------|----------|----------|
| **L1** | 10 s | 10 ms | < 50 µrad | Unit testing, rapid iteration |
| **L2** | 20 s | 5 ms | < 20 µrad | Controller tuning |
| **L3** | 30 s | 2 ms | < 10 µrad | System validation |
| **L4** | 60 s | 1 ms | **< 2 µrad** | Production acceptance |

---

## Testing

```bash
# Unit tests
pytest lasercom_digital_twin/tests/ -v

# CI/CD regression tests (L4 fidelity — all three controllers)
pytest lasercom_digital_twin/core/ci_tests/ -v

# With coverage
pytest --cov=lasercom_digital_twin --cov-report=html
```

**Production pass criteria (L4):**
- RMS Pointing Error < 2.0 µrad
- Peak Error < 30.0 µrad
- FSM Saturation < 1.0 %
- No NaN / Inf in telemetry

---

## Project Structure

```
MicroPrecisionGimbal/
├── demo_feedback_linearization.py   # Main three-way comparison demo
├── demo_frequency_response.py       # Bode plot / frequency sweep
├── requirements.txt
│
├── lasercom_digital_twin/
│   └── core/
│       ├── controllers/             # PID, FBL, FSM control laws
│       ├── dynamics/                # Gimbal & FSM dynamics (Lagrangian)
│       ├── estimators/              # 10-state EKF
│       ├── sensors/                 # QPD, IMU noise models
│       ├── friction/                # LuGre & Tustin friction models
│       ├── plots/                   # Publication-quality plotting
│       └── simulation/              # DigitalTwinRunner orchestrator
│
├── docs/
│   ├── RESULTS.md                   # ← Full simulation results gallery
│   ├── figures/                     # Committed PNG figures (32 total)
│   ├── FEEDBACK_LINEARIZATION_GUIDE.md
│   ├── EKF_ADAPTIVE_TUNING_SUMMARY.md
│   └── CI_CD_Pipeline.md
│
└── config/
    └── fidelity_levels.json
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [RESULTS.md](docs/RESULTS.md) | Full simulation results gallery (32 figures) |
| [FEEDBACK_LINEARIZATION_GUIDE.md](docs/FEEDBACK_LINEARIZATION_GUIDE.md) | FBL theory & implementation |
| [EKF_ADAPTIVE_TUNING_SUMMARY.md](docs/EKF_ADAPTIVE_TUNING_SUMMARY.md) | Kalman filter tuning |
| [CI_CD_Pipeline.md](docs/CI_CD_Pipeline.md) | Test gates & regression workflow |
| [IMPLEMENTATION_SUMMARY.md](docs/IMPLEMENTATION_SUMMARY.md) | Signal flow & architecture |

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{microprecisiongimbal2026,
  author    = {Mustafa, S. Shahid},
  title     = {{MicroPrecisionGimbal}: Digital Twin for Satellite Laser Communication Pointing Systems},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/ShahidMustafa-PhD/MicroPrecisionGimbal}
}
```

---

## License

MIT License — see [LICENSE](LICENSE).

---

## Contact

**Dr. S. Shahid Mustafa**  
GitHub: [@ShahidMustafa-PhD](https://github.com/ShahidMustafa-PhD)

---

<p align="center"><b>Built for precision. Designed for space.</b></p>
