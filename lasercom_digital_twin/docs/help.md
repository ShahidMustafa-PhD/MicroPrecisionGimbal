# Developer Guide: Lasercom Digital Twin
## Execution Modes and Development Workflow

**Version:** 1.0
**Last Updated:** January 9, 2026
**Target Audience:** Control systems engineers, simulation developers, and test engineers

---

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Simulation Execution Modes](#simulation-execution-modes)
  - [Mode 1: Interactive/Visualization Mode](#mode-1-interactivevisualization-mode-developmentdebugging)
  - [Mode 2: Headless/Batch Mode](#mode-2-headlessbatch-mode-cicd)
  - [Mode 3: Regression Test Mode](#mode-3-regression-test-mode-verification)
- [Next Steps: Critical Development To-Do List](#next-steps-critical-development-to-do-list)
- [Further Development Process Guidance](#further-development-process-guidance)
  - [Parameter Management](#parameter-management)
  - [Subsystem Testing](#subsystem-testing)
  - [Documentation Standard](#documentation-standard)

---

## 🚀 Quick Start

This guide assumes you have set up the environment as specified in the `README.md`, either locally with `pip install -r requirements.txt` or using the provided `Dockerfile`.

### Initial Setup Verification
Before running any simulations, verify your environment is correctly configured:
```bash
# Activate your virtual environment
# On Linux/macOS:
# source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Run the regression test suite to confirm all components are integrated
pytest lasercom_digital_twin/core/ci_tests/ -v
```
If all tests pass, you are ready to proceed.

---

## 🎮 Simulation Execution Modes

The digital twin is designed to be executed in three distinct modes, each serving a critical role in the development and V&V lifecycle.

### Mode 1: Interactive/Visualization Mode (Development/Debugging)
**Purpose:** For local development, debugging control algorithms, and visually verifying physical interactions. This mode provides real-time 3D rendering of the simulation.

#### Execution Steps
To launch the simulation with the `MuJoCoVisualizer` active, use the `--visualize` command-line flag with the main runner script.

```bash
# Run a 30-second simulation at L4 fidelity with visualization
python -m lasercom_digital_twin.runner --fidelity L4 --duration 30 --visualize
```

**Note:** Ensure you are in the project root directory (`C:\Active_Projects\MicroPrecisionGimbal`) and that the virtual environment is activated. If you encounter module import errors, verify that `lasercom_digital_twin/runner.py` exists and that all dependencies are installed.

**Expected Outcome:**
1. A console window will display simulation logs.
2. A separate MuJoCo window will open, showing the 3D model of the gimbal.
3. The gimbal will move according to the control system's commands, and you can interact with the view (zoom, pan, rotate).
4. Upon completion, final performance metrics will be printed to the console.

**Interactive Controls (in MuJoCo Window):**
- **Mouse Drag (Left/Right/Middle):** Rotate/Pan/Zoom the camera.
- **Spacebar:** Pause or resume the simulation.
- **Backspace:** Reset the simulation to its initial state.

---

### Mode 2: Headless/Batch Mode (CI/CD)
**Purpose:** For automated testing, performance characterization, and large-scale parameter sweeps (e.g., Monte Carlo analysis) where no GUI is required. This is the standard mode for CI/CD pipelines.

#### Execution Steps
To run the simulation without a GUI, use the `--headless` flag. Output is directed to log files and/or JSON result files.

```bash
# Run a single 60-second headless simulation and save results
python -m lasercom_digital_twin.runner --fidelity L4 --duration 60 --headless --output results/run_01.json
```

**Docker Execution (CI/CD Environment):**
The Docker container is pre-configured to run in a headless environment using OSMesa for rendering.

```bash
# Execute a headless run inside the Docker container
docker run --rm -v "$(pwd)/results:/app/results" lasercom-digital-twin:latest \
  python -m lasercom_digital_twin.runner --fidelity L4 --headless --output /app/results/ci_run.json
```

**Expected Outcome:**
- The simulation runs entirely in the terminal without opening any windows.
- Progress and final metrics are printed to standard output.
- If the `--output` flag is used, a JSON file containing detailed telemetry and performance results is created.

---

### Mode 3: Regression Test Mode (Verification)
**Purpose:** To rigorously verify that code changes have not degraded the system's performance against mandatory, aerospace-grade requirements. This is the gatekeeper for all code merges.

#### Execution Steps
The regression suite is executed using `pytest`.

```bash
# Run the full regression test suite from the project root directory
pytest lasercom_digital_twin/core/ci_tests/ -v
```

**Interpreting the Results:**
- **PASS:** All performance metrics are within the specified thresholds. The code is safe to merge.
- **FAIL:** One or more performance metrics have violated their thresholds. **The code must not be merged.**

**Example of a Critical Failure:**
If the **RMS Pointing Error** exceeds its 2.0 µrad threshold, `pytest` will produce a detailed failure report:

```
================================== FAILURES ===================================
_________ TestRegressionSuite.test_rms_pointing_error_requirement _________

self = <test_regression.TestRegressionSuite object at 0x...>
simulation_results = {'los_error_rms': 2.347, ...}
performance_thresholds = {'rms_pointing_error_urad': 2.0, ...}

    def test_rms_pointing_error_requirement(self, simulation_results, performance_thresholds):
        """FAIL if RMS pointing error exceeds the requirement."""
        threshold = performance_thresholds['rms_pointing_error_urad']
        rms_error = simulation_results['los_error_rms']
>       assert rms_error < threshold, (
            f"PERFORMANCE REGRESSION: RMS Pointing Error {rms_error:.2f} µrad "
            f"exceeds threshold of {threshold} µrad."
        )
E       AssertionError: PERFORMANCE REGRESSION: RMS Pointing Error 2.34 µrad exceeds threshold of 2.0 µrad.
E       assert 2.347 < 2.0

lasercom_digital_twin/core/ci_tests/test_regression.py:150: AssertionError
=========================== 1 failed, 11 passed in 8.5s ===========================
```
A failure like this is a hard stop. The developer must investigate the root cause (e.g., a change in controller gains, sensor noise, or physics model) and fix it before the pull request can be considered.

---

## 📝 Next Steps: Critical Development To-Do List

This list outlines the prioritized engineering tasks required to advance the digital twin from its current state to a fully validated, production-ready framework.

| Module | To-Do Action | Developer Notes |
| :--- | :--- | :--- |
| **MuJoCo Dynamics** | Finalize Inertia Tensors | Replace placeholder mass/inertia values with **CAD-derived values** for all rigid bodies (`az_stage`, `el_stage`, `payload_head`). |
| **Actuators** | Calibrate Non-Ideals | Tune the parameters for **Cogging Torque** (magnitude/period) and **Hysteresis** (FSM) against expected hardware specifications. |
| **Optics** | Validate Misalignment | Implement the explicit mapping for **thermal-induced misalignment errors** and confirm the K-Mirror transformation handles the Field Rotation accurately. |
| **Estimators** | EKF Linearization Check | Verify the process and measurement Jacobians ($\mathbf{F}, \mathbf{H}$) are correctly implemented for the **EKF**, focusing on the non-linear QPD measurement model. |
| **Controllers** | LQR Implementation | Replace the placeholder PID in the `CoarseGimbalController` with a robust **LQR or $\mathcal{H}_{\infty}$ controller** implementation. |
| **Simulation** | Real-Time/HIL Interface | Implement the adapter class necessary to ingest **external sensor data** (e.g., from a UDP port) for Hardware-in-the-Loop testing, as outlined in the `README`. |

---

## 🛠️ Further Development Process Guidance

To ensure consistency, maintainability, and ease of development, all engineers must adhere to the following processes.

### Parameter Management
**Rule:** All physical constants, controller gains, noise parameters, and simulation settings **must** be managed in the central configuration files located in the `lasercom_digital_twin/config/` directory.

- **DO NOT** hardcode numerical values directly in Python modules.
- **DO** load parameters from the appropriate JSON/YAML file at initialization.
- **Example:** To change the coarse loop controller gain, modify `config/controller_gains.json`, not the `CoarseGimbalController` class.

This approach ensures that:
1. Parameters are easily searchable and modifiable.
2. Fidelity levels can be managed centrally.
3. The simulation's behavior is defined entirely by its configuration, promoting reproducibility.

### Subsystem Testing
**Rule:** Any modification to a core subsystem (`actuators/`, `sensors/`, `estimators/`, `controllers/`) requires a corresponding unit test.

- Before running the full closed-loop simulation, write and execute a focused `pytest` test for the modified component.
- This isolates failures, speeds up debugging, and ensures that individual components behave as expected before they are integrated.
- **Example:** If you modify the `GimbalMotorModel`, run `pytest tests/unit/test_motor_models.py` to verify its torque output and friction model before running a full regression test.

### Documentation Standard
**Rule:** All new code must include comprehensive docstrings and type hints. This is non-negotiable for maintaining a professional, aerospace-grade codebase.

- **Type Hints:** All function signatures and class attributes must be fully type-hinted.
- **Docstrings:** Use the NumPy docstring format for all public modules, classes, and functions. It must include `Parameters`, `Returns`, and `Examples` sections.

**Example of a compliant function:**
```python
from typing import Tuple
import numpy as np

def calculate_los_error(
    qpd_voltage: np.ndarray,
    sensitivity_v_urad: float
) -> Tuple[float, float]:
    """Converts QPD voltage to a line-of-sight error in microradians.

    Parameters
    ----------
    qpd_voltage : np.ndarray
        A 2-element array containing the [X, Y] voltages from the QPD.
    sensitivity_v_urad : float
        The sensor's sensitivity in Volts per microradian.

    Returns
    -------
    Tuple[float, float]
        A tuple containing the (azimuth_error_urad, elevation_error_urad).
    """
    if qpd_voltage.shape != (2,):
        raise ValueError("qpd_voltage must be a 2-element array.")
    
    error_urad = qpd_voltage / sensitivity_v_urad
    return (error_urad[0], error_urad[1])
```
