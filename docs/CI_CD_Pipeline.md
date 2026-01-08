# CI/CD Pipeline Documentation
# MicroPrecisionGimbal Digital Twin - Production Deployment Guide

## Overview

This document describes the CI/CD infrastructure for the MicroPrecisionGimbal Digital Twin, including Docker containerization, headless execution, fidelity-based testing, and automated regression validation.

**Compliance Standards:**
- DO-178C Level B: Software Considerations in Airborne Systems
- NASA-STD-8739.8: Software Assurance and Safety
- CCSDS 141.0-B-1: Optical Communications Coding and Synchronization

---

## 1. Architecture Overview

The CI/CD pipeline is built on three core components:

```
┌─────────────────────────────────────────────────────────────┐
│                     CI/CD Pipeline                           │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────┐     ┌──────────────┐     ┌──────────────┐ │
│  │   Docker    │────▶│   Headless   │────▶│  Regression  │ │
│  │ Environment │     │   Execution  │     │    Tests     │ │
│  └─────────────┘     └──────────────┘     └──────────────┘ │
│        │                     │                     │         │
│        ▼                     ▼                     ▼         │
│  MuJoCo + Python      Xvfb Virtual       L4 Fidelity       │
│  Dependencies         Display            Validation         │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Key Features

1. **Containerized Environment**: Full MuJoCo simulation stack in Docker
2. **Headless Rendering**: Virtual display (Xvfb) for CI/CD compatibility
3. **Fidelity Levels**: L1 (quick) → L4 (production) graduated testing
4. **Mandatory Thresholds**: Tests FAIL if performance degrades

---

## 2. Docker Configuration

### Dockerfile Structure

The production Dockerfile uses a multi-stage build for optimization:

**Stage 1: Base Environment**
- Python 3.11 slim base
- System dependencies (OpenGL, GLFW, Xvfb)
- Headless display configuration

**Stage 2: Python Dependencies**
- Install from `requirements.txt`
- Add CI/CD packages (pytest, coverage)

**Stage 3: Application Layer**
- Copy codebase
- Set up directory structure

**Stage 4: Runtime Configuration**
- Health checks
- Entrypoint configuration

### Building the Image

```bash
# Standard build
docker build -t lasercom-digital-twin:latest .

# Build with specific Python version
docker build --build-arg PYTHON_VERSION=3.11 -t lasercom-digital-twin:3.11 .

# Build with cache optimization for faster CI
docker build --cache-from lasercom-digital-twin:latest -t lasercom-digital-twin:latest .
```

### Running the Container

```bash
# Run regression tests (default)
docker run --rm lasercom-digital-twin:latest

# Run specific test file
docker run --rm lasercom-digital-twin:latest pytest core/ci_tests/test_regression.py -v

# Run all tests with coverage
docker run --rm lasercom-digital-twin:latest pytest --cov=lasercom_digital_twin --cov-report=html

# Mount results directory
docker run --rm -v $(pwd)/results:/app/results lasercom-digital-twin:latest

# Interactive shell for debugging
docker run --rm -it lasercom-digital-twin:latest bash
```

### Environment Variables

The Docker environment sets the following critical variables:

```bash
DISPLAY=:99                    # Virtual display number
MUJOCO_GL=osmesa              # MuJoCo rendering backend (CPU-based)
PYOPENGL_PLATFORM=osmesa      # PyOpenGL platform
MPLBACKEND=Agg                # Non-interactive matplotlib backend
PYTHONUNBUFFERED=1            # Real-time log streaming
```

---

## 3. Headless Execution

### run_headless.sh Script

The headless execution wrapper (`scripts/run_headless.sh`) manages the virtual display lifecycle:

**Responsibilities:**
1. Start Xvfb virtual display server
2. Verify display availability
3. Configure MuJoCo for headless rendering
4. Execute simulation/tests
5. Clean up display on exit

### Usage

```bash
# Make executable (first time)
chmod +x scripts/run_headless.sh

# Run regression tests
./scripts/run_headless.sh pytest core/ci_tests/test_regression.py -v

# Run simulation with specific fidelity
./scripts/run_headless.sh python -m lasercom_digital_twin.runner --fidelity L4

# Run with custom display settings
XVFB_RESOLUTION=1280x720x24 ./scripts/run_headless.sh pytest core/ci_tests/
```

### Troubleshooting Headless Issues

**Problem: "Cannot open display :99"**
```bash
# Check if Xvfb is running
ps aux | grep Xvfb

# Manually start Xvfb
Xvfb :99 -screen 0 1920x1080x24 -ac &
export DISPLAY=:99
```

**Problem: "MuJoCo rendering failed"**
```bash
# Verify osmesa backend
python -c "import mujoco; print(mujoco.__version__)"
export MUJOCO_GL=osmesa
```

---

## 4. Fidelity Levels

### Configuration: `config/fidelity_levels.json`

The digital twin supports four fidelity levels for graduated testing:

| Level | Name | Duration | Timestep | Use Case | Performance Threshold |
|-------|------|----------|----------|----------|----------------------|
| **L1** | Quick Test | 10 s | 10 ms | Unit tests, rapid iteration | RMS < 50 µrad |
| **L2** | Integration Test | 20 s | 5 ms | Integration testing | RMS < 20 µrad |
| **L3** | System Test | 30 s | 2 ms | System validation | RMS < 10 µrad |
| **L4** | Production | 60 s | 1 ms | Acceptance testing | RMS < 2 µrad |

### Key Differences by Fidelity

**L1 (Quick Test):**
- Linear actuators
- No sensor noise
- No disturbances
- Euler integration
- **Purpose**: Catch catastrophic failures fast (< 5 seconds)

**L2 (Integration Test):**
- First-order actuators
- Low-level noise
- Basic disturbances
- RK4 integration
- **Purpose**: Validate component interactions

**L3 (System Test):**
- High-fidelity nonlinear models
- Realistic noise and disturbances
- Multi-frequency base motion
- **Purpose**: Performance verification

**L4 (Production):**
- Maximum fidelity
- Flight-representative environment
- Strict performance requirements
- **Purpose**: Acceptance and regression testing

### Loading Fidelity Configuration

```python
import json
from pathlib import Path

# Load configuration
config_path = Path("config/fidelity_levels.json")
with open(config_path, 'r') as f:
    config = json.load(f)

# Get L4 parameters
l4_params = config["fidelity_levels"]["L4"]["parameters"]

# Extract thresholds
thresholds = l4_params["performance_thresholds"]
rms_threshold = thresholds["rms_pointing_error_urad"]  # 2.0 µrad
```

---

## 5. Regression Test Suite

### Test Structure: `core/ci_tests/test_regression.py`

The regression suite implements **mandatory performance gates** that must pass before code can be merged.

### Critical Tests

#### 5.1 RMS Pointing Error Requirement

**Test**: `test_rms_pointing_error_requirement()`

**Threshold**: RMS < 2.0 µrad (L4)

**Failure Mode**: Performance regression → Link budget violation

**Example Failure Output**:
```
================================================================================
RMS POINTING ERROR REQUIREMENT VIOLATION
================================================================================
  Measured RMS:  2.347 µrad
  Threshold:     2.000 µrad
  Margin:        -0.347 µrad (NEGATIVE - FAILURE)

This indicates PERFORMANCE REGRESSION in the control system.
Root causes may include:
  - Controller gain changes
  - Estimator tuning issues
  - Increased disturbance levels
  - Actuator model changes

ACTION REQUIRED: Investigate and restore performance before merge.
================================================================================
```

#### 5.2 Numerical Stability Checks

**Test**: `test_no_nan_in_telemetry()`, `test_no_inf_in_telemetry()`

**Requirement**: No NaN or Inf in state vectors

**Failure Mode**: Numerical instability → Divergence

**Why This Matters**: Flight software MUST be numerically stable. NaN/Inf indicates:
- Integration method issues
- Unbounded state growth
- Division by zero
- Controller saturation handling bugs

#### 5.3 FSM Saturation Limit

**Test**: `test_fsm_saturation_limit()`

**Threshold**: Saturation < 1.0% of simulation time

**Failure Mode**: Insufficient coarse loop authority

**Interpretation**:
- FSM saturating frequently means coarse gimbal isn't nulling error
- Degrades fine pointing performance
- Indicates coarse/fine handoff issues

### Running Regression Tests

```bash
# Run all regression tests
pytest core/ci_tests/test_regression.py -v

# Run with detailed output
pytest core/ci_tests/test_regression.py -v --tb=long

# Run only fast tests (exclude extended duration)
pytest core/ci_tests/test_regression.py -v -m "not slow"

# Generate HTML coverage report
pytest core/ci_tests/test_regression.py --cov=lasercom_digital_twin --cov-report=html

# Run in Docker
docker run --rm lasercom-digital-twin:latest pytest core/ci_tests/test_regression.py -v
```

### Expected Output (Passing)

```
============================== test session starts ==============================
platform linux -- Python 3.11.x, pytest-7.4.x, pluggy-1.x
collected 10 items

core/ci_tests/test_regression.py::TestRegressionSuite::test_simulation_completes_successfully PASSED [10%]
✓ Simulation completed successfully

core/ci_tests/test_regression.py::TestRegressionSuite::test_no_nan_in_telemetry PASSED [20%]
✓ No NaN values detected in telemetry

core/ci_tests/test_regression.py::TestRegressionSuite::test_no_inf_in_telemetry PASSED [30%]
✓ No Inf values detected in telemetry

core/ci_tests/test_regression.py::TestRegressionSuite::test_rms_pointing_error_requirement PASSED [40%]
✓ RMS Pointing Error: 1.534 µrad (Threshold: 2.000 µrad)
  Margin: 23.3%

core/ci_tests/test_regression.py::TestRegressionSuite::test_peak_pointing_error_requirement PASSED [50%]
✓ Peak Pointing Error: 24.821 µrad (Threshold: 30.000 µrad)
  Margin: 17.3%

core/ci_tests/test_regression.py::TestRegressionSuite::test_fsm_saturation_limit PASSED [60%]
✓ FSM Saturation: 0.52% (Threshold: 1.00%)

============================== 6 passed in 45.23s ===============================
```

---

## 6. CI/CD Pipeline Integration

### GitHub Actions Example

```yaml
name: Regression Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  regression-tests:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v3
    
    - name: Build Docker image
      run: docker build -t lasercom-digital-twin:ci .
    
    - name: Run regression tests
      run: |
        docker run --rm \
          -v ${{ github.workspace }}/results:/app/results \
          lasercom-digital-twin:ci \
          pytest core/ci_tests/test_regression.py -v --tb=short
    
    - name: Upload test results
      if: always()
      uses: actions/upload-artifact@v3
      with:
        name: test-results
        path: results/
```

### GitLab CI Example

```yaml
stages:
  - build
  - test
  - deploy

variables:
  DOCKER_IMAGE: $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA

build:
  stage: build
  script:
    - docker build -t $DOCKER_IMAGE .
    - docker push $DOCKER_IMAGE

regression_tests:
  stage: test
  script:
    - docker pull $DOCKER_IMAGE
    - docker run --rm $DOCKER_IMAGE pytest core/ci_tests/test_regression.py -v
  artifacts:
    when: always
    paths:
      - results/
    expire_in: 30 days
```

---

## 7. Performance Monitoring

### Tracking Performance Over Time

Create a CI job to track performance metrics across commits:

```bash
# Run regression and extract metrics
docker run --rm lasercom-digital-twin:latest \
  pytest core/ci_tests/test_regression.py -v --json-report --json-report-file=results/metrics.json

# Parse metrics
python scripts/parse_metrics.py results/metrics.json >> performance_history.csv

# Generate trend plot
python scripts/plot_performance_trends.py performance_history.csv
```

### Metrics to Track

1. **RMS Pointing Error** (µrad) - Primary performance metric
2. **Peak Pointing Error** (µrad) - Transient behavior
3. **FSM Saturation %** - Control authority usage
4. **Settling Time** (s) - Response speed
5. **Test Execution Time** (s) - CI efficiency

---

## 8. Failure Diagnosis Workflow

### When Regression Tests Fail

**Step 1: Identify Failed Test**
```bash
# Re-run with verbose output
pytest core/ci_tests/test_regression.py -v --tb=long
```

**Step 2: Extract Telemetry**
```python
# In test, save telemetry for analysis
import pickle
with open('failed_telemetry.pkl', 'wb') as f:
    pickle.dump(simulation_results['telemetry'], f)
```

**Step 3: Generate Debug Plots**
```python
from lasercom_digital_twin.core.visualization.time_series_plots import TimelinePlotter

plotter = TimelinePlotter()
fig = plotter.plot_full_debug_suite(telemetry)
fig.savefig('debug_failure.png', dpi=300)
```

**Step 4: Compare with Baseline**
- Load baseline telemetry from last passing commit
- Run side-by-side comparison
- Identify divergence point

**Step 5: Root Cause Analysis**

**If RMS Error Increased:**
- Check controller gains (changed?)
- Check estimator convergence (slower?)
- Check disturbance levels (higher?)
- Check actuator models (modified?)

**If NaN/Inf Detected:**
- Check integration timestep (too large?)
- Check state initialization (valid?)
- Check saturation logic (working?)
- Run with smaller dt to isolate instability

**If FSM Saturating:**
- Check coarse loop bandwidth (too low?)
- Check coarse loop gains (insufficient?)
- Check disturbance magnitude (excessive?)

---

## 9. Best Practices

### Development Workflow

1. **Local Testing** (L1/L2): Fast iteration with quick fidelity
   ```bash
   pytest lasercom_digital_twin/tests/ -v  # Unit tests
   ```

2. **Pre-Commit** (L3): System validation before push
   ```bash
   pytest core/ci_tests/test_regression.py -v -m "not slow"
   ```

3. **CI/CD** (L4): Full regression on every PR
   ```bash
   docker run --rm lasercom-digital-twin:latest
   ```

### Tuning Performance

**If struggling to meet thresholds:**

1. **Increase controller bandwidth** (L4 config):
   ```json
   "controller": {
     "gains": {
       "coarse_kp": 75.0,  // Increase from 50.0
       "fine_kp": 150.0     // Increase from 100.0
     }
   }
   ```

2. **Reduce process noise** (estimator tuning):
   ```json
   "estimator": {
     "process_noise_scale": 1.0  // Reduce from 2.0
   }
   ```

3. **Verify disturbance models** (not over-conservative):
   ```json
   "disturbances": {
     "pointing_disturbance": {
       "rms": 10.0  // Check if realistic
     }
   }
   ```

---

## 10. Deployment Checklist

Before deploying to production CI/CD:

- [ ] Docker image builds successfully
- [ ] Headless execution works in container
- [ ] All L4 regression tests pass
- [ ] Performance metrics within thresholds:
  - [ ] RMS < 2.0 µrad
  - [ ] Peak < 30.0 µrad
  - [ ] FSM saturation < 1.0%
- [ ] No NaN/Inf in telemetry
- [ ] Test execution time < 5 minutes
- [ ] Configuration files validated
- [ ] Documentation updated
- [ ] Baseline performance recorded

---

## 11. Contact and Support

**Technical Lead**: Senior Aerospace Control Systems Engineer

**Escalation Path**:
1. Check this documentation
2. Review test failure output
3. Run local debug plots
4. Consult performance history
5. Escalate to control systems team

**Key Files**:
- `Dockerfile`: Container configuration
- `scripts/run_headless.sh`: Headless execution wrapper
- `config/fidelity_levels.json`: Fidelity parameters
- `core/ci_tests/test_regression.py`: Regression test suite

---

## Appendix A: Requirements Traceability

| Requirement ID | Description | Test Coverage | Threshold |
|----------------|-------------|---------------|-----------|
| REQ-POINT-001 | RMS pointing accuracy | `test_rms_pointing_error_requirement` | 2.0 µrad |
| REQ-POINT-002 | Peak transient error | `test_peak_pointing_error_requirement` | 30.0 µrad |
| REQ-STAB-001 | Numerical stability | `test_no_nan_in_telemetry` | 0 NaN |
| REQ-STAB-002 | Bounded behavior | `test_no_inf_in_telemetry` | 0 Inf |
| REQ-CTRL-001 | FSM authority usage | `test_fsm_saturation_limit` | < 1% |
| REQ-PERF-001 | Settling time | `test_settling_time_reasonable` | < 10 s |

---

## Appendix B: Fidelity Selection Guide

**Use L1 when:**
- Writing new unit tests
- Rapid prototyping
- Debugging basic logic
- Quick sanity checks

**Use L2 when:**
- Integration testing
- Controller tuning
- Subsystem validation
- Development testing

**Use L3 when:**
- System validation
- Pre-release testing
- Performance verification
- Monte Carlo prep

**Use L4 when:**
- Acceptance testing
- Regression validation
- Flight readiness review
- Formal V&V

---

**Document Version**: 1.0.0  
**Last Updated**: January 9, 2026  
**Compliance**: DO-178C Level B, NASA-STD-8739.8
