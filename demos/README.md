# Demos & Development Scripts

This directory contains secondary scripts for debugging, testing, analysis, and development purposes. These are not primary entry points but are useful for development and troubleshooting.

## Directory Structure

### 🔬 Analysis Scripts

| Script | Purpose |
|--------|---------|
| `analyze_ndob.py` | Analyze Nonlinear Disturbance Observer (NDOB) performance and tuning |
| `compare_dynamics.py` | Compare different dynamics model implementations |
| `compare_pid_fbl.py` | Head-to-head PID vs Feedback Linearization comparison |
| `compare_velocities.py` | Analyze velocity estimation and clipping behavior |
| `diagnose_velocity.py` | Diagnostic tool for velocity signal issues |

### 🐛 Debug Scripts

| Script | Purpose |
|--------|---------|
| `debug_demo_telemetry.py` | Debug telemetry logging in demo simulations |
| `debug_fbl_detailed.py` | Detailed FBL controller debugging with step-by-step output |
| `debug_ndob.py` | Debug NDOB disturbance estimation pipeline |
| `debug_ndob_values.py` | Inspect NDOB internal state values |
| `debug_ss_error.py` | Debug steady-state error sources |
| `debug_tracking_flow.py` | Trace signal flow through tracking control loop |

### 🧪 Test Scripts

| Script | Purpose |
|--------|---------|
| `test_ekf_qualification.py` | Extended Kalman Filter qualification tests |
| `test_ekf_standalone.py` | Standalone EKF unit tests |
| `test_ekf_true_dynamics.py` | EKF tests with true dynamics (no linearization) |
| `test_fbl_configs.py` | Test various FBL configuration combinations |
| `test_fbl_direct.py` | Direct FBL controller unit tests |
| `test_fbl_sign.py` | Test FBL sign conventions and coordinate frames |
| `test_linearization.py` | Test gimbal linearization accuracy |
| `test_signal_trace.py` | Signal tracing through control pipeline |
| `test_velocity_clipping.py` | Velocity clipping boundary condition tests |

### 📊 Visualization & Figure Generation

| Script | Purpose |
|--------|---------|
| `generatefigs.py` | Generate standard figure set for reports |
| `generate_bode_plots.py` | Generate frequency response Bode plots |
| `compile_diagrams.py` | Compile block diagrams from Graphviz sources |
| `diagram_template.py` | Template for creating new block diagrams |

### 🔧 Utility Scripts

| Script | Purpose |
|--------|---------|
| `quick_test.py` | Rapid sanity check for development iterations |
| `verify_handover.py` | Verify CPA→FSM handover logic and thresholds |
| `demo_velocity_clipping_final.py` | Final velocity clipping implementation demo |
| `demo_feedback_linearization_backup.py` | Backup of original feedback linearization demo |

---

## Usage Examples

### Running Analysis Scripts

```bash
# From project root
cd demos/

# Analyze NDOB performance
python analyze_ndob.py

# Compare controller implementations
python compare_pid_fbl.py
```

### Running Debug Scripts

```bash
# Debug steady-state error
python debug_ss_error.py

# Detailed FBL debugging with verbose output
python debug_fbl_detailed.py
```

### Running Test Scripts

```bash
# Run individual test
python test_fbl_configs.py

# Or use pytest for proper test execution
cd ..
pytest demos/test_ekf_qualification.py -v
```

---

## Notes

- These scripts are **not** part of the main simulation API
- For production use, see the main entry points in the repository root:
  - `demo_feedback_linearization.py` - Three-way controller comparison
  - `demo_frequency_response.py` - Frequency response analysis
- Test scripts here are development tests; official CI/CD tests are in `lasercom_digital_twin/ci_tests/`
