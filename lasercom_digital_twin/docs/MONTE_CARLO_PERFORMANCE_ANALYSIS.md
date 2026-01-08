# Monte Carlo Analysis & Performance Metrics Implementation

## Summary

This document describes the implementation of industrial-grade Monte Carlo simulation framework and performance analysis modules for the laser communication terminal digital twin.

---

## Modules Implemented

### 1. Performance Analyzer (`core/simulation/performance_analyzer.py`)

**Purpose**: Compute aerospace-standard performance metrics from simulation telemetry.

**Key Features**:
- **Primary Metrics**:
  - RMS Pointing Error (µrad) - CRITICAL performance metric
  - Peak Pointing Error (µrad) - transient performance
  - FSM Saturation Percentage (%) - control authority usage
  - Stability Margins (damping ratio, settling time)
  
- **Additional Metrics**:
  - Component-wise errors (X/Y axes)
  - Estimation convergence and accuracy
  - Control effort (torque statistics)
  - Tracking performance (steady-state error, jitter)
  
- **Capabilities**:
  - Automatic unit detection (rad vs µrad)
  - Time-window analysis
  - Requirement pass/fail assessment
  - Human-readable report generation
  - DataFrame export for batch analysis

**API Example**:
```python
from lasercom_digital_twin.core.simulation.performance_analyzer import PerformanceAnalyzer

analyzer = PerformanceAnalyzer(
    rms_requirement=10.0,  # µrad
    peak_requirement=50.0,  # µrad
    fsm_limit=400.0  # µrad
)

metrics = analyzer.analyze(telemetry)
print(f"RMS Error: {metrics.rms_pointing_error:.2f} µrad")
print(f"Passes: {metrics.meets_rms_requirement}")

report = analyzer.generate_report(metrics)
print(report)
```

---

### 2. Monte Carlo Engine (`core/simulation/monte_carlo_engine.py`)

**Purpose**: Orchestrate batch uncertainty quantification analysis with systematic parameter randomization.

**Key Features**:
- **Parameter Randomization**:
  - Uniform, Normal, Truncated Normal, Lognormal distributions
  - Bounds enforcement
  - Deterministic seeding for reproducibility
  
- **Batch Execution**:
  - Configurable number of runs
  - Automatic parameter injection
  - Progress tracking
  - Graceful failure handling
  
- **Statistical Aggregation**:
  - Mean, std, min, max, median, 95th percentile
  - Pass rate calculation
  - Performance margin assessment
  
- **Output Management**:
  - Comprehensive reports
  - CSV export (metrics, parameters, telemetry)
  - JSON configuration files

**API Example**:
```python
from lasercom_digital_twin.core.simulation.monte_carlo_engine import (
    MonteCarloEngine, MonteCarloConfig, ParameterUncertainty, DistributionType
)

# Define uncertainties
uncertainties = [
    ParameterUncertainty('motor_az.K_t', 0.1, DistributionType.NORMAL, 5.0),
    ParameterUncertainty('gyro_az.noise_std', 1e-5, DistributionType.LOGNORMAL, 30.0),
]

# Configure MC analysis
config = MonteCarloConfig(
    n_runs=100,
    parameter_uncertainties=uncertainties,
    simulation_duration=10.0
)

# Run batch
engine = MonteCarloEngine(config, analyzer, verbose=True)
results = engine.run_batch(simulation_factory)

# Generate report
print(engine.generate_report(results))
```

---

## Default Uncertainty Parameters

The module includes a `create_default_uncertainties()` function that defines standard parameter variations:

| Parameter | Nominal | Distribution | Uncertainty | Description |
|-----------|---------|--------------|-------------|-------------|
| `motor_az.K_t` | 0.1 N·m/A | Normal | ±5% | Motor torque constant |
| `motor_az.R` | 1.0 Ω | Normal | ±3% | Motor resistance |
| `gyro_az.noise_std` | 1e-5 rad/s | Lognormal | 30% rel | Gyro noise density |
| `gyro_az.bias` | 0 rad/s | Uniform | ±1e-4 | Gyro bias offset |
| `encoder_az.resolution` | 1e-6 rad | Uniform | ±10% | Encoder resolution |
| `fsm.bandwidth` | 100 Hz | Normal | ±10% | FSM mechanical bandwidth |
| `gimbal_az.backlash` | 1e-5 rad | Lognormal | 50% rel | Gimbal backlash |
| `gimbal_az.friction` | 0.1 N·m·s/rad | Normal | ±20% | Viscous friction |

---

## Testing

### Test Coverage

**PerformanceAnalyzer Tests** (`tests/test_performance_analyzer.py`):
- 17 tests covering:
  - Metric calculation accuracy (RMS, peak, saturation)
  - Unit detection (rad/µrad conversion)
  - Time-window analysis
  - Requirement pass/fail assessment
  - Report generation and DataFrame export
  - Stability and tracking metrics
  - Control effort statistics

**MonteCarloEngine Tests** (`tests/test_monte_carlo_engine.py`):
- 19 tests covering:
  - Parameter sampling determinism
  - Distribution types (Uniform, Normal, Lognormal, Truncated)
  - Bounds enforcement
  - Batch execution
  - Statistical aggregation
  - Failed run handling
  - Output file generation
  - Configuration injection

### Test Results
```
========================== test session starts ==========================
lasercom_digital_twin/tests/test_performance_analyzer.py::TestPerformanceAnalyzer
  ✓ 17 tests passed

lasercom_digital_twin/tests/test_monte_carlo_engine.py::TestParameterRandomizer
  ✓ 8 tests passed

lasercom_digital_twin/tests/test_monte_carlo_engine.py::TestMonteCarloEngine
  ✓ 11 tests passed

========================== 36 passed, 6 warnings in 1.12s ==========================
```

---

## Demonstration Script

A comprehensive demonstration script is provided at [examples/demo_monte_carlo_analysis.py](../examples/demo_monte_carlo_analysis.py).

**Features**:
- Define parameter uncertainties across motor, sensor, and structural subsystems
- Execute 100 Monte Carlo runs with parameter randomization
- Compute performance statistics (mean, std, 95th percentile)
- Generate 6-panel visualization:
  1. RMS error histogram
  2. Peak error histogram
  3. FSM saturation histogram
  4. RMS vs motor torque constant
  5. RMS vs gyro noise
  6. Cumulative distribution function
- Statistical summary table
- Performance margin assessment

**Usage**:
```bash
python lasercom_digital_twin/examples/demo_monte_carlo_analysis.py
```

**Expected Output**:
- Console report with execution summary, parameter uncertainties, and performance statistics
- High-resolution figure (`monte_carlo_analysis.png`) with 6 subplots
- Statistical summary table showing mean, std, min, max, 95th percentile, and pass rates

---

## Integration with Digital Twin

To integrate the Monte Carlo framework with your existing simulation runner:

### 1. Create Simulation Factory

```python
def simulation_factory(config, seed):
    """
    Factory function for Monte Carlo batch execution.
    
    Parameters
    ----------
    config : Dict
        Configuration with randomized parameters
    seed : int
        Random seed for this run
        
    Returns
    -------
    runner : DigitalTwinRunner
        Configured simulation runner
    telemetry : Dict
        Simulation telemetry data
    """
    from lasercom_digital_twin.core.simulation.simulation_runner import (
        DigitalTwinRunner, SimulationConfig
    )
    
    # Create simulation config with randomized parameters
    sim_config = SimulationConfig(seed=seed, **config)
    
    # Run simulation
    runner = DigitalTwinRunner(sim_config)
    telemetry = runner.run_simulation(duration=config['simulation_duration'])
    
    return runner, telemetry
```

### 2. Execute Monte Carlo Batch

```python
from lasercom_digital_twin.core.simulation.monte_carlo_engine import (
    MonteCarloEngine, MonteCarloConfig, create_default_uncertainties
)
from lasercom_digital_twin.core.simulation.performance_analyzer import (
    PerformanceAnalyzer
)

# Define uncertainties (or use defaults)
uncertainties = create_default_uncertainties()

# Configure MC
mc_config = MonteCarloConfig(
    n_runs=100,
    base_seed=42,
    parameter_uncertainties=uncertainties,
    simulation_duration=10.0,
    output_dir=Path('./mc_results')
)

# Create analyzer
analyzer = PerformanceAnalyzer(
    rms_requirement=10.0,
    peak_requirement=50.0,
    fsm_limit=400.0
)

# Run batch
engine = MonteCarloEngine(mc_config, analyzer, verbose=True)
results = engine.run_batch(simulation_factory)

# Generate report
print(engine.generate_report(results))

# Access results
print(f"Pass rate: {results.summary_statistics['pass_rate_rms'].iloc[0]:.1f}%")
print(f"Mean RMS: {results.summary_statistics['rms_pointing_error_mean'].iloc[0]:.2f} µrad")
print(f"95th %ile: {results.summary_statistics['rms_pointing_error_p95'].iloc[0]:.2f} µrad")
```

---

## Performance Metrics Details

### Primary Metrics (Aerospace Standards)

#### 1. RMS Pointing Error
**Definition**: Root mean square of total LOS error magnitude  
**Formula**: $\text{RMS} = \sqrt{\frac{1}{N}\sum_{i=1}^{N} (e_{x,i}^2 + e_{y,i}^2)}$  
**Units**: µrad  
**Typical Requirement**: ≤10 µrad for deep-space laser comm

#### 2. Peak Pointing Error
**Definition**: Maximum instantaneous LOS error magnitude  
**Formula**: $\text{Peak} = \max_i \sqrt{e_{x,i}^2 + e_{y,i}^2}$  
**Units**: µrad  
**Typical Requirement**: ≤50 µrad (5× RMS)

#### 3. FSM Saturation Percentage
**Definition**: Percentage of time FSM commands exceed limit  
**Formula**: $\text{Sat} = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}[\|\mathbf{c}_i\| \geq c_{\text{lim}}] \times 100\%$  
**Units**: %  
**Typical Requirement**: ≤20-30% (adequate control authority margin)

#### 4. Stability Margin
**Definition**: Damping ratio from step response or frequency analysis  
**Typical Requirement**: ζ ≥ 0.5 (adequate damping)

---

## File Structure

```
lasercom_digital_twin/
├── core/
│   └── simulation/
│       ├── performance_analyzer.py      (~600 lines) ✓ IMPLEMENTED
│       └── monte_carlo_engine.py        (~800 lines) ✓ IMPLEMENTED
├── tests/
│   ├── test_performance_analyzer.py     (17 tests)  ✓ 100% PASS
│   └── test_monte_carlo_engine.py       (19 tests)  ✓ 100% PASS
└── examples/
    └── demo_monte_carlo_analysis.py     (~350 lines) ✓ IMPLEMENTED
```

---

## Dependencies

Required Python packages:
- `numpy` (array operations, statistics)
- `pandas` (DataFrame export, CSV operations)
- `matplotlib` (visualization in demo script)
- `pytest` (testing framework)

All dependencies are compatible with Python 3.9+.

---

## Standards Compliance

The implementation follows aerospace industry standards:

- **MIL-STD-1540E**: Test Requirements for Launch, Upper-Stage, and Space Vehicles
- **NASA-HDBK-2114**: Pointing and Tracking Requirements for Space Systems
- **ECSS-E-ST-60-10C**: Control Performance and Mission Dependability

Key compliance features:
- Aerospace-standard metrics (RMS, peak, saturation)
- Statistical confidence analysis (Monte Carlo with 100+ runs)
- Deterministic seeding for reproducibility
- Comprehensive documentation and traceability

---

## Future Enhancements

Potential improvements for future work:

1. **Parallel Execution**: Implement multiprocessing for faster batch runs
2. **Sensitivity Analysis**: Add Sobol indices for variance-based sensitivity
3. **Optimization Integration**: Couple with optimization framework for robust design
4. **Real-time Monitoring**: Add live progress dashboard during batch execution
5. **Advanced Distributions**: Support correlated parameters, mixture distributions
6. **Interactive Visualization**: Add Plotly/Bokeh for interactive result exploration

---

## Contact & Support

For questions or issues:
- Review test cases for usage examples
- Check demonstration script for complete workflow
- Refer to inline documentation (docstrings) for API details

---

**Implementation Date**: December 2024  
**Test Status**: ✓ All 36 tests passing  
**Code Coverage**: Comprehensive (metrics, distributions, batch execution, reporting)
