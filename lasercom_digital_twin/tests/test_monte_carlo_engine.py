"""
Unit tests for MonteCarloEngine.

Test coverage:
- Parameter randomization with various distributions
- Deterministic seeding (reproducibility)
- Batch execution orchestration
- Statistical aggregation
- Configuration parsing
- Edge cases (failed runs, no uncertainties)
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from lasercom_digital_twin.core.simulation.monte_carlo_engine import (
    MonteCarloEngine,
    MonteCarloConfig,
    ParameterUncertainty,
    ParameterRandomizer,
    DistributionType,
    create_default_uncertainties
)
from lasercom_digital_twin.core.simulation.performance_analyzer import (
    PerformanceAnalyzer,
    PerformanceMetrics
)


class TestParameterRandomizer:
    """Test suite for ParameterRandomizer."""
    
    def test_deterministic_sampling(self):
        """Test that same seed produces same samples."""
        param = ParameterUncertainty(
            name='test_param',
            nominal=10.0,
            distribution=DistributionType.NORMAL,
            uncertainty=5.0
        )
        
        # Two randomizers with same seed
        rng1 = ParameterRandomizer(seed=42)
        rng2 = ParameterRandomizer(seed=42)
        
        samples1 = [rng1.sample(param) for _ in range(10)]
        samples2 = [rng2.sample(param) for _ in range(10)]
        
        # Should produce identical sequences
        np.testing.assert_array_almost_equal(samples1, samples2)
    
    def test_different_seeds(self):
        """Test that different seeds produce different samples."""
        param = ParameterUncertainty(
            name='test_param',
            nominal=10.0,
            distribution=DistributionType.NORMAL,
            uncertainty=5.0
        )
        
        rng1 = ParameterRandomizer(seed=42)
        rng2 = ParameterRandomizer(seed=123)
        
        samples1 = [rng1.sample(param) for _ in range(10)]
        samples2 = [rng2.sample(param) for _ in range(10)]
        
        # Should be different
        assert not np.allclose(samples1, samples2)
    
    def test_uniform_distribution(self):
        """Test uniform distribution sampling."""
        param = ParameterUncertainty(
            name='test_param',
            nominal=100.0,
            distribution=DistributionType.UNIFORM,
            uncertainty=10.0  # ±10%
        )
        
        rng = ParameterRandomizer(seed=42)
        samples = [rng.sample(param) for _ in range(1000)]
        
        # Check bounds: should be in [90, 110]
        assert all(90.0 <= s <= 110.0 for s in samples)
        
        # Check mean (should be near nominal)
        assert abs(np.mean(samples) - 100.0) < 2.0
    
    def test_normal_distribution(self):
        """Test normal distribution sampling."""
        param = ParameterUncertainty(
            name='test_param',
            nominal=50.0,
            distribution=DistributionType.NORMAL,
            uncertainty=10.0  # ±10% = σ = 5.0
        )
        
        rng = ParameterRandomizer(seed=42)
        samples = [rng.sample(param) for _ in range(10000)]
        
        # Check mean
        assert abs(np.mean(samples) - 50.0) < 0.5
        
        # Check std (should be ~5.0)
        assert abs(np.std(samples) - 5.0) < 0.5
    
    def test_truncated_normal(self):
        """Test truncated normal distribution."""
        param = ParameterUncertainty(
            name='test_param',
            nominal=10.0,
            distribution=DistributionType.TRUNCATED_NORMAL,
            uncertainty=20.0,
            bounds=(5.0, 15.0)
        )
        
        rng = ParameterRandomizer(seed=42)
        samples = [rng.sample(param) for _ in range(1000)]
        
        # All samples should be within bounds
        assert all(5.0 <= s <= 15.0 for s in samples)
    
    def test_lognormal_distribution(self):
        """Test lognormal distribution (for strictly positive params)."""
        param = ParameterUncertainty(
            name='test_param',
            nominal=1.0,
            distribution=DistributionType.LOGNORMAL,
            uncertainty=20.0  # Relative std
        )
        
        rng = ParameterRandomizer(seed=42)
        samples = [rng.sample(param) for _ in range(1000)]
        
        # All samples should be positive
        assert all(s > 0 for s in samples)
        
        # Median should be near nominal
        assert abs(np.median(samples) - 1.0) < 0.2
    
    def test_bounds_enforcement(self):
        """Test that bounds are enforced for all distributions."""
        param = ParameterUncertainty(
            name='test_param',
            nominal=10.0,
            distribution=DistributionType.NORMAL,
            uncertainty=50.0,  # Large variation
            bounds=(5.0, 15.0)
        )
        
        rng = ParameterRandomizer(seed=42)
        samples = [rng.sample(param) for _ in range(1000)]
        
        # All samples must respect bounds
        assert all(5.0 <= s <= 15.0 for s in samples)
    
    def test_sample_batch(self):
        """Test batch sampling of multiple parameters."""
        uncertainties = [
            ParameterUncertainty('param1', 10.0, DistributionType.NORMAL, 5.0),
            ParameterUncertainty('param2', 20.0, DistributionType.UNIFORM, 10.0),
            ParameterUncertainty('param3', 1.0, DistributionType.LOGNORMAL, 15.0),
        ]
        
        rng = ParameterRandomizer(seed=42)
        params = rng.sample_batch(uncertainties)
        
        # Should return dictionary with all parameters
        assert len(params) == 3
        assert 'param1' in params
        assert 'param2' in params
        assert 'param3' in params
        
        # All values should be reasonable
        assert 5.0 < params['param1'] < 15.0
        assert 15.0 < params['param2'] < 25.0
        assert 0.5 < params['param3'] < 2.0


class TestMonteCarloEngine:
    """Test suite for MonteCarloEngine."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary directory for test outputs."""
        tmpdir = Path(tempfile.mkdtemp())
        yield tmpdir
        shutil.rmtree(tmpdir)
    
    @pytest.fixture
    def simple_config(self):
        """Create simple MC configuration."""
        uncertainties = [
            ParameterUncertainty('param1', 10.0, DistributionType.NORMAL, 5.0),
            ParameterUncertainty('param2', 20.0, DistributionType.UNIFORM, 10.0),
        ]
        
        return MonteCarloConfig(
            n_runs=10,
            base_seed=42,
            parameter_uncertainties=uncertainties,
            simulation_duration=1.0,
            save_telemetry=False
        )
    
    @pytest.fixture
    def analyzer(self):
        """Create performance analyzer."""
        return PerformanceAnalyzer(
            rms_requirement=10.0,
            peak_requirement=50.0,
            fsm_limit=400.0
        )
    
    def mock_simulation_factory(self, config, seed):
        """
        Mock simulation factory for testing.
        
        Returns dummy telemetry that varies slightly with seed.
        """
        np.random.seed(seed)
        
        t = np.linspace(0, config['simulation_duration'], 100)
        
        # Generate dummy telemetry (varies with seed)
        base_error = 5.0 + np.random.randn() * 0.5  # 5 ± 0.5 µrad
        
        telemetry = {
            'time': t.tolist(),
            'los_error_x': (base_error * np.sin(2 * np.pi * 1.0 * t)).tolist(),
            'los_error_y': (base_error * 0.5 * np.cos(2 * np.pi * 0.5 * t)).tolist(),
            'fsm_cmd_alpha': (100.0 * np.sin(2 * np.pi * 5.0 * t)).tolist(),
            'fsm_cmd_beta': (50.0 * np.cos(2 * np.pi * 3.0 * t)).tolist(),
            'gimbal_az': np.zeros(len(t)).tolist(),
            'gimbal_el': np.zeros(len(t)).tolist(),
            'coarse_torque_az': (0.01 * np.sin(2 * np.pi * 2.0 * t)).tolist(),
            'coarse_torque_el': (0.005 * np.cos(2 * np.pi * 1.0 * t)).tolist(),
        }
        
        # Return (runner_mock, telemetry)
        return None, telemetry
    
    def test_engine_initialization(self, simple_config, analyzer):
        """Test engine initialization."""
        engine = MonteCarloEngine(simple_config, analyzer, verbose=False)
        
        assert engine.config == simple_config
        assert engine.analyzer == analyzer
    
    def test_run_batch_basic(self, simple_config, analyzer):
        """Test basic batch execution."""
        engine = MonteCarloEngine(simple_config, analyzer, verbose=False)
        
        results = engine.run_batch(self.mock_simulation_factory)
        
        # Check results structure
        assert len(results.runs) == 10
        assert results.n_successful == 10
        assert results.n_failed == 0
        assert results.total_execution_time > 0
    
    def test_deterministic_batch(self, simple_config, analyzer):
        """Test that same configuration produces same results."""
        engine1 = MonteCarloEngine(simple_config, analyzer, verbose=False)
        engine2 = MonteCarloEngine(simple_config, analyzer, verbose=False)
        
        results1 = engine1.run_batch(self.mock_simulation_factory)
        results2 = engine2.run_batch(self.mock_simulation_factory)
        
        # Check that metrics match
        for r1, r2 in zip(results1.runs, results2.runs):
            assert r1.seed == r2.seed
            # Allow small floating-point differences
            assert abs(r1.metrics.rms_pointing_error - r2.metrics.rms_pointing_error) < 0.01
    
    def test_parameter_randomization(self, simple_config, analyzer):
        """Test that parameters are randomized across runs."""
        engine = MonteCarloEngine(simple_config, analyzer, verbose=False)
        
        results = engine.run_batch(self.mock_simulation_factory)
        
        # Extract parameter values across runs
        param1_values = [r.parameters['param1'] for r in results.runs]
        param2_values = [r.parameters['param2'] for r in results.runs]
        
        # Parameters should vary across runs
        assert len(set(param1_values)) > 1
        assert len(set(param2_values)) > 1
        
        # Check that values are in reasonable range
        assert all(5.0 < p < 15.0 for p in param1_values)
        assert all(15.0 < p < 25.0 for p in param2_values)
    
    def test_summary_statistics(self, simple_config, analyzer):
        """Test that summary statistics are computed correctly."""
        engine = MonteCarloEngine(simple_config, analyzer, verbose=False)
        
        results = engine.run_batch(self.mock_simulation_factory)
        
        summary = results.summary_statistics
        
        # Check that summary contains expected columns
        assert 'rms_pointing_error_mean' in summary.columns
        assert 'rms_pointing_error_std' in summary.columns
        assert 'rms_pointing_error_min' in summary.columns
        assert 'rms_pointing_error_max' in summary.columns
        assert 'rms_pointing_error_median' in summary.columns
        assert 'rms_pointing_error_p95' in summary.columns
        
        # Check statistical consistency
        mean = summary['rms_pointing_error_mean'].iloc[0]
        min_val = summary['rms_pointing_error_min'].iloc[0]
        max_val = summary['rms_pointing_error_max'].iloc[0]
        
        assert min_val <= mean <= max_val
    
    def test_report_generation(self, simple_config, analyzer):
        """Test Monte Carlo report generation."""
        engine = MonteCarloEngine(simple_config, analyzer, verbose=False)
        
        results = engine.run_batch(self.mock_simulation_factory)
        report = engine.generate_report(results)
        
        # Check that report contains key sections
        assert "MONTE CARLO ANALYSIS REPORT" in report
        assert "EXECUTION SUMMARY" in report
        assert "PARAMETER UNCERTAINTIES" in report
        assert "PERFORMANCE STATISTICS" in report
        assert "RMS Pointing Error" in report
        
        # Check that parameter names appear
        assert "param1" in report
        assert "param2" in report
    
    def test_failed_run_handling(self, simple_config, analyzer):
        """Test handling of failed simulation runs."""
        def failing_factory(config, seed):
            # Fail every 3rd run
            if seed % 3 == 0:
                raise RuntimeError(f"Simulated failure for seed {seed}")
            return self.mock_simulation_factory(config, seed)
        
        engine = MonteCarloEngine(simple_config, analyzer, verbose=False)
        
        # Should not crash on failures
        results = engine.run_batch(failing_factory)
        
        # Check that some runs failed
        assert results.n_failed > 0
        assert results.n_successful < simple_config.n_runs
        
        # Summary should only include successful runs
        assert results.n_successful == len([r for r in results.runs if r.success])
    
    def test_save_outputs(self, simple_config, analyzer, temp_output_dir):
        """Test saving outputs to disk."""
        simple_config.output_dir = temp_output_dir
        simple_config.save_telemetry = True
        
        engine = MonteCarloEngine(simple_config, analyzer, verbose=False)
        results = engine.run_batch(self.mock_simulation_factory)
        
        # Check that files were created
        assert (temp_output_dir / "config.json").exists()
        assert (temp_output_dir / "summary_statistics.csv").exists()
        
        # Check individual run directories
        run_dirs = list(temp_output_dir.glob("run_*"))
        assert len(run_dirs) == simple_config.n_runs
        
        # Check run contents
        run_0 = temp_output_dir / "run_0000"
        assert (run_0 / "parameters.json").exists()
        assert (run_0 / "metrics.csv").exists()
        assert (run_0 / "telemetry.csv").exists()  # Because save_telemetry=True
    
    def test_apply_parameters_to_config(self, simple_config, analyzer):
        """Test parameter injection into configuration dictionary."""
        engine = MonteCarloEngine(simple_config, analyzer, verbose=False)
        
        base_config = {
            'motor': {
                'K_t': 0.1,
                'R': 1.0
            },
            'sensor': {
                'noise': 1e-5
            }
        }
        
        parameters = {
            'motor.K_t': 0.15,
            'sensor.noise': 2e-5
        }
        
        modified_config = engine._apply_parameters_to_config(base_config, parameters)
        
        # Check that parameters were applied
        assert modified_config['motor']['K_t'] == 0.15
        assert modified_config['sensor']['noise'] == 2e-5
        
        # Original should be unchanged
        assert base_config['motor']['K_t'] == 0.1
    
    def test_empty_uncertainties(self, analyzer):
        """Test batch with no parameter uncertainties (nominal only)."""
        config = MonteCarloConfig(
            n_runs=5,
            base_seed=42,
            parameter_uncertainties=[],  # No uncertainties
            simulation_duration=1.0
        )
        
        engine = MonteCarloEngine(config, analyzer, verbose=False)
        results = engine.run_batch(self.mock_simulation_factory)
        
        # Should still run successfully
        assert results.n_successful == 5
        
        # All runs should have empty parameter dictionaries
        assert all(len(r.parameters) == 0 for r in results.runs)
    
    def test_create_default_uncertainties(self):
        """Test default uncertainty creation utility."""
        uncertainties = create_default_uncertainties()
        
        # Should return a non-empty list
        assert len(uncertainties) > 0
        
        # Check that it covers motor, sensor, structural parameters
        param_names = [u.name for u in uncertainties]
        
        assert any('motor' in name for name in param_names)
        assert any('gyro' in name or 'sensor' in name for name in param_names)
        assert any('backlash' in name or 'friction' in name for name in param_names)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
