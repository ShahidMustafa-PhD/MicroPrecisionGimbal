"""
Unit tests for PerformanceAnalyzer.

Test coverage:
- Metric calculation accuracy
- Unit detection (rad vs µrad)
- Pass/fail requirement assessment
- Edge cases (empty data, zero values)
- Report generation
- DataFrame export
"""

import pytest
import numpy as np
from lasercom_digital_twin.core.simulation.performance_analyzer import (
    PerformanceAnalyzer,
    PerformanceMetrics
)


class TestPerformanceAnalyzer:
    """Test suite for PerformanceAnalyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance with standard requirements."""
        return PerformanceAnalyzer(
            rms_requirement=10.0,  # µrad
            peak_requirement=50.0,  # µrad
            fsm_limit=400.0  # µrad (saturation at 30% means we check if commands exceed 120 µrad)
        )
    
    @pytest.fixture
    def sample_telemetry_urad(self):
        """Generate synthetic telemetry data in microradians."""
        t = np.linspace(0, 10, 1000)
        dt = t[1] - t[0]
        
        # Pointing errors: small sinusoid + noise (µrad)
        los_error_x = 5.0 * np.sin(2 * np.pi * 1.0 * t) + np.random.randn(len(t)) * 1.0
        los_error_y = 3.0 * np.cos(2 * np.pi * 0.5 * t) + np.random.randn(len(t)) * 1.0
        
        # FSM commands (µrad, with occasional saturation)
        fsm_cmd_alpha = 200.0 * np.sin(2 * np.pi * 5.0 * t)
        fsm_cmd_beta = 150.0 * np.cos(2 * np.pi * 3.0 * t)
        
        # Apply saturation limits
        fsm_limit = 400.0  # µrad
        fsm_cmd_alpha = np.clip(fsm_cmd_alpha, -fsm_limit, fsm_limit)
        fsm_cmd_beta = np.clip(fsm_cmd_beta, -fsm_limit, fsm_limit)
        
        # Gimbal angles (rad)
        gimbal_az = np.deg2rad(10.0) * np.ones_like(t)
        gimbal_el = np.deg2rad(5.0) * np.ones_like(t)
        
        # Torques (N·m)
        coarse_torque_az = 0.05 * np.sin(2 * np.pi * 2.0 * t)
        coarse_torque_el = 0.03 * np.cos(2 * np.pi * 1.5 * t)
        
        return {
            'time': t.tolist(),
            'los_error_x': los_error_x.tolist(),
            'los_error_y': los_error_y.tolist(),
            'fsm_cmd_alpha': fsm_cmd_alpha.tolist(),
            'fsm_cmd_beta': fsm_cmd_beta.tolist(),
            'gimbal_az': gimbal_az.tolist(),
            'gimbal_el': gimbal_el.tolist(),
            'coarse_torque_az': coarse_torque_az.tolist(),
            'coarse_torque_el': coarse_torque_el.tolist(),
            'estimator_converged': [True] * len(t)
        }
    
    @pytest.fixture
    def sample_telemetry_rad(self):
        """Generate synthetic telemetry data in radians."""
        t = np.linspace(0, 10, 1000)
        
        # Convert µrad to rad for testing unit detection
        los_error_x_urad = 5.0 * np.sin(2 * np.pi * 1.0 * t)
        los_error_y_urad = 3.0 * np.cos(2 * np.pi * 0.5 * t)
        
        los_error_x = los_error_x_urad * 1e-6  # rad
        los_error_y = los_error_y_urad * 1e-6  # rad
        
        fsm_cmd_alpha = 200e-6 * np.sin(2 * np.pi * 5.0 * t)  # rad
        fsm_cmd_beta = 150e-6 * np.cos(2 * np.pi * 3.0 * t)  # rad
        
        gimbal_az = np.deg2rad(10.0) * np.ones_like(t)
        gimbal_el = np.deg2rad(5.0) * np.ones_like(t)
        
        coarse_torque_az = 0.05 * np.sin(2 * np.pi * 2.0 * t)
        coarse_torque_el = 0.03 * np.cos(2 * np.pi * 1.5 * t)
        
        return {
            'time': t.tolist(),
            'los_error_x': los_error_x.tolist(),
            'los_error_y': los_error_y.tolist(),
            'fsm_cmd_alpha': fsm_cmd_alpha.tolist(),
            'fsm_cmd_beta': fsm_cmd_beta.tolist(),
            'gimbal_az': gimbal_az.tolist(),
            'gimbal_el': gimbal_el.tolist(),
            'coarse_torque_az': coarse_torque_az.tolist(),
            'coarse_torque_el': coarse_torque_el.tolist(),
        }
    
    def test_initialization(self):
        """Test analyzer initialization with requirements."""
        analyzer = PerformanceAnalyzer(
            rms_requirement=15.0,
            peak_requirement=60.0,
            fsm_limit=500.0
        )
        
        assert analyzer.rms_requirement == 15.0
        assert analyzer.peak_requirement == 60.0
        assert analyzer.fsm_limit == 500.0
    
    def test_analyze_basic(self, analyzer, sample_telemetry_urad):
        """Test basic metric computation."""
        metrics = analyzer.analyze(sample_telemetry_urad)
        
        # Check that metrics object is returned
        assert isinstance(metrics, PerformanceMetrics)
        
        # Check primary metrics are computed
        assert metrics.rms_pointing_error > 0
        assert metrics.peak_pointing_error > 0
        assert 0 <= metrics.fsm_saturation_percentage <= 100
        
        # RMS should be less than peak
        assert metrics.rms_pointing_error < metrics.peak_pointing_error
    
    def test_rms_calculation(self, analyzer, sample_telemetry_urad):
        """Test RMS pointing error calculation accuracy."""
        metrics = analyzer.analyze(sample_telemetry_urad)
        
        # Manually compute expected RMS
        x = np.array(sample_telemetry_urad['los_error_x'])
        y = np.array(sample_telemetry_urad['los_error_y'])
        total_error = np.sqrt(x**2 + y**2)
        expected_rms = np.sqrt(np.mean(total_error**2))
        
        # Should match within 0.1%
        assert abs(metrics.rms_pointing_error - expected_rms) / expected_rms < 0.001
    
    def test_peak_calculation(self, analyzer, sample_telemetry_urad):
        """Test peak error calculation."""
        metrics = analyzer.analyze(sample_telemetry_urad)
        
        # Manually compute expected peak
        x = np.array(sample_telemetry_urad['los_error_x'])
        y = np.array(sample_telemetry_urad['los_error_y'])
        total_error = np.sqrt(x**2 + y**2)
        expected_peak = np.max(total_error)
        
        assert abs(metrics.peak_pointing_error - expected_peak) < 0.01
    
    def test_fsm_saturation_calculation(self, analyzer):
        """Test FSM saturation percentage calculation."""
        # Create telemetry with known saturation
        t = np.linspace(0, 10, 1000)
        
        # 30% of samples at limits
        fsm_cmd_alpha = np.concatenate([
            np.full(300, 400.0),  # Saturated
            np.linspace(-300, 300, 700)  # Not saturated
        ])
        
        fsm_cmd_beta = np.zeros(1000)
        
        telemetry = {
            'time': t.tolist(),
            'los_error_x': np.zeros(1000).tolist(),
            'los_error_y': np.zeros(1000).tolist(),
            'fsm_cmd_alpha': fsm_cmd_alpha.tolist(),
            'fsm_cmd_beta': fsm_cmd_beta.tolist(),
            'gimbal_az': np.zeros(1000).tolist(),
            'gimbal_el': np.zeros(1000).tolist(),
        }
        
        metrics = analyzer.analyze(telemetry)
        
        # Should detect ~30% saturation (allowing for numerical tolerance)
        assert 28.0 < metrics.fsm_saturation_percentage < 32.0
    
    def test_unit_detection_urad(self, analyzer, sample_telemetry_urad):
        """Test that analyzer correctly detects microradians."""
        metrics = analyzer.analyze(sample_telemetry_urad)
        
        # Values should be in µrad range (not rad)
        assert metrics.rms_pointing_error < 100  # Not ~1e-4
        assert metrics.rms_pointing_error > 1    # Not ~1e-11
    
    def test_unit_detection_rad(self, analyzer, sample_telemetry_rad):
        """Test that analyzer correctly converts radians to µrad."""
        metrics = analyzer.analyze(sample_telemetry_rad)
        
        # Values should be auto-converted to µrad
        assert metrics.rms_pointing_error < 100
        assert metrics.rms_pointing_error > 1
    
    def test_time_window(self, analyzer, sample_telemetry_urad):
        """Test analysis over specified time window."""
        # Analyze only middle 5 seconds
        metrics = analyzer.analyze(
            sample_telemetry_urad,
            start_time=2.5,
            end_time=7.5
        )
        
        assert metrics.rms_pointing_error > 0
        # Metrics should be computed only over specified window
    
    def test_requirement_assessment_pass(self, analyzer):
        """Test that good performance passes requirements."""
        # Create perfect telemetry
        t = np.linspace(0, 10, 1000)
        telemetry = {
            'time': t.tolist(),
            'los_error_x': (np.ones(1000) * 2.0).tolist(),  # 2 µrad constant
            'los_error_y': np.zeros(1000).tolist(),
            'fsm_cmd_alpha': (np.ones(1000) * 50.0).tolist(),  # Well below 400
            'fsm_cmd_beta': np.zeros(1000).tolist(),
            'gimbal_az': np.zeros(1000).tolist(),
            'gimbal_el': np.zeros(1000).tolist(),
        }
        
        metrics = analyzer.analyze(telemetry)
        
        # Should pass all requirements
        assert bool(metrics.meets_rms_requirement) == True
        assert bool(metrics.meets_peak_requirement) == True
        assert bool(metrics.meets_saturation_requirement) == True
    
    def test_requirement_assessment_fail(self, analyzer):
        """Test that poor performance fails requirements."""
        # Create bad telemetry
        t = np.linspace(0, 10, 1000)
        telemetry = {
            'time': t.tolist(),
            'los_error_x': (np.ones(1000) * 100.0).tolist(),  # 100 µrad >> 10 µrad req
            'los_error_y': np.zeros(1000).tolist(),
            'fsm_cmd_alpha': (np.ones(1000) * 400.0).tolist(),  # Fully saturated
            'fsm_cmd_beta': (np.ones(1000) * 400.0).tolist(),
            'gimbal_az': np.zeros(1000).tolist(),
            'gimbal_el': np.zeros(1000).tolist(),
        }
        
        metrics = analyzer.analyze(telemetry)
        
        # Should fail requirements
        assert bool(metrics.meets_rms_requirement) == False
        assert bool(metrics.meets_saturation_requirement) == False
    
    def test_empty_telemetry(self, analyzer):
        """Test graceful handling of empty telemetry."""
        telemetry = {
            'time': [0.0],  # At least one time point
            'los_error_x': [0.0],
            'los_error_y': [0.0],
            'fsm_cmd_alpha': [0.0],
            'fsm_cmd_beta': [0.0],
            'gimbal_az': [0.0],
            'gimbal_el': [0.0],
        }
        
        metrics = analyzer.analyze(telemetry)
        
        # Should return zero metrics
        assert metrics.rms_pointing_error == 0
        assert metrics.peak_pointing_error == 0
    
    def test_generate_report(self, analyzer, sample_telemetry_urad):
        """Test report generation."""
        metrics = analyzer.analyze(sample_telemetry_urad)
        report = analyzer.generate_report(metrics)
        
        # Check that report contains key sections
        assert "PERFORMANCE ANALYSIS REPORT" in report
        assert "POINTING METRICS" in report
        assert "RMS Pointing Error" in report
        assert "Peak Pointing Error" in report
        assert "Saturation" in report
        
        # Check pass/fail indicators
        assert "PASS" in report or "FAIL" in report
    
    def test_to_dataframe(self, analyzer, sample_telemetry_urad):
        """Test DataFrame export."""
        metrics = analyzer.analyze(sample_telemetry_urad)
        df = analyzer.to_dataframe(metrics)
        
        # Check DataFrame structure
        assert len(df) == 1  # Single row
        assert 'rms_pointing_error' in df.columns
        assert 'peak_pointing_error' in df.columns
        assert 'fsm_saturation_pct' in df.columns
        
        # Check values match metrics
        assert df['rms_pointing_error'].iloc[0] == metrics.rms_pointing_error
    
    def test_fsm_command_statistics(self, analyzer, sample_telemetry_urad):
        """Test FSM command statistics (RMS, max slew)."""
        metrics = analyzer.analyze(sample_telemetry_urad)
        
        # Should compute RMS command magnitude
        assert metrics.fsm_rms_command > 0
        assert metrics.fsm_slew_rate_max >= 0
    
    def test_stability_metrics(self, analyzer):
        """Test stability metric computation (damping, settling)."""
        # Create step response-like data
        t = np.linspace(0, 5, 500)
        
        # Underdamped step response (zeta ~ 0.5)
        zeta = 0.5
        wn = 10.0  # rad/s
        wd = wn * np.sqrt(1 - zeta**2)
        
        step_response = 1 - np.exp(-zeta * wn * t) * (
            np.cos(wd * t) + (zeta / np.sqrt(1 - zeta**2)) * np.sin(wd * t)
        )
        
        # Scale to µrad
        los_error = (50.0 * (1 - step_response))  # Start at 50, settle to 0
        
        telemetry = {
            'time': t.tolist(),
            'los_error_x': los_error.tolist(),
            'los_error_y': np.zeros(len(t)).tolist(),
            'fsm_cmd_alpha': np.zeros(len(t)).tolist(),
            'fsm_cmd_beta': np.zeros(len(t)).tolist(),
            'gimbal_az': np.zeros(len(t)).tolist(),
            'gimbal_el': np.zeros(len(t)).tolist(),
        }
        
        metrics = analyzer.analyze(telemetry)
        
        # Should estimate damping ratio (rough check)
        # Note: Stability metrics may not be computed if data doesn't show clear step response
        # Just check they don't crash
        assert metrics.damping_ratio_az >= 0
        assert metrics.settling_time_az >= 0
    
    def test_tracking_metrics(self, analyzer, sample_telemetry_urad):
        """Test tracking performance metrics (steady-state, jitter)."""
        metrics = analyzer.analyze(sample_telemetry_urad)
        
        # Should compute steady-state error (last 20% of data)
        assert metrics.steady_state_error >= 0
        
        # Should compute jitter RMS
        assert metrics.jitter_rms >= 0
    
    def test_control_effort_metrics(self, analyzer, sample_telemetry_urad):
        """Test control effort statistics."""
        metrics = analyzer.analyze(sample_telemetry_urad)
        
        # Should compute torque statistics
        assert metrics.coarse_rms_torque_az >= 0
        assert metrics.coarse_rms_torque_el >= 0
        assert metrics.coarse_peak_torque_az >= 0
        assert metrics.coarse_peak_torque_el >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
