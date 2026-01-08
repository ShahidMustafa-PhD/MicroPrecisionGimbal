"""
Unit tests for environmental disturbance models.

Tests verify:
1. Deterministic behavior with seeded RNG
2. Correct spectral characteristics (filtering)
3. Physical realism (magnitude bounds)
4. State management (reset, diagnostics)
"""

import pytest
import numpy as np
from lasercom_digital_twin.core.disturbances.disturbance_models import (
    EnvironmentalDisturbances,
    SimpleDisturbanceModel,
    DisturbanceState
)


class TestDisturbanceState:
    """Test DisturbanceState dataclass."""
    
    def test_initialization(self):
        """Test default initialization."""
        state = DisturbanceState()
        
        assert state.wind_torque_az == 0.0
        assert state.wind_torque_el == 0.0
        assert state.total_torque_az == 0.0
        assert state.total_torque_el == 0.0
        
    def test_custom_values(self):
        """Test initialization with custom values."""
        state = DisturbanceState(
            wind_torque_az=0.5,
            wind_torque_el=0.3,
            vibration_accel_z=0.01
        )
        
        assert state.wind_torque_az == 0.5
        assert state.wind_torque_el == 0.3
        assert state.vibration_accel_z == 0.01


class TestEnvironmentalDisturbances:
    """Test comprehensive environmental disturbance model."""
    
    def test_initialization(self):
        """Test initialization with default config."""
        config = {
            'wind_rms': 0.5,
            'vibration_psd': 1e-6,
            'structural_noise_std': 0.01,
            'seed': 42
        }
        
        disturbances = EnvironmentalDisturbances(config)
        
        assert disturbances.wind_rms == 0.5
        assert disturbances.vibration_psd == 1e-6
        assert disturbances.structural_noise_std == 0.01
        assert disturbances.seed == 42
        
    def test_deterministic_behavior(self):
        """Test that same seed produces identical sequences."""
        config = {
            'wind_rms': 0.5,
            'seed': 123
        }
        
        # Create two instances with same seed
        dist1 = EnvironmentalDisturbances(config)
        dist2 = EnvironmentalDisturbances(config)
        
        # Generate sequences
        states1 = [dist1.step(dt=0.001) for _ in range(100)]
        states2 = [dist2.step(dt=0.001) for _ in range(100)]
        
        # Should be identical
        for s1, s2 in zip(states1, states2):
            assert s1.wind_torque_az == s2.wind_torque_az
            assert s1.wind_torque_el == s2.wind_torque_el
            
    def test_different_seeds_produce_different_output(self):
        """Test that different seeds produce different sequences."""
        config1 = {'wind_rms': 0.5, 'seed': 42}
        config2 = {'wind_rms': 0.5, 'seed': 43}
        
        dist1 = EnvironmentalDisturbances(config1)
        dist2 = EnvironmentalDisturbances(config2)
        
        states1 = [dist1.step(dt=0.001) for _ in range(100)]
        states2 = [dist2.step(dt=0.001) for _ in range(100)]
        
        # Should be different
        differences = sum(
            1 for s1, s2 in zip(states1, states2)
            if s1.wind_torque_az != s2.wind_torque_az
        )
        assert differences > 50  # Most should be different
        
    def test_wind_torque_generation(self):
        """Test wind torque has correct statistical properties."""
        config = {
            'wind_rms': 1.0,
            'wind_correlation_time': 1.0,
            'wind_enabled': True,
            'seed': 42
        }
        
        disturbances = EnvironmentalDisturbances(config)
        
        # Generate long sequence to allow settling
        n_steps = 20000  # Longer sequence for better statistics
        dt = 0.001
        wind_az = []
        
        for _ in range(n_steps):
            state = disturbances.step(dt=dt)
            wind_az.append(state.wind_torque_az)
        
        wind_az = np.array(wind_az)
        
        # Check statistics (after discarding transient)
        wind_steady = wind_az[5000:]  # Discard first 5 seconds
        std = np.std(wind_steady)
        
        # Std should be close to configured RMS (within 40%)
        # Gauss-Markov process has variance equal to configured sigma²
        # Allow wider tolerance due to finite sample size
        assert 0.6 * config['wind_rms'] < std < 1.4 * config['wind_rms'], \
            f"Std {std} should be near {config['wind_rms']}"
        
        # Verify output is not constant
        assert np.max(wind_steady) != np.min(wind_steady)
        
    def test_wind_correlation_time(self):
        """Test wind has correct temporal correlation."""
        config = {
            'wind_rms': 1.0,
            'wind_correlation_time': 0.5,  # 0.5 second correlation
            'wind_enabled': True,
            'seed': 42
        }
        
        disturbances = EnvironmentalDisturbances(config)
        
        # Generate sequence
        n_steps = 5000
        dt = 0.001
        wind_az = []
        
        for _ in range(n_steps):
            state = disturbances.step(dt=dt)
            wind_az.append(state.wind_torque_az)
        
        wind_az = np.array(wind_az)
        
        # Compute autocorrelation
        autocorr = np.correlate(wind_az, wind_az, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]
        
        # At correlation time, should decay to ~1/e
        lag_samples = int(config['wind_correlation_time'] / dt)
        assert 0.2 < autocorr[lag_samples] < 0.5
        
    def test_ground_vibration_filtering(self):
        """Test vibration is properly band-limited."""
        config = {
            'vibration_psd': 1e-5,
            'vibration_freq_low': 10.0,
            'vibration_freq_high': 100.0,
            'vibration_enabled': True,
            'seed': 42
        }
        
        disturbances = EnvironmentalDisturbances(config)
        
        # Generate sequence
        n_steps = 10000
        dt = 0.001
        fs = 1.0 / dt
        vibration_z = []
        
        for _ in range(n_steps):
            state = disturbances.step(dt=dt)
            vibration_z.append(state.vibration_accel_z)
        
        vibration_z = np.array(vibration_z)
        
        # Compute FFT
        fft = np.fft.rfft(vibration_z)
        freqs = np.fft.rfftfreq(len(vibration_z), dt)
        psd = np.abs(fft) ** 2 / len(vibration_z)
        
        # Check that most energy is in passband
        in_band = (freqs >= config['vibration_freq_low']) & \
                  (freqs <= config['vibration_freq_high'])
        out_band = ~in_band
        
        energy_in_band = np.sum(psd[in_band])
        energy_out_band = np.sum(psd[out_band])
        
        # At least 70% of energy should be in passband
        assert energy_in_band > 0.7 * (energy_in_band + energy_out_band)
        
    def test_structural_noise_filtering(self):
        """Test structural noise is high-frequency."""
        config = {
            'structural_noise_std': 0.1,
            'structural_freq_low': 100.0,
            'structural_freq_high': 500.0,
            'structural_enabled': True,
            'seed': 42
        }
        
        disturbances = EnvironmentalDisturbances(config)
        
        # Generate sequence
        n_steps = 5000
        dt = 0.001
        structural_az = []
        
        for _ in range(n_steps):
            state = disturbances.step(dt=dt)
            structural_az.append(state.structural_torque_az)
        
        structural_az = np.array(structural_az)
        
        # Compute FFT
        fft = np.fft.rfft(structural_az)
        freqs = np.fft.rfftfreq(len(structural_az), dt)
        psd = np.abs(fft) ** 2 / len(structural_az)
        
        # Check that most energy is in high-frequency range
        in_band = (freqs >= config['structural_freq_low']) & \
                  (freqs <= config['structural_freq_high'])
        
        energy_in_band = np.sum(psd[in_band])
        energy_total = np.sum(psd)
        
        # At least 60% should be in passband
        assert energy_in_band > 0.6 * energy_total
        
    def test_enable_disable_flags(self):
        """Test individual disturbance enable/disable."""
        config = {
            'wind_rms': 1.0,
            'wind_enabled': False,
            'vibration_psd': 1e-6,
            'vibration_enabled': False,
            'structural_noise_std': 0.1,
            'structural_enabled': False,
            'seed': 42
        }
        
        disturbances = EnvironmentalDisturbances(config)
        
        # All outputs should be zero
        for _ in range(100):
            state = disturbances.step(dt=0.001)
            assert state.wind_torque_az == 0.0
            assert state.wind_torque_el == 0.0
            assert state.vibration_accel_z == 0.0
            assert state.structural_torque_az == 0.0
            assert state.total_torque_az == 0.0
            
    def test_total_torque_combination(self):
        """Test that total torque correctly combines components."""
        config = {
            'wind_rms': 0.5,
            'wind_enabled': True,
            'structural_noise_std': 0.1,
            'structural_enabled': True,
            'vibration_enabled': False,
            'seed': 42
        }
        
        disturbances = EnvironmentalDisturbances(config)
        
        state = disturbances.step(dt=0.001)
        
        # Total should equal sum of components
        expected_az = state.wind_torque_az + state.structural_torque_az
        assert abs(state.total_torque_az - expected_az) < 1e-10
        
    def test_angle_dependent_wind(self):
        """Test wind torque varies with gimbal angle."""
        config = {
            'wind_rms': 1.0,
            'wind_enabled': True,
            'seed': 42
        }
        
        disturbances = EnvironmentalDisturbances(config)
        
        # Wind at different elevations should scale differently
        state_low = disturbances.step(dt=0.001, gimbal_el=0.0)
        
        disturbances.reset()
        state_high = disturbances.step(dt=0.001, gimbal_el=np.pi/2)
        
        # Scaling should be different (but both non-zero)
        assert state_low.wind_torque_az != 0.0
        assert state_high.wind_torque_az != 0.0
        assert state_low.wind_torque_az != state_high.wind_torque_az
        
    def test_reset(self):
        """Test reset returns to initial state."""
        config = {
            'wind_rms': 1.0,
            'seed': 42
        }
        
        disturbances = EnvironmentalDisturbances(config)
        
        # Generate some states
        states1 = [disturbances.step(dt=0.001) for _ in range(50)]
        
        # Reset and generate again
        disturbances.reset()
        states2 = [disturbances.step(dt=0.001) for _ in range(50)]
        
        # Should be identical
        for s1, s2 in zip(states1, states2):
            assert s1.wind_torque_az == s2.wind_torque_az
            
    def test_diagnostics(self):
        """Test diagnostic information retrieval."""
        config = {
            'wind_rms': 0.5,
            'vibration_psd': 1e-6,
            'seed': 42
        }
        
        disturbances = EnvironmentalDisturbances(config)
        disturbances.step(dt=0.001)
        
        diag = disturbances.get_diagnostics()
        
        assert 'iteration' in diag
        assert 'wind_rms' in diag
        assert 'vibration_psd' in diag
        assert diag['wind_rms'] == 0.5
        assert diag['iteration'] == 1


class TestSimpleDisturbanceModel:
    """Test simplified disturbance model."""
    
    def test_initialization(self):
        """Test simple model initialization."""
        config = {'torque_std': 0.1, 'seed': 42}
        
        model = SimpleDisturbanceModel(config)
        
        assert model.torque_std == 0.1
        assert model.seed == 42
        
    def test_white_noise_generation(self):
        """Test generates white noise with correct statistics."""
        config = {'torque_std': 1.0, 'seed': 42}
        
        model = SimpleDisturbanceModel(config)
        
        # Generate sequence
        torques = [model.step(dt=0.001).total_torque_az for _ in range(5000)]
        torques = np.array(torques)
        
        # Check statistics
        mean = np.mean(torques)
        std = np.std(torques)
        
        assert abs(mean) < 0.1
        assert 0.9 < std < 1.1
        
    def test_deterministic(self):
        """Test simple model is deterministic."""
        config = {'torque_std': 0.5, 'seed': 123}
        
        model1 = SimpleDisturbanceModel(config)
        model2 = SimpleDisturbanceModel(config)
        
        states1 = [model1.step(dt=0.001) for _ in range(100)]
        states2 = [model2.step(dt=0.001) for _ in range(100)]
        
        for s1, s2 in zip(states1, states2):
            assert s1.total_torque_az == s2.total_torque_az
            
    def test_reset(self):
        """Test reset functionality."""
        config = {'torque_std': 0.5, 'seed': 42}
        
        model = SimpleDisturbanceModel(config)
        
        states1 = [model.step(dt=0.001) for _ in range(50)]
        
        model.reset()
        states2 = [model.step(dt=0.001) for _ in range(50)]
        
        for s1, s2 in zip(states1, states2):
            assert s1.total_torque_az == s2.total_torque_az


class TestDisturbanceIntegration:
    """Test integration scenarios."""
    
    def test_typical_mission_profile(self):
        """Test disturbances over typical mission duration."""
        config = {
            'wind_rms': 0.3,
            'vibration_psd': 1e-6,
            'structural_noise_std': 0.01,
            'seed': 42
        }
        
        disturbances = EnvironmentalDisturbances(config)
        
        # Simulate 10 seconds at 1ms timestep
        n_steps = 10000
        dt = 0.001
        
        max_torque = 0.0
        
        for _ in range(n_steps):
            state = disturbances.step(dt=dt)
            max_torque = max(
                max_torque,
                abs(state.total_torque_az),
                abs(state.total_torque_el)
            )
        
        # Should generate reasonable magnitudes (not NaN or infinite)
        assert np.isfinite(max_torque)
        assert max_torque > 0.0
        assert max_torque < 10.0  # Reasonable upper bound
        
    def test_high_sample_rate(self):
        """Test disturbances at high sample rate (0.1ms)."""
        config = {
            'wind_rms': 0.5,
            'vibration_psd': 1e-6,
            'seed': 42
        }
        
        disturbances = EnvironmentalDisturbances(config)
        
        # Run at 10 kHz
        dt = 0.0001
        states = [disturbances.step(dt=dt) for _ in range(1000)]
        
        # Should complete without errors
        assert len(states) == 1000
        assert all(np.isfinite(s.total_torque_az) for s in states)
        
    def test_variable_timestep(self):
        """Test with variable timesteps."""
        config = {
            'wind_rms': 0.5,
            'seed': 42
        }
        
        disturbances = EnvironmentalDisturbances(config)
        
        # Vary dt
        timesteps = [0.001, 0.002, 0.0005, 0.003, 0.001]
        
        for dt in timesteps:
            state = disturbances.step(dt=dt)
            assert np.isfinite(state.total_torque_az)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
