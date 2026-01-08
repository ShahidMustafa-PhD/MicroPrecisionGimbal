"""
Unit tests for optical chain and coordinate transformations.

This module contains pytest-based unit tests for the telescope optics,
coordinate frame transformations, and integrated optical chain functionality.
Tests verify small-angle approximations, FSM corrections, field rotation
compensation, and micron-level accuracy calculations.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.optics.optical_chain import TelescopeOptics, FocalPlanePosition
from core.coordinate_frames.transformations import OpticalFrameCompensator
from core.sensors.quadrant_detector import QuadrantDetector


class TestTelescopeOptics:
    """Test suite for TelescopeOptics."""
    
    @pytest.fixture
    def telescope_config(self):
        """Standard telescope configuration."""
        return {
            'focal_length_m': 1.5,
            'aperture_diameter_m': 0.3,
            'fsm_gain': 2.0,
            'wavelength_um': 1.55,
            'beam_diameter_mm': 50.0,
            'detector_size_mm': 5.0
        }
    
    def test_initialization(self, telescope_config):
        """Test telescope initializes and computes derived parameters."""
        telescope = TelescopeOptics(telescope_config)
        
        assert telescope.focal_length_m == 1.5
        assert telescope.fsm_gain == 2.0
        assert telescope.f_number > 0
        assert telescope.airy_disk_radius_um > 0
    
    def test_small_angle_approximation(self, telescope_config):
        """
        Test that angle-to-focal-plane mapping uses small-angle approximation.
        
        For small angles: x = f * tan(θ) ≈ f * θ
        """
        telescope = TelescopeOptics(telescope_config)
        
        # Small angle (10 µrad)
        angle_rad = 10.0e-6
        
        # Compute focal plane position
        pos = telescope.angle_to_focal_plane_position(angle_rad, 0.0)
        
        # Expected position: x = f * θ
        expected_x_m = telescope.focal_length_m * angle_rad
        expected_x_um = expected_x_m * 1e6
        
        # Should match to high precision for small angles
        assert np.abs(pos.x_um - expected_x_um) < 0.001, \
            f"Small-angle approximation error: {pos.x_um} vs {expected_x_um}"
    
    def test_fsm_gain_factor(self, telescope_config):
        """
        Test that FSM applies correct gain factor (2x for mirror).
        """
        telescope = TelescopeOptics(telescope_config)
        
        # Initial LOS error
        los_error = 100.0e-6  # 100 µrad
        
        # FSM command to fully correct error
        fsm_command = los_error / telescope.fsm_gain
        
        # Apply correction
        corrected_x, corrected_y = telescope.apply_fsm_correction(
            los_error, 0.0, fsm_command, 0.0
        )
        
        # Should be near zero (fully corrected)
        assert np.abs(corrected_x) < 1e-9, \
            f"FSM correction incomplete: residual = {corrected_x}"
    
    def test_pointing_accuracy_calculation(self, telescope_config):
        """
        Test computation of angular accuracy from spot centering requirement.
        """
        telescope = TelescopeOptics(telescope_config)
        
        # Require 1 µm spot centering
        required_spot_um = 1.0
        
        # Compute angular requirement
        required_angle_urad = telescope.get_pointing_accuracy_requirement(required_spot_um)
        
        # Verify conversion: θ = x / f
        expected_angle_rad = (required_spot_um * 1e-6) / telescope.focal_length_m
        expected_angle_urad = expected_angle_rad * 1e6
        
        assert np.abs(required_angle_urad - expected_angle_urad) < 0.01, \
            f"Accuracy calculation error: {required_angle_urad} vs {expected_angle_urad}"
    
    def test_detector_boundary_check(self, telescope_config):
        """
        Test that spot detection within detector bounds works correctly.
        """
        telescope = TelescopeOptics(telescope_config)
        
        # Position at detector center
        pos_center = FocalPlanePosition(x_um=0.0, y_um=0.0, x_m=0.0, y_m=0.0)
        assert telescope.is_within_detector(pos_center), "Center should be within detector"
        
        # Position beyond detector edge
        pos_outside = FocalPlanePosition(
            x_um=10000.0,  # 10 mm, beyond 5 mm detector
            y_um=0.0,
            x_m=0.01,
            y_m=0.0
        )
        assert not telescope.is_within_detector(pos_outside), "Should be outside detector"
    
    def test_focal_plane_to_angle_inversion(self, telescope_config):
        """
        Test that focal plane position converts back to angle correctly.
        """
        telescope = TelescopeOptics(telescope_config)
        
        # Original angle
        angle_x_orig = 50.0e-6  # 50 µrad
        angle_y_orig = 30.0e-6  # 30 µrad
        
        # Convert to focal plane
        pos = telescope.angle_to_focal_plane_position(angle_x_orig, angle_y_orig)
        
        # Convert back to angle
        angle_x_recovered, angle_y_recovered = telescope.focal_plane_position_to_angle(
            pos.x_m, pos.y_m
        )
        
        # Should recover original angles
        assert np.abs(angle_x_recovered - angle_x_orig) < 1e-12, \
            "X angle inversion error"
        assert np.abs(angle_y_recovered - angle_y_orig) < 1e-12, \
            "Y angle inversion error"
    
    def test_sensitivity_metrics(self, telescope_config):
        """
        Test that sensitivity analysis returns reasonable values.
        """
        telescope = TelescopeOptics(telescope_config)
        
        metrics = telescope.get_sensitivity_analysis()
        
        # Check key metrics exist and are reasonable
        assert metrics['focal_length_m'] == 1.5
        assert metrics['f_number'] > 0
        assert metrics['airy_disk_radius_um'] > 0
        assert metrics['fsm_gain'] == 2.0
        
        # Check that 1 µrad converts to correct focal plane distance
        um_per_urad = metrics['1_urad_to_focal_plane_um']
        expected = telescope.focal_length_m * 1e-6 * 1e6  # f * 1µrad * (1e6 µm/m)
        
        assert np.abs(um_per_urad - expected) < 0.01, \
            f"Sensitivity metric error: {um_per_urad} vs {expected}"


class TestOpticalFrameCompensator:
    """Test suite for OpticalFrameCompensator."""
    
    @pytest.fixture
    def compensator_config(self):
        """Standard compensator configuration."""
        return {
            'site_latitude_deg': 35.0,
            'use_field_rotation': True,
            'optical_misalignment_az_rad': 1.0e-5,
            'optical_misalignment_el_rad': 1.0e-5
        }
    
    def test_initialization(self, compensator_config):
        """Test compensator initializes correctly."""
        comp = OpticalFrameCompensator(compensator_config)
        
        assert comp.use_field_rotation == True
        assert comp.site_latitude_rad > 0
        assert comp.epsilon_az == 1.0e-5
    
    def test_rotation_matrices(self):
        """Test rotation matrix generation."""
        # Test Z-axis rotation (azimuth)
        angle = np.pi / 4  # 45 degrees
        R_z = OpticalFrameCompensator.rotation_matrix_z(angle)
        
        # Should be orthonormal
        assert np.allclose(R_z @ R_z.T, np.eye(3)), "R_z not orthonormal"
        
        # Test Y-axis rotation (elevation)
        R_y = OpticalFrameCompensator.rotation_matrix_y(angle)
        assert np.allclose(R_y @ R_y.T, np.eye(3)), "R_y not orthonormal"
    
    def test_los_vector_computation(self, compensator_config):
        """
        Test that LOS vector is computed correctly from Az/El angles.
        """
        comp = OpticalFrameCompensator(compensator_config)
        
        # Pointing at zenith (Az=0, El=90°)
        los_zenith = comp.mechanical_to_los_vector(0.0, np.pi/2)
        
        # Should point straight up [0, 0, 1]
        expected = np.array([0.0, 0.0, 1.0])
        assert np.allclose(los_zenith, expected, atol=1e-10), \
            f"Zenith LOS error: {los_zenith} vs {expected}"
    
    def test_field_rotation_at_zenith(self, compensator_config):
        """
        Test that field rotation is zero at zenith.
        """
        comp = OpticalFrameCompensator(compensator_config)
        
        # At zenith (El = 90°), field rotation should be zero
        gamma = comp.compute_field_rotation_angle(0.0, np.pi/2)
        
        assert np.abs(gamma) < 1e-6, \
            f"Field rotation should be ~0 at zenith: {gamma}"
    
    def test_field_rotation_varies_with_azimuth(self, compensator_config):
        """
        Test that field rotation varies with azimuth at fixed elevation.
        """
        comp = OpticalFrameCompensator(compensator_config)
        
        # At mid-elevation (45°), field rotation should vary with azimuth
        el = np.pi / 4
        
        gamma_0 = comp.compute_field_rotation_angle(0.0, el)
        gamma_90 = comp.compute_field_rotation_angle(np.pi/2, el)
        
        # Should be different
        assert np.abs(gamma_90 - gamma_0) > 0.01, \
            "Field rotation should vary with azimuth"
    
    def test_los_error_small_angle(self, compensator_config):
        """
        Test LOS error computation for small pointing errors.
        """
        comp = OpticalFrameCompensator(compensator_config)
        
        # Actual pointing
        actual_az = 0.0
        actual_el = np.pi / 4
        
        # Small error (10 µrad in azimuth)
        target_az = actual_az + 10.0e-6
        target_el = actual_el
        
        # Compute error
        tip_error, tilt_error = comp.compute_los_error(
            actual_az, actual_el, target_az, target_el
        )
        
        # Tip error should be ~10 µrad scaled by cos(el)
        expected_tip = -10.0e-6 * np.cos(actual_el)  # Negative because actual < target
        
        # Should be close (within misalignment)
        assert np.abs(tip_error - expected_tip) < 20.0e-6, \
            f"LOS error computation: {tip_error} vs {expected_tip}"
    
    def test_full_transform_consistency(self, compensator_config):
        """
        Test that full transform produces consistent results.
        """
        comp = OpticalFrameCompensator(compensator_config)
        
        # Zero error case
        actual_az = np.pi / 6
        actual_el = np.pi / 4
        
        tip, tilt, metadata = comp.full_transform(
            actual_az, actual_el,
            actual_az, actual_el
        )
        
        # With zero pointing error, should have small errors (only misalignment)
        assert np.abs(tip) < 50.0e-6, \
            f"Zero-error tip should be small: {tip}"
        assert np.abs(tilt) < 50.0e-6, \
            f"Zero-error tilt should be small: {tilt}"
        
        # Metadata should contain field rotation
        assert 'field_rotation_rad' in metadata
        assert 'field_rotation_deg' in metadata


class TestIntegratedOpticalChain:
    """Test integrated optical chain from angles to QPD."""
    
    @pytest.fixture
    def integrated_config(self):
        """Configuration for integrated test."""
        return {
            'telescope': {
                'focal_length_m': 1.5,
                'aperture_diameter_m': 0.3,
                'fsm_gain': 2.0,
                'wavelength_um': 1.55,
                'beam_diameter_mm': 50.0,
                'detector_size_mm': 5.0
            },
            'compensator': {
                'site_latitude_deg': 35.0,
                'use_field_rotation': True,
                'optical_misalignment_az_rad': 0.0,
                'optical_misalignment_el_rad': 0.0
            },
            'qpd': {
                'sensitivity': 2000.0,
                'linear_range': 100.0e-6,
                'noise_voltage_rms': 1.0e-4,
                'bias_x': 0.0,
                'bias_y': 0.0,
                'saturation_voltage': 10.0,
                'nonlinearity_factor': 0.15,
                'detector_size_um': 5000.0,
                'spot_size_um': 50.0
            }
        }
    
    def test_end_to_end_chain(self, integrated_config):
        """
        Test complete chain: angle error → focal plane → QPD voltage.
        """
        telescope = TelescopeOptics(integrated_config['telescope'])
        qpd = QuadrantDetector(integrated_config['qpd'], seed=42)
        
        # Small pointing error
        los_error_rad = 50.0e-6  # 50 µrad
        
        # Convert to focal plane position
        spot_pos = telescope.angle_to_focal_plane_position(los_error_rad, 0.0)
        
        # Measure with QPD
        v_x, v_y = qpd.measure_from_focal_plane(spot_pos.x_um, spot_pos.y_um)
        
        # Voltage should be non-zero and proportional to error
        assert np.abs(v_x) > 0.001, "QPD should produce measurable voltage"
        
        # Check approximate linearity for small error
        expected_voltage_order = qpd.sensitivity * los_error_rad
        assert np.abs(v_x) < expected_voltage_order * 10, \
            "Voltage magnitude unreasonable"
    
    def test_micron_accuracy_metrics(self, integrated_config):
        """
        Test that micron-level accuracy calculations are consistent.
        """
        telescope = TelescopeOptics(integrated_config['telescope'])
        qpd = QuadrantDetector(integrated_config['qpd'], seed=42)
        
        # Get QPD accuracy metrics
        qpd_metrics = qpd.get_micron_accuracy_metrics()
        
        # Noise-limited position accuracy should be calculable
        noise_limit_um = qpd_metrics['noise_limited_accuracy_um']
        
        # Should be a reasonable value (< 10 µm for good system)
        assert noise_limit_um > 0.01, "Noise limit too small (unrealistic)"
        assert noise_limit_um < 10.0, "Noise limit too large (poor system)"
        
        # Convert to angular accuracy using telescope
        noise_limit_m = noise_limit_um * 1e-6
        angular_accuracy_rad = noise_limit_m / telescope.focal_length_m
        angular_accuracy_urad = angular_accuracy_rad * 1e6
        
        # Should be sub-microradian for good system
        assert angular_accuracy_urad < 10.0, \
            f"Angular accuracy poor: {angular_accuracy_urad} µrad"
    
    def test_fsm_correction_in_chain(self, integrated_config):
        """
        Test that FSM correction properly reduces spot error.
        """
        telescope = TelescopeOptics(integrated_config['telescope'])
        
        # Large initial error
        los_error = 100.0e-6  # 100 µrad
        
        # Compute spot without FSM
        spot_no_fsm = telescope.compute_spot_position(los_error, 0.0, 0.0, 0.0)
        
        # Compute spot with partial FSM correction
        fsm_command = los_error / telescope.fsm_gain * 0.8  # 80% correction
        spot_with_fsm = telescope.compute_spot_position(
            los_error, 0.0, fsm_command, 0.0
        )
        
        # Residual spot position should be smaller
        residual_ratio = np.abs(spot_with_fsm.x_um) / np.abs(spot_no_fsm.x_um)
        
        assert residual_ratio < 0.3, \
            f"FSM correction insufficient: {residual_ratio}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
