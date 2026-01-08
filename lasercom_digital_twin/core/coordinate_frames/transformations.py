"""
Coordinate Frame Transformations for Laser Communication Terminal

This module implements the mathematical transformations between the various
coordinate frames used in the pointing system, with special emphasis on the
field rotation compensation required for non-mechanical roll stabilization.

Key Frames:
- Site Frame (S): Local vertical, azimuth reference
- Mechanical Frame (M): Gimbal axes (Az/El)
- Optical Frame (O): Line-of-sight aligned, field-rotation compensated
- Sensor Frame (Q): Detector/QPD aligned
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class RotationMatrix:
    """Container for rotation matrix with metadata."""
    matrix: np.ndarray
    angle_rad: float
    axis: str


class OpticalFrameCompensator:
    """
    Optical frame transformation and field rotation compensator.
    
    This class implements the critical transformations between mechanical
    gimbal motion and optical line-of-sight (LOS) vectors, including the
    non-mechanical roll compensation achieved through optical elements
    (K-mirror or prism equivalent).
    
    The transformation chain is:
        Site → Mechanical → Optical → Sensor
        
    Field Rotation:
    ---------------
    When a two-axis Az/El gimbal tracks a moving target, the optical image
    rotates relative to the mechanical frame. For a fixed detector orientation,
    this rotation must be compensated optically to maintain stable sensor
    alignment. The field rotation angle γ_FR is computed from the gimbal
    geometry and compensated via K-mirror rotation.
    
    Mathematical Foundation:
    -----------------------
    LOS vector: L = R_Az(θ_az) · R_El(θ_el) · [0, 0, 1]^T
    Field Rotation: γ_FR = f(θ_az, θ_el, latitude)
    
    For Alt-Az mount: γ_FR ≈ atan2(sin(Az), tan(El)*cos(Az) - cos(El)*sin(lat))
    """
    
    def __init__(self, config: dict):
        """
        Initialize optical frame compensator.
        
        Parameters
        ----------
        config : dict
            Configuration containing:
            - 'site_latitude_deg': Observer latitude [deg] (e.g., 35.0)
            - 'use_field_rotation': Enable field rotation compensation [bool]
            - 'optical_misalignment_az_rad': Az misalignment [rad] (e.g., 1e-5)
            - 'optical_misalignment_el_rad': El misalignment [rad] (e.g., 1e-5)
        """
        self.config = config
        
        # Site parameters
        self.site_latitude_rad: float = np.deg2rad(
            config.get('site_latitude_deg', 35.0)
        )
        
        # Field rotation control
        self.use_field_rotation: bool = config.get('use_field_rotation', True)
        
        # Optical misalignment (mechanical to optical axes)
        self.epsilon_az: float = config.get('optical_misalignment_az_rad', 1.0e-5)
        self.epsilon_el: float = config.get('optical_misalignment_el_rad', 1.0e-5)
        
        # Current state
        self.current_az_rad: float = 0.0
        self.current_el_rad: float = 0.0
        self.current_field_rotation_rad: float = 0.0
        
    @staticmethod
    def rotation_matrix_z(angle_rad: float) -> np.ndarray:
        """
        Compute rotation matrix about Z-axis (azimuth rotation).
        
        R_z(θ) = [cos(θ)  -sin(θ)  0]
                 [sin(θ)   cos(θ)  0]
                 [  0        0     1]
        
        Parameters
        ----------
        angle_rad : float
            Rotation angle [rad]
            
        Returns
        -------
        np.ndarray
            3x3 rotation matrix
        """
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)
        return np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])
    
    @staticmethod
    def rotation_matrix_y(angle_rad: float) -> np.ndarray:
        """
        Compute rotation matrix about Y-axis (elevation rotation).
        
        R_y(θ) = [ cos(θ)  0  sin(θ)]
                 [   0     1    0   ]
                 [-sin(θ)  0  cos(θ)]
        
        Parameters
        ----------
        angle_rad : float
            Rotation angle [rad]
            
        Returns
        -------
        np.ndarray
            3x3 rotation matrix
        """
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)
        return np.array([
            [ c, 0, s],
            [ 0, 1, 0],
            [-s, 0, c]
        ])
    
    def compute_field_rotation_angle(
        self, 
        az_rad: float, 
        el_rad: float
    ) -> float:
        """
        Compute field rotation angle for Alt-Az mount.
        
        The field rotation angle is the rotation of the optical field relative
        to the horizon as the gimbal tracks in azimuth and elevation. This
        must be compensated to keep the detector frame aligned with the target.
        
        For an Alt-Az (Azimuth-Elevation) mount at latitude φ:
        
            tan(γ_FR) = sin(Az) / (cos(El)*sin(φ) - tan(El)*cos(Az)*cos(φ))
        
        Simplified small-angle approximation for near-zenith pointing:
            γ_FR ≈ Az * sin(φ)
        
        Parameters
        ----------
        az_rad : float
            Azimuth angle [rad], measured from North clockwise
        el_rad : float
            Elevation angle [rad], measured from horizon
            
        Returns
        -------
        float
            Field rotation angle [rad]
        """
        if not self.use_field_rotation:
            return 0.0
        
        # Full formula for field rotation (parallactic angle variant)
        sin_az = np.sin(az_rad)
        cos_az = np.cos(az_rad)
        sin_el = np.sin(el_rad)
        cos_el = np.cos(el_rad)
        sin_lat = np.sin(self.site_latitude_rad)
        cos_lat = np.cos(self.site_latitude_rad)
        
        # Numerator: sin(Az)
        numerator = sin_az
        
        # Denominator: tan(El)*cos(Az) - cos(El)*sin(lat)/cos(El)
        # Simplified: tan(El)*cos(Az) - sin(lat)
        denominator = (sin_el / (cos_el + 1e-10)) * cos_az - sin_lat
        
        # Compute field rotation angle
        gamma_fr = np.arctan2(numerator, denominator + 1e-10)
        
        return gamma_fr
    
    def mechanical_to_los_vector(
        self, 
        az_rad: float, 
        el_rad: float
    ) -> np.ndarray:
        """
        Convert mechanical gimbal angles to line-of-sight unit vector.
        
        The LOS vector is computed by applying azimuth and elevation rotations
        to the initial boresight direction [0, 0, 1]^T (pointing up in site frame).
        
        Transformation: L = R_Az(θ_az) · R_El(θ_el) · [0, 0, 1]^T
        
        Parameters
        ----------
        az_rad : float
            Azimuth angle [rad]
        el_rad : float
            Elevation angle [rad]
            
        Returns
        -------
        np.ndarray
            3D unit vector representing line-of-sight direction
        """
        # Initial boresight (pointing up in site frame)
        boresight = np.array([0.0, 0.0, 1.0])
        
        # Apply elevation rotation (about Y-axis)
        R_el = self.rotation_matrix_y(el_rad)
        
        # Apply azimuth rotation (about Z-axis)
        R_az = self.rotation_matrix_z(az_rad)
        
        # Combined rotation: Az then El
        R_total = R_az @ R_el
        
        # Compute LOS vector
        los_vector = R_total @ boresight
        
        return los_vector
    
    def compute_los_error(
        self,
        actual_az_rad: float,
        actual_el_rad: float,
        target_az_rad: float,
        target_el_rad: float
    ) -> Tuple[float, float]:
        """
        Compute line-of-sight pointing error in tip/tilt coordinates.
        
        The LOS error is the angular deviation between the actual and target
        LOS vectors, decomposed into tip (horizontal) and tilt (vertical)
        components in the optical frame.
        
        Small-angle approximation:
            Δtip ≈ Δaz * cos(el)
            Δtilt ≈ Δel
        
        Parameters
        ----------
        actual_az_rad : float
            Actual azimuth angle [rad]
        actual_el_rad : float
            Actual elevation angle [rad]
        target_az_rad : float
            Target azimuth angle [rad]
        target_el_rad : float
            Target elevation angle [rad]
            
        Returns
        -------
        Tuple[float, float]
            (tip_error_rad, tilt_error_rad) - LOS errors in optical frame [rad]
        """
        # Compute actual and target LOS vectors
        los_actual = self.mechanical_to_los_vector(actual_az_rad, actual_el_rad)
        los_target = self.mechanical_to_los_vector(target_az_rad, target_el_rad)
        
        # Compute error vector
        los_error_vec = los_actual - los_target
        
        # Small-angle approximation: project error onto optical frame axes
        # Tip (horizontal) ~ x-component in optical frame
        # Tilt (vertical) ~ y-component in optical frame
        
        # For small errors, use simplified projection
        delta_az = actual_az_rad - target_az_rad
        delta_el = actual_el_rad - target_el_rad
        
        # Include misalignment
        delta_az += self.epsilon_az
        delta_el += self.epsilon_el
        
        # Transform to tip/tilt (small-angle approximation)
        tip_error = delta_az * np.cos(actual_el_rad)  # Az error scaled by elevation
        tilt_error = delta_el  # El error directly maps to tilt
        
        return tip_error, tilt_error
    
    def apply_field_rotation_compensation(
        self,
        tip_error_rad: float,
        tilt_error_rad: float,
        az_rad: float,
        el_rad: float
    ) -> Tuple[float, float]:
        """
        Apply field rotation compensation to tip/tilt errors.
        
        The field rotation compensation rotates the error vector by the
        field rotation angle to align with the detector frame.
        
        [tip']   = [cos(γ)  -sin(γ)] [tip ]
        [tilt']    [sin(γ)   cos(γ)] [tilt]
        
        Parameters
        ----------
        tip_error_rad : float
            Tip error before compensation [rad]
        tilt_error_rad : float
            Tilt error before compensation [rad]
        az_rad : float
            Current azimuth angle [rad]
        el_rad : float
            Current elevation angle [rad]
            
        Returns
        -------
        Tuple[float, float]
            (tip_compensated, tilt_compensated) - Errors in detector frame [rad]
        """
        # Compute field rotation angle
        gamma_fr = self.compute_field_rotation_angle(az_rad, el_rad)
        self.current_field_rotation_rad = gamma_fr
        
        if not self.use_field_rotation or np.abs(gamma_fr) < 1e-10:
            return tip_error_rad, tilt_error_rad
        
        # Apply 2D rotation
        cos_gamma = np.cos(gamma_fr)
        sin_gamma = np.sin(gamma_fr)
        
        tip_compensated = (cos_gamma * tip_error_rad - 
                          sin_gamma * tilt_error_rad)
        tilt_compensated = (sin_gamma * tip_error_rad + 
                           cos_gamma * tilt_error_rad)
        
        return tip_compensated, tilt_compensated
    
    def full_transform(
        self,
        actual_az_rad: float,
        actual_el_rad: float,
        target_az_rad: float,
        target_el_rad: float
    ) -> Tuple[float, float, dict]:
        """
        Complete transformation from mechanical angles to detector-frame errors.
        
        This is the top-level method that performs the full transformation
        chain including LOS error computation and field rotation compensation.
        
        Parameters
        ----------
        actual_az_rad : float
            Actual gimbal azimuth [rad]
        actual_el_rad : float
            Actual gimbal elevation [rad]
        target_az_rad : float
            Target azimuth [rad]
        target_el_rad : float
            Target elevation [rad]
            
        Returns
        -------
        Tuple[float, float, dict]
            - tip_error_rad: Tip error in detector frame [rad]
            - tilt_error_rad: Tilt error in detector frame [rad]
            - metadata: Dictionary with intermediate values
        """
        # Update state
        self.current_az_rad = actual_az_rad
        self.current_el_rad = actual_el_rad
        
        # Step 1: Compute LOS error in optical frame
        tip_optical, tilt_optical = self.compute_los_error(
            actual_az_rad, actual_el_rad,
            target_az_rad, target_el_rad
        )
        
        # Step 2: Apply field rotation compensation
        tip_detector, tilt_detector = self.apply_field_rotation_compensation(
            tip_optical, tilt_optical,
            actual_az_rad, actual_el_rad
        )
        
        # Metadata for debugging/logging
        metadata = {
            'tip_optical_rad': tip_optical,
            'tilt_optical_rad': tilt_optical,
            'field_rotation_rad': self.current_field_rotation_rad,
            'field_rotation_deg': np.rad2deg(self.current_field_rotation_rad)
        }
        
        return tip_detector, tilt_detector, metadata
    
    def get_transformation_matrix(
        self,
        az_rad: float,
        el_rad: float
    ) -> np.ndarray:
        """
        Get complete transformation matrix from site to optical frame.
        
        Parameters
        ----------
        az_rad : float
            Azimuth angle [rad]
        el_rad : float
            Elevation angle [rad]
            
        Returns
        -------
        np.ndarray
            3x3 transformation matrix
        """
        R_az = self.rotation_matrix_z(az_rad)
        R_el = self.rotation_matrix_y(el_rad)
        
        # Field rotation compensation (in-plane rotation)
        gamma_fr = self.compute_field_rotation_angle(az_rad, el_rad)
        R_fr = self.rotation_matrix_z(gamma_fr)
        
        # Combined transformation
        R_total = R_fr @ R_az @ R_el
        
        return R_total
