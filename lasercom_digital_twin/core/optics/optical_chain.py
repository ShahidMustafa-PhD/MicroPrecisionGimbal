"""
Optical Chain Model for Laser Communication Terminal

This module implements the physics of light propagation through the telescope
optical system, including beam steering via the Fast Steering Mirror (FSM)
and focal plane mapping for detector positioning.

Key Physics:
- Small-angle approximation for all optical calculations
- Paraxial optics for beam propagation
- FSM mirror gain (2x for reflected beam)
- Focal plane projection geometry
"""

import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class FocalPlanePosition:
    """Position on the detector focal plane."""
    x_um: float  # Micrometers
    y_um: float  # Micrometers
    x_m: float   # Meters
    y_m: float   # Meters


class TelescopeOptics:
    """
    Telescope optical chain model for precision pointing analysis.
    
    This class models the conversion from angular pointing errors (in radians)
    to physical positions on the detector focal plane (in micrometers). The
    model uses the small-angle approximation throughout, which is valid for
    the sub-milliradian errors typical in precision pointing systems.
    
    Optical Path:
    ------------
    1. Incoming beam (target line-of-sight)
    2. Coarse pointing via gimbal (Az/El)
    3. Fast Steering Mirror (FSM) fine correction
    4. Telescope primary optics (focal length f)
    5. Focal plane (detector position)
    
    Small-Angle Approximation:
    -------------------------
    For θ << 1 rad:
        tan(θ) ≈ θ
        sin(θ) ≈ θ
        cos(θ) ≈ 1
        
    Focal Plane Position:
        x_focal = f * tan(θ_x) ≈ f * θ_x
        y_focal = f * tan(θ_y) ≈ f * θ_y
    
    FSM Correction:
    --------------
    The FSM is a flat mirror that steers the beam. For a plane mirror at angle α,
    the reflected beam is deflected by 2α. Thus:
        
        θ_beam = G_FSM * α_mirror
        
    where G_FSM = 2 for a standard FSM configuration.
    """
    
    def __init__(self, config: dict):
        """
        Initialize telescope optics model.
        
        Parameters
        ----------
        config : dict
            Configuration containing:
            - 'focal_length_m': Telescope focal length [m] (e.g., 1.5)
            - 'aperture_diameter_m': Primary aperture diameter [m] (e.g., 0.3)
            - 'fsm_gain': FSM angle-to-beam gain factor (typically 2.0)
            - 'wavelength_um': Laser wavelength [µm] (e.g., 1.55)
            - 'beam_diameter_mm': Input beam diameter [mm] (e.g., 50.0)
            - 'detector_size_mm': QPD active area size [mm] (e.g., 5.0)
        """
        self.config = config
        
        # Telescope parameters
        self.focal_length_m: float = config.get('focal_length_m', 1.5)
        self.aperture_diameter_m: float = config.get('aperture_diameter_m', 0.3)
        
        # FSM parameters
        self.fsm_gain: float = config.get('fsm_gain', 2.0)  # Mirror factor of 2
        
        # Optical parameters
        self.wavelength_um: float = config.get('wavelength_um', 1.55)
        self.wavelength_m: float = self.wavelength_um * 1e-6
        self.beam_diameter_mm: float = config.get('beam_diameter_mm', 50.0)
        self.beam_diameter_m: float = self.beam_diameter_mm * 1e-3
        
        # Detector parameters
        self.detector_size_mm: float = config.get('detector_size_mm', 5.0)
        self.detector_size_m: float = self.detector_size_mm * 1e-3
        
        # Compute derived parameters
        self._compute_optical_parameters()
        
    def _compute_optical_parameters(self) -> None:
        """
        Compute derived optical parameters.
        
        Calculates:
        - Diffraction-limited spot size
        - F-number
        - Focal plane scale (arcsec/mm)
        """
        # F-number: F/# = f / D
        self.f_number: float = self.focal_length_m / self.aperture_diameter_m
        
        # Diffraction-limited spot size (Airy disk radius to first null)
        # θ_airy = 1.22 * λ / D  [rad]
        self.airy_disk_radius_rad: float = (1.22 * self.wavelength_m / 
                                            self.aperture_diameter_m)
        
        # Physical spot size on focal plane
        # r_spot = f * θ_airy
        self.airy_disk_radius_m: float = (self.focal_length_m * 
                                          self.airy_disk_radius_rad)
        self.airy_disk_radius_um: float = self.airy_disk_radius_m * 1e6
        
        # Gaussian beam spot size (1/e² intensity radius)
        # For Gaussian beam: w_0 ≈ 0.5 * Airy disk diameter
        self.spot_size_radius_um: float = 0.5 * 2 * self.airy_disk_radius_um
        
        # Focal plane scale: rad/m
        self.focal_plane_scale_rad_per_m: float = 1.0 / self.focal_length_m
        
        # Focal plane scale: arcsec/mm
        self.focal_plane_scale_arcsec_per_mm: float = (
            np.rad2deg(self.focal_plane_scale_rad_per_m) * 3600.0 * 1e-3
        )
        
    def angle_to_focal_plane_position(
        self,
        angle_x_rad: float,
        angle_y_rad: float
    ) -> FocalPlanePosition:
        """
        Convert angular error to focal plane position.
        
        Uses small-angle approximation:
            x_focal = f * tan(θ_x) ≈ f * θ_x
            y_focal = f * tan(θ_y) ≈ f * θ_y
        
        Parameters
        ----------
        angle_x_rad : float
            Angular error in X (tip) [rad]
        angle_y_rad : float
            Angular error in Y (tilt) [rad]
            
        Returns
        -------
        FocalPlanePosition
            Position on focal plane in both meters and micrometers
        """
        # Small-angle approximation: x = f * θ
        x_m = self.focal_length_m * angle_x_rad
        y_m = self.focal_length_m * angle_y_rad
        
        # Convert to micrometers
        x_um = x_m * 1e6
        y_um = y_m * 1e6
        
        return FocalPlanePosition(x_um=x_um, y_um=y_um, x_m=x_m, y_m=y_m)
    
    def focal_plane_position_to_angle(
        self,
        x_m: float,
        y_m: float
    ) -> Tuple[float, float]:
        """
        Convert focal plane position to angular error (inverse operation).
        
        Parameters
        ----------
        x_m : float
            X position on focal plane [m]
        y_m : float
            Y position on focal plane [m]
            
        Returns
        -------
        Tuple[float, float]
            (angle_x_rad, angle_y_rad) - Angular errors [rad]
        """
        angle_x_rad = x_m / self.focal_length_m
        angle_y_rad = y_m / self.focal_length_m
        
        return angle_x_rad, angle_y_rad
    
    def apply_fsm_correction(
        self,
        los_error_x_rad: float,
        los_error_y_rad: float,
        fsm_command_x_rad: float,
        fsm_command_y_rad: float
    ) -> Tuple[float, float]:
        """
        Apply FSM correction to line-of-sight error.
        
        The FSM steers the beam by deflecting it with a small mirror. For a
        plane mirror, the beam deflection angle is twice the mirror rotation
        angle (law of reflection).
        
        Corrected LOS = Original LOS + G_FSM * FSM_angle
        
        where G_FSM = 2.0 for standard configuration.
        
        Parameters
        ----------
        los_error_x_rad : float
            Uncorrected LOS error in X (tip) [rad]
        los_error_y_rad : float
            Uncorrected LOS error in Y (tilt) [rad]
        fsm_command_x_rad : float
            FSM mirror angle command in X [rad]
        fsm_command_y_rad : float
            FSM mirror angle command in Y [rad]
            
        Returns
        -------
        Tuple[float, float]
            (corrected_x_rad, corrected_y_rad) - Residual LOS errors [rad]
        """
        # FSM correction (beam deflection = gain * mirror angle)
        fsm_correction_x = self.fsm_gain * fsm_command_x_rad
        fsm_correction_y = self.fsm_gain * fsm_command_y_rad
        
        # Apply correction (FSM steers beam to reduce error)
        corrected_x = los_error_x_rad - fsm_correction_x
        corrected_y = los_error_y_rad - fsm_correction_y
        
        return corrected_x, corrected_y
    
    def compute_spot_position(
        self,
        los_error_tip_rad: float,
        los_error_tilt_rad: float,
        fsm_tip_rad: float = 0.0,
        fsm_tilt_rad: float = 0.0
    ) -> FocalPlanePosition:
        """
        Compute final spot position on focal plane including FSM correction.
        
        This is the main method that combines LOS error and FSM correction
        to determine where the laser spot appears on the detector.
        
        Parameters
        ----------
        los_error_tip_rad : float
            Line-of-sight error in tip (X) direction [rad]
        los_error_tilt_rad : float
            Line-of-sight error in tilt (Y) direction [rad]
        fsm_tip_rad : float, optional
            FSM correction in tip direction [rad]
        fsm_tilt_rad : float, optional
            FSM correction in tilt direction [rad]
            
        Returns
        -------
        FocalPlanePosition
            Laser spot position on detector focal plane
        """
        # Apply FSM correction to LOS error
        residual_tip, residual_tilt = self.apply_fsm_correction(
            los_error_tip_rad,
            los_error_tilt_rad,
            fsm_tip_rad,
            fsm_tilt_rad
        )
        
        # Convert residual error to focal plane position
        spot_position = self.angle_to_focal_plane_position(
            residual_tip,
            residual_tilt
        )
        
        return spot_position
    
    def get_pointing_accuracy_requirement(
        self,
        required_spot_centering_um: float
    ) -> float:
        """
        Compute angular pointing accuracy required for spot centering.
        
        This method answers: "What pointing accuracy (in µrad) do I need
        to keep the spot within X micrometers of the detector center?"
        
        Parameters
        ----------
        required_spot_centering_um : float
            Maximum allowable spot deviation [µm]
            
        Returns
        -------
        float
            Required pointing accuracy [µrad]
        """
        # Convert micrometers to meters
        required_spot_m = required_spot_centering_um * 1e-6
        
        # Use focal plane scale to get angular requirement
        required_angle_rad = required_spot_m / self.focal_length_m
        
        # Convert to microradians
        required_angle_urad = required_angle_rad * 1e6
        
        return required_angle_urad
    
    def get_sensitivity_analysis(self) -> Dict[str, float]:
        """
        Provide sensitivity metrics for system design.
        
        Returns key optical parameters that define the pointing accuracy
        requirements and sensor sensitivity.
        
        Returns
        -------
        Dict[str, float]
            Dictionary of optical sensitivity parameters
        """
        return {
            'focal_length_m': self.focal_length_m,
            'f_number': self.f_number,
            'airy_disk_radius_um': self.airy_disk_radius_um,
            'spot_size_radius_um': self.spot_size_radius_um,
            'focal_plane_scale_arcsec_per_mm': self.focal_plane_scale_arcsec_per_mm,
            'focal_plane_scale_urad_per_um': 1.0 / self.focal_length_m * 1e6,
            'fsm_gain': self.fsm_gain,
            'detector_size_mm': self.detector_size_mm,
            '1_urad_to_focal_plane_um': self.focal_length_m * 1e-6 * 1e6,
            'wavelength_um': self.wavelength_um
        }
    
    def get_visualization_grid(
        self,
        grid_size_mm: float = 2.0,
        grid_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate focal plane coordinate grid for visualization.
        
        Creates a meshgrid suitable for plotting spot positions and
        detector boundaries.
        
        Parameters
        ----------
        grid_size_mm : float, optional
            Half-width of grid [mm]
        grid_points : int, optional
            Number of points in each direction
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (X, Y) - Meshgrid arrays in micrometers
        """
        # Create grid in millimeters
        grid_mm = np.linspace(-grid_size_mm, grid_size_mm, grid_points)
        X_mm, Y_mm = np.meshgrid(grid_mm, grid_mm)
        
        # Convert to micrometers
        X_um = X_mm * 1e3
        Y_um = Y_mm * 1e3
        
        return X_um, Y_um
    
    def is_within_detector(
        self,
        spot_position: FocalPlanePosition
    ) -> bool:
        """
        Check if spot position is within the detector active area.
        
        Parameters
        ----------
        spot_position : FocalPlanePosition
            Spot position on focal plane
            
        Returns
        -------
        bool
            True if spot is within detector bounds
        """
        half_size = self.detector_size_m / 2.0
        
        within_x = np.abs(spot_position.x_m) <= half_size
        within_y = np.abs(spot_position.y_m) <= half_size
        
        return within_x and within_y
