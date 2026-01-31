"""
Gimbal Model - Digital Twin Geometry Generator
===============================================
Exact replica of the 2-axis laser communication gimbal CAD model.

This module contains ONLY geometry generation and transformation logic.
No simulation loops or trajectory planning should exist here.

Coordinate System (matching CAD image view):
- X: Forward (towards viewer in CAD)
- Y: Right (left/right in CAD, elevation axis direction)
- Z: Up (vertical, azimuth rotation axis, optical boresight direction)

Key Geometry (from CAD):
- Square mounting plate with 4 corner holes
- Cylindrical pedestal rising from plate
- U-shaped yoke with arms along Y-axis (left/right)
- Vertical optical barrel (axis along Z, perpendicular to Y elevation axis)
- Motor housings on both sides of barrel

Author: Expert Robotics Simulation Engineer
Date: January 31, 2026
"""

import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation as R
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class GimbalGeometry:
    """Storage for original mesh geometries (for resetting transforms)."""
    base_meshes: List[pv.PolyData]
    yoke_meshes: List[pv.PolyData]
    housing_meshes: List[pv.PolyData]


class AzimuthBaseAssembly:
    """
    Azimuth base assembly matching CAD model exactly:
    - Square mounting plate with 4 corner mounting holes
    - Cylindrical pedestal/column rising from center
    - Bearing interface ring at top
    
    This assembly is FIXED (does not rotate).
    """
    
    def __init__(self):
        """Initialize azimuth base geometry."""
        self.actors = []
        self.original_meshes = []
    
    def build(self, plotter: pv.Plotter) -> List:
        """
        Construct azimuth base components matching CAD.
        """
        # === Square Mounting Plate ===
        plate_size = 90
        plate_thickness = 6
        mount_plate = pv.Box(
            bounds=[
                -plate_size/2, plate_size/2,   # X
                -plate_size/2, plate_size/2,   # Y
                0, plate_thickness              # Z
            ]
        )
        self.original_meshes.append(mount_plate.copy())
        actor = plotter.add_mesh(
            mount_plate,
            color='#6B7B8C',  # Gray steel matching CAD
            metallic=0.85,
            roughness=0.35,
            pbr=True,
            name='mount_plate'
        )
        self.actors.append(actor)
        
        # Corner mounting holes (4 corners) - dark circles
        hole_offset = 38
        hole_radius = 3.5
        for x_sign in [-1, 1]:
            for y_sign in [-1, 1]:
                hole = pv.Cylinder(
                    center=(x_sign * hole_offset, y_sign * hole_offset, plate_thickness/2),
                    direction=(0, 0, 1),
                    radius=hole_radius,
                    height=plate_thickness + 1
                )
                self.original_meshes.append(hole.copy())
                actor = plotter.add_mesh(
                    hole,
                    color='#0A0A0A',
                    pbr=True
                )
                self.actors.append(actor)
        
        # === Cylindrical Pedestal ===
        pedestal_height = 50
        pedestal_radius = 25
        pedestal_z_start = plate_thickness
        pedestal = pv.Cylinder(
            center=(0, 0, pedestal_z_start + pedestal_height/2),
            direction=(0, 0, 1),
            radius=pedestal_radius,
            height=pedestal_height
        )
        self.original_meshes.append(pedestal.copy())
        actor = plotter.add_mesh(
            pedestal,
            color='#6B7B8C',  # Same gray as plate
            metallic=0.85,
            roughness=0.35,
            pbr=True,
            name='pedestal'
        )
        self.actors.append(actor)
        
        # Bearing ring at top of pedestal
        bearing_z = pedestal_z_start + pedestal_height
        bearing_ring = pv.Cylinder(
            center=(0, 0, bearing_z + 3),
            direction=(0, 0, 1),
            radius=pedestal_radius + 4,
            height=6
        )
        self.original_meshes.append(bearing_ring.copy())
        actor = plotter.add_mesh(
            bearing_ring,
            color='#6B7B8C',  # Aluminum matching base
            metallic=0.85,
            roughness=0.35,
            pbr=True
        )
        self.actors.append(actor)
        
        # Decorative holes on pedestal face
        for angle_deg in [45, 135, 225, 315]:
            angle_rad = np.deg2rad(angle_deg)
            x = pedestal_radius * 0.7 * np.cos(angle_rad)
            y = pedestal_radius * 0.7 * np.sin(angle_rad)
            for z_offset in [20, 35]:
                hole = pv.Cylinder(
                    center=(x, y, pedestal_z_start + z_offset),
                    direction=(np.cos(angle_rad), np.sin(angle_rad), 0),
                    radius=2,
                    height=5
                )
                self.original_meshes.append(hole.copy())
                actor = plotter.add_mesh(
                    hole,
                    color='#1A1A1A',
                    pbr=True
                )
                self.actors.append(actor)
        
        return self.actors


class ElevationYokeAssembly:
    """
    U-shaped yoke/fork assembly matching CAD model exactly:
    - Horizontal base connecting to azimuth
    - Two vertical arms along Y-axis (LEFT and RIGHT sides)
    - Rounded tops on arms
    - Elevation pivot bearings
    
    This rotates with azimuth (about Z-axis).
    Arms are on LEFT and RIGHT (Y-axis), providing Y-axis elevation pivot.
    """
    
    def __init__(self):
        """Initialize yoke geometry."""
        self.actors = []
        self.original_meshes = []
        self.yoke_base_z = 62  # Height where yoke mounts (top of pedestal + bearing)
    
    def build(self, plotter: pv.Plotter) -> List:
        """
        Construct yoke frame components matching CAD exactly.
        Arms are along Y-axis (left/right in CAD view).
        """
        base_z = self.yoke_base_z
        arm_height = 75
        arm_thickness = 15  # X dimension (front-back)
        arm_width = 18      # Y dimension (left-right thickness of each arm)
        arm_y_pos = 45      # Y position of arm centers
        
        # === Yoke Base Bar (Horizontal, spans between arms) ===
        yoke_base = pv.Box(
            bounds=[
                -arm_thickness/2, arm_thickness/2,           # X (thin front-back)
                -arm_y_pos - arm_width/2, arm_y_pos + arm_width/2,  # Y (wide left-right)
                base_z, base_z + arm_width                   # Z
            ]
        )
        self.original_meshes.append(yoke_base.copy())
        actor = plotter.add_mesh(
            yoke_base,
            color='#6B7B8C',  # Gray steel
            metallic=0.85,
            roughness=0.35,
            pbr=True,
            name='yoke_base'
        )
        self.actors.append(actor)
        
        # === Yoke Vertical Arms (LEFT and RIGHT along Y-axis) ===
        for y_sign in [-1, 1]:
            y_center = y_sign * arm_y_pos
            
            # Main vertical arm body
            arm = pv.Box(
                bounds=[
                    -arm_thickness/2, arm_thickness/2,           # X
                    y_center - arm_width/2, y_center + arm_width/2,  # Y
                    base_z, base_z + arm_height                  # Z
                ]
            )
            self.original_meshes.append(arm.copy())
            actor = plotter.add_mesh(
                arm,
                color='#6B7B8C',
                metallic=0.85,
                roughness=0.35,
                pbr=True
            )
            self.actors.append(actor)
            
            # Rounded top cap (cylinder along Y)
            cap_z = base_z + arm_height
            cap = pv.Cylinder(
                center=(0, y_center, cap_z),
                direction=(0, 1, 0),
                radius=arm_thickness/2,
                height=arm_width
            )
            self.original_meshes.append(cap.copy())
            actor = plotter.add_mesh(
                cap,
                color='#6B7B8C',
                metallic=0.85,
                roughness=0.35,
                pbr=True
            )
            self.actors.append(actor)
            
            # Decorative holes on arm face (matching CAD)
            for z_offset in [15, 30, 45, 60]:
                hole = pv.Cylinder(
                    center=(arm_thickness/2 * 0.9, y_center, base_z + z_offset),
                    direction=(1, 0, 0),
                    radius=2.5,
                    height=4
                )
                self.original_meshes.append(hole.copy())
                actor = plotter.add_mesh(
                    hole,
                    color='#1A1A1A',
                    pbr=True
                )
                self.actors.append(actor)
        
        return self.actors


class ElevationHousingAssembly:
    """
    Optical telescope barrel assembly matching CAD model exactly:
    - VERTICAL cylindrical barrel (axis along Z)
    - Barrel is PERPENDICULAR to elevation axis (Y-axis)
    - White/light gray front lens/aperture at TOP
    - Motor housings on BOTH sides (left and right, along Y)
    
    Elevation rotation is about Y-axis.
    The barrel axis (Z) is at 90 degrees to the elevation axis (Y).
    """
    
    def __init__(self):
        """Initialize housing geometry."""
        self.actors = []
        self.original_meshes = []
        # Elevation pivot point - where barrel rotates
        # Located at center height of yoke arms
        self.pivot_z = 110  # Mid-height of yoke
        self.elevation_pivot = np.array([0, 0, self.pivot_z])
    
    def build(self, plotter: pv.Plotter) -> List:
        """
        Construct optical housing matching CAD exactly.
        
        Barrel is VERTICAL (along Z-axis).
        Perpendicular to elevation axis (Y-axis).
        """
        pivot_z = self.pivot_z
        barrel_radius = 28  # Reduced from 32 to avoid yoke collision
        barrel_length = 80
        
        # === Main Telescope Barrel (VERTICAL) ===
        # Axis along Z, centered at pivot height
        barrel = pv.Cylinder(
            center=(0, 0, pivot_z),
            direction=(0, 0, 1),  # Pointing UP
            radius=barrel_radius,
            height=barrel_length
        )
        self.original_meshes.append(barrel.copy())
        actor = plotter.add_mesh(
            barrel,
            color='#2D3748',  # Dark blue-gray (anodized aluminum)
            metallic=0.75,
            roughness=0.5,
            pbr=True,
            name='telescope_barrel'
        )
        self.actors.append(actor)
        
        # === Front Lens/Aperture (TOP of barrel) ===
        # Transparent foggy white glass
        lens_z = pivot_z + barrel_length/2
        front_lens = pv.Disc(
            center=(0, 0, lens_z - 2),
            inner=0,
            outer=barrel_radius - 4,
            normal=(0, 0, 1),
            c_res=64
        )
        self.original_meshes.append(front_lens.copy())
        actor = plotter.add_mesh(
            front_lens,
            color='white',  # Foggy white
            opacity=0.35,   # Semi-transparent
            specular=0.9,
            specular_power=100,
            pbr=True,
            name='front_lens'
        )
        self.actors.append(actor)
        
        # Lens retaining ring (top edge)
        lens_ring = pv.Cylinder(
            center=(0, 0, lens_z - 3),
            direction=(0, 0, 1),
            radius=barrel_radius + 1,
            height=4
        )
        self.original_meshes.append(lens_ring.copy())
        actor = plotter.add_mesh(
            lens_ring,
            color='#2D3748',
            metallic=0.75,
            roughness=0.5,
            pbr=True
        )
        self.actors.append(actor)
        
        # === Rear of Barrel (Bottom) ===
        rear_z = pivot_z - barrel_length/2
        rear_cap = pv.Disc(
            center=(0, 0, rear_z + 2),
            inner=0,
            outer=barrel_radius - 5,
            normal=(0, 0, -1),
            c_res=64
        )
        self.original_meshes.append(rear_cap.copy())
        actor = plotter.add_mesh(
            rear_cap,
            color='#1A202C',
            metallic=0.3,
            roughness=0.7,
            pbr=True
        )
        self.actors.append(actor)
        
        # === Motor Housings on BOTH sides (matching CAD) ===
        # Black rectangular boxes attached to left and right of barrel
        # CRITICAL: Position motors OUTBOARD of yoke arms (±54mm outer edge)
        motor_size_x = 28  # Front-back
        motor_size_y = 16  # Left-right thickness
        motor_size_z = 26  # Height
        # Position motor inner face outboard of yoke outer edge
        motor_y_offset = 62  # Motor center: inner=54mm, outer=70mm → clearance OK
        # Clearance: 0mm from yoke (motor starts exactly at yoke edge)
        
        for y_sign in [-1, 1]:
            y_center = y_sign * motor_y_offset
            
            # Main motor housing (black box)
            motor_box = pv.Box(
                bounds=[
                    -motor_size_x/2, motor_size_x/2,           # X
                    y_center - motor_size_y/2, y_center + motor_size_y/2,  # Y
                    pivot_z - motor_size_z/2, pivot_z + motor_size_z/2    # Z
                ]
            )
            self.original_meshes.append(motor_box.copy())
            actor = plotter.add_mesh(
                motor_box,
                color='#1A1A1A',  # Matte black
                metallic=0.1,
                roughness=0.9,
                pbr=True,
                name=f'motor_{"left" if y_sign < 0 else "right"}'
            )
            self.actors.append(actor)
            
            # Motor end cap (cylindrical, protruding outward)
            cap_y = y_center + y_sign * motor_size_y/2
            motor_cap = pv.Cylinder(
                center=(0, cap_y + y_sign * 4, pivot_z),
                direction=(0, y_sign, 0),
                radius=12,
                height=8
            )
            self.original_meshes.append(motor_cap.copy())
            actor = plotter.add_mesh(
                motor_cap,
                color='#1A1A1A',
                metallic=0.1,
                roughness=0.9,
                pbr=True
            )
            self.actors.append(actor)
            
            # Mounting bracket connecting motor to barrel (bridges the gap)
            bracket_y_start = barrel_radius + 1  # Just outside barrel
            bracket_y_end = y_center - y_sign * motor_size_y/2  # Motor inner face
            bracket_center = (bracket_y_start + bracket_y_end) / 2 * y_sign
            bracket_length = abs(bracket_y_end - bracket_y_start)
            
            mounting_bracket = pv.Box(
                bounds=[
                    -8, 8,  # Narrow X (front-back)
                    bracket_center - bracket_length/2, bracket_center + bracket_length/2,  # Y
                    pivot_z - 10, pivot_z + 10  # Z height at pivot level
                ]
            )
            self.original_meshes.append(mounting_bracket.copy())
            actor = plotter.add_mesh(
                mounting_bracket,
                color='#4A5568',  # Steel bracket
                metallic=0.7,
                roughness=0.4,
                pbr=True
            )
            self.actors.append(actor)
            
            # Shaft bearing (smaller cylinder at end)
            shaft = pv.Cylinder(
                center=(0, cap_y + y_sign * 12, pivot_z),
                direction=(0, y_sign, 0),
                radius=9,
                height=6
            )
            self.original_meshes.append(shaft.copy())
            actor = plotter.add_mesh(
                shaft,
                color='#4A5568',
                metallic=0.6,
                roughness=0.4,
                pbr=True
            )
            self.actors.append(actor)
        
        return self.actors


class GimbalModel:
    """
    Complete gimbal digital twin - exact replica of CAD model.
    
    Hierarchical kinematic chain:
    - Base Assembly (FIXED) - Square plate + cylindrical pedestal
    - Yoke Assembly (rotates about Z - azimuth) - U-shaped fork, arms along Y
    - Housing Assembly (rotates about Y - elevation) - Vertical barrel along Z
    
    Coordinate System:
    - X: Forward (towards camera in CAD view)
    - Y: Right (left-right, elevation rotation axis)
    - Z: Up (azimuth rotation axis, barrel points up)
    """
    
    def __init__(self, plotter: pv.Plotter):
        """
        Initialize gimbal model and build all assemblies.
        
        Parameters:
        -----------
        plotter : pv.Plotter
            PyVista plotter for rendering
        """
        self.plotter = plotter
        
        # Current state (radians)
        self.az_angle = 0.0
        self.el_angle = 0.0
        
        # Build assemblies
        self.base = AzimuthBaseAssembly()
        self.yoke = ElevationYokeAssembly()
        self.housing = ElevationHousingAssembly()
        
        # Construct geometry
        self.base_actors = self.base.build(plotter)
        self.yoke_actors = self.yoke.build(plotter)
        self.housing_actors = self.housing.build(plotter)
        
        # Store original geometries for transforms
        self._store_original_geometries()
    
    def _store_original_geometries(self):
        """Cache original mesh geometries for efficient transforms."""
        self.original_base = [actor.GetMapper().GetInput().copy() 
                              for actor in self.base_actors]
        self.original_yoke = [actor.GetMapper().GetInput().copy() 
                              for actor in self.yoke_actors]
        self.original_housing = [actor.GetMapper().GetInput().copy() 
                                 for actor in self.housing_actors]
    
    def update_pose(self, az_deg: float, el_deg: float):
        """
        Update gimbal orientation with hierarchical transforms.
        
        Parameters:
        -----------
        az_deg : float
            Azimuth angle in degrees (rotation about Z-axis)
        el_deg : float
            Elevation angle in degrees (rotation about Y-axis)
            
        Transform Chain:
        1. Base: Fixed (no transform)
        2. Yoke: Rotate about Z (azimuth only)
        3. Housing: Rotate about Y (elevation), then about Z (azimuth)
        """
        # Convert to radians
        self.az_angle = np.deg2rad(az_deg)
        self.el_angle = np.deg2rad(el_deg)
        
        # Elevation pivot point
        el_pivot = self.housing.elevation_pivot
        
        # Rotation matrices
        R_az = R.from_euler('z', self.az_angle).as_matrix()
        R_el = R.from_euler('y', self.el_angle).as_matrix()
        
        # === Base remains FIXED ===
        
        # === Transform Yoke (azimuth rotation only) ===
        for i, actor in enumerate(self.yoke_actors):
            mesh = actor.GetMapper().GetInput()
            original = self.original_yoke[i]
            
            # Rotate about Z-axis (azimuth)
            rotated = original.points @ R_az.T
            mesh.points = rotated
            mesh.Modified()
        
        # === Transform Housing (elevation + azimuth) ===
        for i, actor in enumerate(self.housing_actors):
            mesh = actor.GetMapper().GetInput()
            original = self.original_housing[i]
            
            # Step 1: Translate to pivot origin
            centered = original.points - el_pivot
            
            # Step 2: Apply elevation rotation (about Y-axis)
            rotated_el = centered @ R_el.T
            
            # Step 3: Apply azimuth rotation (about Z-axis)
            rotated_both = rotated_el @ R_az.T
            
            # Step 4: Translate back (pivot also rotated by azimuth)
            final = rotated_both + (el_pivot @ R_az.T)
            
            mesh.points = final
            mesh.Modified()
    
    def get_current_pose(self) -> Tuple[float, float]:
        """
        Get current gimbal orientation.
        
        Returns:
        --------
        tuple : (azimuth_deg, elevation_deg)
        """
        return (np.rad2deg(self.az_angle), np.rad2deg(self.el_angle))
    
    def reset_pose(self):
        """Reset gimbal to home position (0, 0)."""
        self.update_pose(0.0, 0.0)
