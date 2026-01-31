"""
Gimbal Model - High-Fidelity Digital Twin Geometry
===================================================
Professional replica of 2-axis laser communication gimbal CAD model.

This module generates industrial-grade CAD geometry with:
- Machined aluminum structural components
- Proper kinematic constraints (zero collisions)
- PBR materials matching reference imagery
- Accurate drive train detail (motors/encoders on yoke arms)

Coordinate System (CAD Standard):
- X: Forward (towards viewer)
- Y: Right (elevation rotation axis)  
- Z: Up (azimuth rotation axis)

Kinematic Chain:
- Base Assembly (FIXED) - Square plate + cylindrical tower
- Yoke Assembly (rotates about Z) - U-fork with motor mounts on arms
- Barrel Assembly (rotates about Y) - Telescope tube suspended between arms

Author: Senior Mechanical CAD Designer
Date: January 31, 2026
"""

import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation as R
from typing import List, Tuple


# =============================================================================
# MATERIAL DEFINITIONS (PBR)
# =============================================================================

class Materials:
    """PBR material presets matching CAD reference image."""
    
    # Machined aluminum - light gray, highly reflective
    MACHINED_ALUMINUM = {
        'color': '#8C919A',
        'metallic': 1.0,
        'roughness': 0.35,
        'pbr': True
    }
    
    # Black anodized aluminum - dark, semi-matte
    BLACK_ANODIZED = {
        'color': '#1F1F1F',
        'metallic': 0.6,
        'roughness': 0.45,
        'pbr': True
    }
    
    # Black engineering plastic - matte
    BLACK_PLASTIC = {
        'color': '#0D0D0D',
        'metallic': 0.0,
        'roughness': 0.85,
        'pbr': True
    }
    
    # Steel bearing/shaft
    STEEL = {
        'color': '#4A4A4A',
        'metallic': 0.9,
        'roughness': 0.4,
        'pbr': True
    }
    
    # Dark hole/recess
    VOID = {
        'color': '#050505',
        'metallic': 0.0,
        'roughness': 1.0,
        'pbr': True
    }


# =============================================================================
# GEOMETRY DIMENSIONS (mm)
# =============================================================================

class Dimensions:
    """
    Master dimension table ensuring collision-free 360° rotation.
    
    CRITICAL CONSTRAINT: Barrel must clear yoke arms during full elevation sweep.
    Solution: Barrel diameter < gap between yoke arm inner faces.
    """
    
    # --- Base Plate ---
    BASE_PLATE_SIZE = 90.0       # Square plate edge length
    BASE_PLATE_THICK = 8.0       # Plate thickness
    BASE_HOLE_OFFSET = 36.0      # Corner hole distance from center
    BASE_HOLE_RADIUS = 4.0       # Mounting hole radius
    
    # --- Pedestal Tower ---
    PEDESTAL_RADIUS = 28.0       # Tower radius
    PEDESTAL_HEIGHT = 55.0       # Tower height above plate
    PEDESTAL_FLANGE_H = 8.0      # Top flange height
    PEDESTAL_FLANGE_R = 32.0     # Top flange radius
    
    # --- Yoke Fork ---
    YOKE_BASE_Z = 71.0           # Z-height where yoke mounts (plate + pedestal + flange)
    YOKE_BASE_THICK = 12.0       # Horizontal bar thickness (Z)
    YOKE_ARM_HEIGHT = 100.0       # Vertical arm height
    YOKE_ARM_THICK_X = 18.0      # Arm thickness (X direction, front-back)
    YOKE_ARM_THICK_Y = 15.0      # Arm thickness (Y direction, left-right)
    YOKE_ARM_SPACING = 50.0      # Distance from center to arm centerline (Y)
    YOKE_ARM_INNER = 42.5        # Inner face Y-position (spacing - thick/2)
    YOKE_ARM_OUTER = 57.5        # Outer face Y-position (spacing + thick/2)
    
    # --- Elevation Barrel ---
    # Top of Yoke = YOKE_BASE_Z + YOKE_ARM_HEIGHT = 171.0
    # 10mm from top = 161.0
    ELEVATION_PIVOT_Z = 161.0    # Z-height of elevation axis
    BARREL_RADIUS = 32.0         # Telescope barrel radius
    BARREL_LENGTH = 120.0         # Barrel length along optical axis
    BARREL_CLEARANCE = 20.5      # Gap between barrel and yoke arms (42.5 - 32 = 10.5mm)
    
    # --- Motor/Encoder Housings (mounted on yoke arms) ---
    MOTOR_RADIUS = 18.0          # Motor housing radius
    MOTOR_LENGTH = 32.0          # Motor body length (protrudes outward from arm)
    MOTOR_CAP_RADIUS = 12.0      # End cap radius
    MOTOR_CAP_LENGTH = 8.0       # End cap protrusion


# =============================================================================
# AZIMUTH BASE ASSEMBLY
# =============================================================================

class AzimuthBaseAssembly:
    """
    Fixed base assembly - machined aluminum.
    
    Components:
    - Square mounting plate with chamfered edges
    - 4 corner mounting holes with counterbores
    - Cylindrical pedestal tower
    - Top flange with bearing interface
    - Decorative machined features
    """
    
    def __init__(self):
        self.actors = []
        self.original_meshes = []
    
    def build(self, plotter: pv.Plotter) -> List:
        """Construct all base assembly components."""
        D = Dimensions
        M = Materials.MACHINED_ALUMINUM
        
        # === Square Mounting Plate ===
        plate = pv.Box(bounds=[
            -D.BASE_PLATE_SIZE/2, D.BASE_PLATE_SIZE/2,
            -D.BASE_PLATE_SIZE/2, D.BASE_PLATE_SIZE/2,
            0, D.BASE_PLATE_THICK
        ])
        self._add_mesh(plotter, plate, **M, name='base_plate')
        
        # Corner mounting holes (counterbored appearance)
        for x_sign in [-1, 1]:
            for y_sign in [-1, 1]:
                x = x_sign * D.BASE_HOLE_OFFSET
                y = y_sign * D.BASE_HOLE_OFFSET
                
                # Through hole (dark)
                hole = pv.Cylinder(
                    center=(x, y, D.BASE_PLATE_THICK/2),
                    direction=(0, 0, 1),
                    radius=D.BASE_HOLE_RADIUS,
                    height=D.BASE_PLATE_THICK + 1
                )
                self._add_mesh(plotter, hole, **Materials.VOID)
                
                # Counterbore ring (subtle)
                cbore = pv.Cylinder(
                    center=(x, y, D.BASE_PLATE_THICK - 1),
                    direction=(0, 0, 1),
                    radius=D.BASE_HOLE_RADIUS + 2,
                    height=2.5
                )
                self._add_mesh(plotter, cbore, color="#F37979", metallic=0.8, 
                              roughness=0.5, pbr=True)
        
        # === Pedestal Tower ===
        pedestal_z = D.BASE_PLATE_THICK + D.PEDESTAL_HEIGHT/2
        pedestal = pv.Cylinder(
            center=(0, 0, pedestal_z),
            direction=(0, 0, 1),
            radius=D.PEDESTAL_RADIUS,
            height=D.PEDESTAL_HEIGHT
        )
        self._add_mesh(plotter, pedestal, **M, name='pedestal')
        
        # Pedestal base flange (wider ring at bottom)
        base_flange = pv.Cylinder(
            center=(0, 0, D.BASE_PLATE_THICK + 4),
            direction=(0, 0, 1),
            radius=D.PEDESTAL_RADIUS + 5,
            height=8
        )
        self._add_mesh(plotter, base_flange, **M)
        
        # === Top Flange with Bearing Interface ===
        flange_z = D.BASE_PLATE_THICK + D.PEDESTAL_HEIGHT + D.PEDESTAL_FLANGE_H/2
        top_flange = pv.Cylinder(
            center=(0, 0, flange_z),
            direction=(0, 0, 1),
            radius=D.PEDESTAL_FLANGE_R,
            height=D.PEDESTAL_FLANGE_H
        )
        self._add_mesh(plotter, top_flange, **M, name='bearing_flange')
        
        # Bearing ring (decorative groove)
        bearing_ring = pv.Cylinder(
            center=(0, 0, flange_z + 2),
            direction=(0, 0, 1),
            radius=D.PEDESTAL_FLANGE_R + 2,
            height=3
        )
        self._add_mesh(plotter, bearing_ring, **Materials.STEEL)
        
        # Decorative holes on pedestal (2 rows of 4)
        for angle_deg in [0, 90, 180, 270]:
            angle_rad = np.deg2rad(angle_deg)
            x = D.PEDESTAL_RADIUS * 0.75 * np.cos(angle_rad)
            y = D.PEDESTAL_RADIUS * 0.75 * np.sin(angle_rad)
            
            for z_offset in [28, 38]:
                z = D.BASE_PLATE_THICK + z_offset
                hole = pv.Cylinder(
                    center=(x, y, z),
                    direction=(np.cos(angle_rad), np.sin(angle_rad), 0),
                    radius=2.5,
                    height=6
                )
                self._add_mesh(plotter, hole, **Materials.VOID)
        
        return self.actors
    
    def _add_mesh(self, plotter, mesh, **kwargs):
        """Helper to add mesh and track for transforms."""
        self.original_meshes.append(mesh.copy())
        name = kwargs.pop('name', None)
        actor = plotter.add_mesh(mesh, name=name, **kwargs)
        self.actors.append(actor)
        return actor


# =============================================================================
# ELEVATION YOKE ASSEMBLY
# =============================================================================

class ElevationYokeAssembly:
    """
    U-shaped yoke fork - machined aluminum with motor mounts.
    
    Components:
    - Horizontal base bar (sits on bearing flange)
    - Two vertical arms (Y = ±YOKE_ARM_SPACING)
    - Rounded tops on arms
    - Motor/encoder housings mounted on outer faces of arms
    - Decorative bolt holes on arm faces
    
    CRITICAL: Motors are attached to YOKE ARMS, not the barrel.
    This ensures collision-free 360° barrel rotation.
    """
    
    def __init__(self):
        self.actors = []
        self.original_meshes = []
    
    def build(self, plotter: pv.Plotter) -> List:
        """Construct yoke with integrated motor mounts."""
        D = Dimensions
        M = Materials.MACHINED_ALUMINUM
        
        base_z = D.YOKE_BASE_Z
        
        # === Yoke Base Bar (Horizontal) ===
        yoke_base = pv.Box(bounds=[
            -D.YOKE_ARM_THICK_X/2, D.YOKE_ARM_THICK_X/2,
            -D.YOKE_ARM_OUTER, D.YOKE_ARM_OUTER,
            base_z, base_z + D.YOKE_BASE_THICK
        ])
        self._add_mesh(plotter, yoke_base, **M, name='yoke_base')
        
        # === Vertical Arms with Motor Mounts ===
        for y_sign in [-1, 1]:
            arm_y = y_sign * D.YOKE_ARM_SPACING
            
            # Main arm body
            arm = pv.Box(bounds=[
                -D.YOKE_ARM_THICK_X/2, D.YOKE_ARM_THICK_X/2,
                arm_y - D.YOKE_ARM_THICK_Y/2, arm_y + D.YOKE_ARM_THICK_Y/2,
                base_z, base_z + D.YOKE_ARM_HEIGHT
            ])
            self._add_mesh(plotter, arm, **M)
            
            # Rounded top cap
            cap_z = base_z + D.YOKE_ARM_HEIGHT
            cap = pv.Cylinder(
                center=(0, arm_y, cap_z),
                direction=(0, 1, 0),
                radius=D.YOKE_ARM_THICK_X/2,
                height=D.YOKE_ARM_THICK_Y
            )
            self._add_mesh(plotter, cap, **M)
            
            # Decorative bolt holes on front face (4 per arm)
            for z_offset in [20, 35, 50, 65]:
                z = base_z + z_offset
                hole = pv.Cylinder(
                    center=(D.YOKE_ARM_THICK_X/2 + 0.5, arm_y, z),
                    direction=(1, 0, 0),
                    radius=2.5,
                    height=4
                )
                self._add_mesh(plotter, hole, **Materials.VOID)
            
            # === Motor/Encoder Housing (mounted on outer face of arm) ===
            # Motor body - cylindrical, centered at elevation pivot height
            motor_y = arm_y + y_sign * (D.YOKE_ARM_THICK_Y/2 + D.MOTOR_LENGTH/2)
            motor = pv.Cylinder(
                center=(0, motor_y, D.ELEVATION_PIVOT_Z),
                direction=(0, y_sign, 0),
                radius=D.MOTOR_RADIUS,
                height=D.MOTOR_LENGTH
            )
            self._add_mesh(plotter, motor, **Materials.BLACK_ANODIZED,
                          name=f'motor_{"L" if y_sign < 0 else "R"}')
            
            # Motor mounting flange (where it meets the arm)
            flange_y = arm_y + y_sign * D.YOKE_ARM_THICK_Y/2
            motor_flange = pv.Cylinder(
                center=(0, flange_y + y_sign * 2, D.ELEVATION_PIVOT_Z),
                direction=(0, y_sign, 0),
                radius=D.MOTOR_RADIUS + 4,
                height=4
            )
            self._add_mesh(plotter, motor_flange, **Materials.BLACK_ANODIZED)
            
            # Motor end cap (encoder housing)
            cap_y = motor_y + y_sign * D.MOTOR_LENGTH/2
            end_cap = pv.Cylinder(
                center=(0, cap_y + y_sign * D.MOTOR_CAP_LENGTH/2, D.ELEVATION_PIVOT_Z),
                direction=(0, y_sign, 0),
                radius=D.MOTOR_CAP_RADIUS,
                height=D.MOTOR_CAP_LENGTH
            )
            self._add_mesh(plotter, end_cap, **Materials.BLACK_PLASTIC)
            
            # Encoder cable connector (small cylinder)
            connector = pv.Cylinder(
                center=(0, cap_y + y_sign * (D.MOTOR_CAP_LENGTH + 3), D.ELEVATION_PIVOT_Z),
                direction=(0, y_sign, 0),
                radius=5,
                height=4
            )
            self._add_mesh(plotter, connector, **Materials.BLACK_PLASTIC)
        
        return self.actors
    
    def _add_mesh(self, plotter, mesh, **kwargs):
        """Helper to add mesh and track for transforms."""
        self.original_meshes.append(mesh.copy())
        name = kwargs.pop('name', None)
        actor = plotter.add_mesh(mesh, name=name, **kwargs)
        self.actors.append(actor)
        return actor


# =============================================================================
# ELEVATION BARREL ASSEMBLY
# =============================================================================

class ElevationBarrelAssembly:
    """
    Telescope barrel - black anodized aluminum.
    
    Components:
    - Main cylindrical barrel (vertical, along Z-axis)
    - Front lens aperture with retaining ring
    - Rear cap with heat sink fins
    - Bearing interfaces on sides (rotate in yoke arm holes)
    
    COLLISION AVOIDANCE:
    Barrel radius (32mm) < Yoke arm inner face (42.5mm)
    Clearance: 10.5mm on each side, total 21mm gap for free rotation.
    """
    
    def __init__(self):
        self.actors = []
        self.original_meshes = []
        self.pivot_z = Dimensions.ELEVATION_PIVOT_Z
        self.elevation_pivot = np.array([0, 0, self.pivot_z])
    
    def build(self, plotter: pv.Plotter) -> List:
        """Construct telescope barrel assembly."""
        D = Dimensions
        pivot_z = self.pivot_z
        
        # === Main Barrel Body ===
        barrel = pv.Cylinder(
            center=(0, 0, pivot_z),
            direction=(0, 0, 1),
            radius=D.BARREL_RADIUS,
            height=D.BARREL_LENGTH
        )
        self._add_mesh(plotter, barrel, **Materials.BLACK_ANODIZED, 
                      name='barrel')
        
        # Barrel end rings (decorative machining)
        for z_sign in [-1, 1]:
            z = pivot_z + z_sign * D.BARREL_LENGTH/2
            ring = pv.Cylinder(
                center=(0, 0, z - z_sign * 3),
                direction=(0, 0, 1),
                radius=D.BARREL_RADIUS + 2,
                height=6
            )
            self._add_mesh(plotter, ring, **Materials.BLACK_ANODIZED)
        
        # === Front Lens Aperture (Top) ===
        lens_z = pivot_z + D.BARREL_LENGTH/2
        
        # Lens glass (foggy white)
        lens = pv.Disc(
            center=(0, 0, lens_z - 3),
            inner=0,
            outer=D.BARREL_RADIUS - 5,
            normal=(0, 0, 1),
            c_res=64
        )
        self.original_meshes.append(lens.copy())
        actor = plotter.add_mesh(
            lens,
            color='#E8E8E8',
            opacity=0.45,
            specular=0.95,
            specular_power=80,
            pbr=True,
            name='lens'
        )
        self.actors.append(actor)
        
        # Lens retaining ring
        lens_ring = pv.Cylinder(
            center=(0, 0, lens_z - 2),
            direction=(0, 0, 1),
            radius=D.BARREL_RADIUS + 3,
            height=5
        )
        self._add_mesh(plotter, lens_ring, **Materials.BLACK_ANODIZED)
        
        # Lens inner bevel
        inner_ring = pv.Cylinder(
            center=(0, 0, lens_z - 4),
            direction=(0, 0, 1),
            radius=D.BARREL_RADIUS - 3,
            height=3
        )
        self._add_mesh(plotter, inner_ring, color="#F07F7F", metallic=0.5,
                      roughness=0.6, pbr=True)
        
        # === Rear Cap (Bottom) ===
        rear_z = pivot_z - D.BARREL_LENGTH/2
        
        rear_cap = pv.Disc(
            center=(0, 0, rear_z + 3),
            inner=0,
            outer=D.BARREL_RADIUS - 4,
            normal=(0, 0, -1),
            c_res=64
        )
        self._add_mesh(plotter, rear_cap, color="#DF8F8F", metallic=0.3,
                      roughness=0.7, pbr=True)
        
        # === Bearing Stubs (interface with yoke arm bearings) ===
        for y_sign in [-1, 1]:
            # These are short cylinders that align with the motor axes
            stub_y = y_sign * D.YOKE_ARM_INNER
            stub = pv.Cylinder(
                center=(0, stub_y - y_sign * 5, pivot_z),
                direction=(0, y_sign, 0),
                radius=8,
                height=12
            )
            self._add_mesh(plotter, stub, **Materials.STEEL)
        
        return self.actors
    
    def _add_mesh(self, plotter, mesh, **kwargs):
        """Helper to add mesh and track for transforms."""
        self.original_meshes.append(mesh.copy())
        name = kwargs.pop('name', None)
        actor = plotter.add_mesh(mesh, name=name, **kwargs)
        self.actors.append(actor)
        return actor


# =============================================================================
# GIMBAL MODEL (MAIN CLASS)
# =============================================================================

class GimbalModel:
    """
    Complete gimbal digital twin - high-fidelity CAD replica.
    
    Kinematic Chain:
    1. Base Assembly (FIXED) - Does not move
    2. Yoke Assembly (Azimuth) - Rotates about Z-axis, carries motors
    3. Barrel Assembly (Elevation) - Rotates about Y-axis within yoke
    
    Collision Avoidance:
    - Barrel radius: 32mm
    - Yoke arm inner face: 42.5mm from center
    - Clearance: 10.5mm per side (21mm total gap)
    - Result: Full 360° elevation rotation with zero collision
    
    Usage:
        plotter = pv.Plotter()
        gimbal = GimbalModel(plotter)
        gimbal.update_pose(az_deg=45, el_deg=30)
    """
    
    def __init__(self, plotter: pv.Plotter):
        """
        Build complete gimbal assembly.
        
        Parameters
        ----------
        plotter : pv.Plotter
            PyVista plotter instance for rendering
        """
        self.plotter = plotter
        
        # Current pose (radians internally)
        self.az_angle = 0.0
        self.el_angle = 0.0
        
        # Instantiate assemblies
        self.base = AzimuthBaseAssembly()
        self.yoke = ElevationYokeAssembly()
        self.barrel = ElevationBarrelAssembly()
        
        # Build geometry
        self.base_actors = self.base.build(plotter)
        self.yoke_actors = self.yoke.build(plotter)
        self.barrel_actors = self.barrel.build(plotter)
        
        # For backward compatibility - expose housing pivot
        self.housing = self.barrel
        
        # Cache original meshes for efficient transforms
        self._cache_original_geometry()
    
    def _cache_original_geometry(self):
        """Store original mesh points for rotation calculations."""
        self.original_base = [
            actor.GetMapper().GetInput().copy() for actor in self.base_actors
        ]
        self.original_yoke = [
            actor.GetMapper().GetInput().copy() for actor in self.yoke_actors
        ]
        self.original_barrel = [
            actor.GetMapper().GetInput().copy() for actor in self.barrel_actors
        ]
        # Backward compatibility
        self.original_housing = self.original_barrel
        self.housing_actors = self.barrel_actors
    
    def update_pose(self, az_deg: float, el_deg: float):
        """
        Update gimbal orientation with proper kinematic chain.
        
        Parameters
        ----------
        az_deg : float
            Azimuth angle in degrees (rotation about Z-axis)
        el_deg : float
            Elevation angle in degrees (rotation about Y-axis)
        
        Transform Chain:
        - Base: Fixed (no transform)
        - Yoke: R_z(azimuth) about origin
        - Barrel: R_y(elevation) about pivot, then R_z(azimuth) about origin
        """
        self.az_angle = np.deg2rad(az_deg)
        self.el_angle = np.deg2rad(el_deg)
        
        # Rotation matrices
        R_az = R.from_euler('z', self.az_angle).as_matrix()
        R_el = R.from_euler('y', self.el_angle).as_matrix()
        
        # Elevation pivot point
        pivot = self.barrel.elevation_pivot
        
        # === Base: Fixed ===
        # (No transform needed)
        
        # === Yoke: Azimuth rotation only ===
        for i, actor in enumerate(self.yoke_actors):
            mesh = actor.GetMapper().GetInput()
            original = self.original_yoke[i]
            mesh.points = original.points @ R_az.T
            mesh.Modified()
        
        # === Barrel: Elevation about Y, then Azimuth about Z ===
        for i, actor in enumerate(self.barrel_actors):
            mesh = actor.GetMapper().GetInput()
            original = self.original_barrel[i]
            
            # Translate to pivot origin
            pts = original.points - pivot
            
            # Apply elevation rotation (about Y)
            pts = pts @ R_el.T
            
            # Apply azimuth rotation (about Z)
            pts = pts @ R_az.T
            
            # Translate back (pivot rotates with azimuth)
            mesh.points = pts + (pivot @ R_az.T)
            mesh.Modified()
    
    def get_current_pose(self) -> Tuple[float, float]:
        """Return current (azimuth_deg, elevation_deg)."""
        return (np.rad2deg(self.az_angle), np.rad2deg(self.el_angle))
    
    def reset_pose(self):
        """Reset gimbal to home position (0, 0)."""
        self.update_pose(0.0, 0.0)
