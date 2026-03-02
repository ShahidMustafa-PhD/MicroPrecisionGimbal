"""
PyVista High-Fidelity Gimbal Simulation
========================================
Aerospace-grade 3D visualization of a 2-axis laser communication gimbal.

Features:
- Hierarchical kinematic chain (Base → Yoke → Optical Housing)
- PBR materials (metallic aluminum, matte plastics, transparent optics)
- Real-time azimuth/elevation tracking
- Laser beam pointing simulation
- Internal components (motors, encoders, MEMS gyro)

Author: Expert Robotics Simulation Engineer
Date: January 31, 2026
"""

import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation as R


class GimbalSim:
    """
    High-fidelity gimbal simulation with hierarchical kinematics.
    
    Coordinate System:
    - X: Forward (laser beam direction)
    - Y: Right
    - Z: Up (azimuth rotation axis)
    """
    
    def __init__(self, plotter):
        """Initialize gimbal simulation with PyVista plotter."""
        self.plotter = plotter
        
        # Actor storage (hierarchical groups)
        self.base_actors = []
        self.yoke_actors = []
        self.housing_actors = []
        self.cable_actors = []
        self.laser_actor = None
        
        # Current gimbal state (radians)
        self.az_angle = 0.0
        self.el_angle = 0.0
        
        # Build the gimbal
        self._build_azimuth_base()
        self._build_elevation_yoke()
        self._build_optical_housing()
        self._build_cables()
        self._build_laser_beam()
        
    def _build_azimuth_base(self):
        """Construct azimuth base assembly with internal components."""
        
        # === Main Base Housing ===
        base_housing = pv.Cylinder(
            center=(0, 0, -30),
            direction=(0, 0, 1),
            radius=50,
            height=60
        )
        actor = self.plotter.add_mesh(
            base_housing,
            color='silver',
            metallic=1.0,
            roughness=0.3,
            pbr=True,
            name='base_housing'
        )
        self.base_actors.append(actor)
        
        # Base mounting plate (bottom)
        mount_plate = pv.Cylinder(
            center=(0, 0, -65),
            direction=(0, 0, 1),
            radius=60,
            height=10
        )
        actor = self.plotter.add_mesh(
            mount_plate,
            color='#2C3E50',  # Dark steel
            metallic=0.8,
            roughness=0.4,
            pbr=True,
            name='mount_plate'
        )
        self.base_actors.append(actor)
        
        # === Internal Azimuth Motor ===
        motor_core = pv.Cylinder(
            center=(0, 0, -20),
            direction=(0, 0, 1),
            radius=25,
            height=35
        )
        actor = self.plotter.add_mesh(
            motor_core,
            color='#1C1C1C',  # Matte black
            metallic=0.2,
            roughness=0.8,
            pbr=True,
            name='az_motor'
        )
        self.base_actors.append(actor)
        
        # Motor end caps
        for z_offset in [-37.5, -2.5]:
            cap = pv.Disc(
                center=(0, 0, z_offset),
                inner=0,
                outer=25,
                normal=(0, 0, 1),
                c_res=32
            )
            actor = self.plotter.add_mesh(
                cap,
                color='#0A0A0A',
                metallic=0.3,
                roughness=0.7,
                pbr=True
            )
            self.base_actors.append(actor)
        
        # === Base Encoder (thin disc near top) ===
        encoder = pv.Cylinder(
            center=(0, 0, 5),
            direction=(0, 0, 1),
            radius=30,
            height=3
        )
        actor = self.plotter.add_mesh(
            encoder,
            color='#8B4513',  # Circuit board brown
            metallic=0.4,
            roughness=0.6,
            pbr=True,
            name='base_encoder'
        )
        self.base_actors.append(actor)
        
        # === MEMS Rate Gyro (small rectangular housing) ===
        mems_housing = pv.Box(
            bounds=[-15, -5, 25, 35, 10, 20]
        )
        actor = self.plotter.add_mesh(
            mems_housing,
            color='#34495E',  # Dark gray
            metallic=0.3,
            roughness=0.7,
            pbr=True,
            name='mems_gyro'
        )
        self.base_actors.append(actor)
        
        # MEMS label (text annotation)
        label_point = np.array([-10, 30, 15])
        self.plotter.add_point_labels(
            [label_point],
            ['MEMS'],
            font_size=10,
            text_color='white',
            point_size=0,
            name='mems_label'
        )
        
    def _build_elevation_yoke(self):
        """Construct U-shaped yoke frame (elevation gimbal)."""
        
        # === Yoke Base (horizontal bar) ===
        yoke_base = pv.Box(bounds=[-80, 80, -20, 20, 0, 10])
        actor = self.plotter.add_mesh(
            yoke_base,
            color='silver',
            metallic=1.0,
            roughness=0.3,
            pbr=True,
            name='yoke_base'
        )
        self.yoke_actors.append(actor)
        
        # === Yoke Vertical Arms (left and right) ===
        for x_pos in [-70, 70]:
            arm = pv.Box(bounds=[x_pos-10, x_pos+10, -20, 20, 10, 120])
            actor = self.plotter.add_mesh(
                arm,
                color='silver',
                metallic=1.0,
                roughness=0.3,
                pbr=True
            )
            self.yoke_actors.append(actor)
        
        # === Elevation Pivot Bearings (cylindrical journals) ===
        for x_pos in [-70, 70]:
            bearing = pv.Cylinder(
                center=(x_pos, 0, 65),
                direction=(1, 0, 0),
                radius=12,
                height=30
            )
            actor = self.plotter.add_mesh(
                bearing,
                color='#4A4A4A',  # Dark gray
                metallic=0.6,
                roughness=0.4,
                pbr=True
            )
            self.yoke_actors.append(actor)
        
        # === Elevation Motor Housing (attached to left arm) ===
        el_motor = pv.Box(bounds=[-90, -70, -15, 15, 60, 80])
        actor = self.plotter.add_mesh(
            el_motor,
            color='#1C1C1C',  # Matte black
            metallic=0.2,
            roughness=0.8,
            pbr=True,
            name='el_motor'
        )
        self.yoke_actors.append(actor)
        
    def _build_optical_housing(self):
        """Construct telescope barrel with internal optics."""
        
        # === Main Telescope Barrel (hollow tube) ===
        outer_barrel = pv.Cylinder(
            center=(0, 0, 65),
            direction=(1, 0, 0),
            radius=40,
            height=180
        )
        actor = self.plotter.add_mesh(
            outer_barrel,
            color='#2C3E50',  # Dark blue-gray
            metallic=0.4,
            roughness=0.5,
            pbr=True,
            name='telescope_barrel'
        )
        self.housing_actors.append(actor)
        
        # Inner cavity (make it look hollow)
        inner_cavity = pv.Cylinder(
            center=(0, 0, 65),
            direction=(1, 0, 0),
            radius=35,
            height=175
        )
        actor = self.plotter.add_mesh(
            inner_cavity,
            color='#0A0A0A',  # Black interior
            metallic=0.0,
            roughness=1.0,
            pbr=True
        )
        self.housing_actors.append(actor)
        
        # === Front Lens (transparent cyan) ===
        front_lens = pv.Disc(
            center=(90, 0, 65),
            inner=0,
            outer=35,
            normal=(1, 0, 0),
            c_res=64
        )
        actor = self.plotter.add_mesh(
            front_lens,
            color='cyan',
            opacity=0.4,
            specular=1.0,
            specular_power=100,
            pbr=True,
            name='front_lens'
        )
        self.housing_actors.append(actor)
        
        # Lens mount (retaining ring)
        lens_mount = pv.Tube(
            pointa=(88, 0, 65),
            pointb=(92, 0, 65),
            resolution=64,
            radius=38,
            inner_radius=36
        )
        actor = self.plotter.add_mesh(
            lens_mount,
            color='#34495E',
            metallic=0.8,
            roughness=0.3,
            pbr=True
        )
        self.housing_actors.append(actor)
        
        # === Primary Mirror (rear) ===
        mirror = pv.Disc(
            center=(-85, 0, 65),
            inner=0,
            outer=30,
            normal=(-1, 0, 0),
            c_res=64
        )
        actor = self.plotter.add_mesh(
            mirror,
            color='gold',
            metallic=1.0,
            roughness=0.05,
            pbr=True,
            name='primary_mirror'
        )
        self.housing_actors.append(actor)
        
        # === Detector Housing (rear protrusion) ===
        detector_box = pv.Box(bounds=[-100, -85, -15, 15, 50, 80])
        actor = self.plotter.add_mesh(
            detector_box,
            color='#1C1C1C',
            metallic=0.2,
            roughness=0.8,
            pbr=True,
            name='detector'
        )
        self.housing_actors.append(actor)
        
    def _build_cables(self):
        """Build flexible cables connecting housing to yoke."""
        
        # Create natural-looking cable paths using splines
        for y_offset in [-25, 25]:
            # Define cable waypoints (rear of housing to yoke arm)
            points = np.array([
                [-85, y_offset, 65],      # Start: rear of housing
                [-75, y_offset*1.5, 75],  # Control point 1 (droop)
                [-65, y_offset*1.8, 85],  # Control point 2
                [-55, y_offset*1.5, 95],  # Control point 3
                [-45, y_offset*1.2, 100], # End: yoke connection
            ])
            
            # Create spline curve
            spline = pv.Spline(points, n_points=100)
            
            # Convert to tube
            cable = spline.tube(radius=2.5)
            actor = self.plotter.add_mesh(
                cable,
                color='#FF6B35',  # Orange cable jacket
                metallic=0.1,
                roughness=0.9,
                pbr=True
            )
            self.cable_actors.append(actor)
            
            # Cable connectors (small cylinders at ends)
            for point in [points[0], points[-1]]:
                connector = pv.Cylinder(
                    center=point,
                    direction=(0, 0, 1),
                    radius=4,
                    height=8
                )
                actor = self.plotter.add_mesh(
                    connector,
                    color='#4A4A4A',
                    metallic=0.5,
                    roughness=0.6,
                    pbr=True
                )
                self.cable_actors.append(actor)
    
    def _build_laser_beam(self):
        """Build incoming laser beam (emissive red)."""
        
        # Laser originates from far distance, points at gimbal
        laser_beam = pv.Cylinder(
            center=(500, 0, 65),  # 500mm away
            direction=(1, 0, 0),
            radius=1.5,
            height=1000
        )
        
        self.laser_actor = self.plotter.add_mesh(
            laser_beam,
            color='red',
            emissive=True,
            opacity=0.7,
            name='laser_beam'
        )
    
    def update(self, az_deg, el_deg):
        """
        Update gimbal orientation and redraw all components.
        
        Parameters:
        -----------
        az_deg : float
            Azimuth angle in degrees (rotation about Z-axis)
        el_deg : float
            Elevation angle in degrees (rotation about Y-axis)
        """
        # Convert to radians
        self.az_angle = np.deg2rad(az_deg)
        self.el_angle = np.deg2rad(el_deg)
        
        # === Base remains fixed (no transformation needed) ===
        
        # === Transform Yoke (azimuth rotation only) ===
        az_rot_matrix = R.from_euler('z', self.az_angle).as_matrix()
        
        for actor in self.yoke_actors:
            mesh = actor.GetMapper().GetInput()
            original_points = mesh.points.copy()
            
            # Rotate about Z-axis (world origin)
            rotated_points = original_points @ az_rot_matrix.T
            mesh.points = rotated_points
            mesh.Modified()
        
        # === Transform Optical Housing (azimuth + elevation) ===
        # First rotate by azimuth, then by elevation (in yoke's frame)
        
        # Elevation pivot point is at (0, 0, 65) in world frame
        # But needs to rotate in the yoke's reference frame
        el_pivot = np.array([0, 0, 65])
        
        for actor in self.housing_actors:
            mesh = actor.GetMapper().GetInput()
            original_points = mesh.points.copy()
            
            # Step 1: Translate to origin
            centered_points = original_points - el_pivot
            
            # Step 2: Apply elevation rotation (about Y-axis in yoke frame)
            el_rot_matrix = R.from_euler('y', self.el_angle).as_matrix()
            rotated_el = centered_points @ el_rot_matrix.T
            
            # Step 3: Apply azimuth rotation
            rotated_both = rotated_el @ az_rot_matrix.T
            
            # Step 4: Translate back
            final_points = rotated_both + el_pivot @ az_rot_matrix.T
            
            mesh.points = final_points
            mesh.Modified()
        
        # === Transform Cables (same as housing) ===
        for actor in self.cable_actors:
            mesh = actor.GetMapper().GetInput()
            original_points = mesh.points.copy()
            
            centered_points = original_points - el_pivot
            el_rot_matrix = R.from_euler('y', self.el_angle).as_matrix()
            rotated_el = centered_points @ el_rot_matrix.T
            rotated_both = rotated_el @ az_rot_matrix.T
            final_points = rotated_both + el_pivot @ az_rot_matrix.T
            
            mesh.points = final_points
            mesh.Modified()
        
        # === Laser beam remains fixed in world frame ===
        # (Could be updated to track gimbal if needed)
    
    def reset_geometry(self):
        """Reset all meshes to initial positions (call before update)."""
        # Remove and rebuild (simplified for this demo)
        # In production, store original geometries separately
        pass


def main():
    """Main simulation loop with tracking trajectory."""
    
    # === Initialize PyVista Plotter ===
    pl = pv.Plotter(
        window_size=[1920, 1080],
        notebook=False
    )
    
    # Set professional lighting
    pl.set_background('ghostwhite')
    pl.enable_shadows()
    
    # Add environment lighting
    light = pv.Light(
        position=(500, 500, 500),
        focal_point=(0, 0, 50),
        color='white',
        intensity=0.8
    )
    pl.add_light(light)
    
    # Add ambient light
    ambient = pv.Light(
        light_type='headlight',
        intensity=0.3
    )
    pl.add_light(ambient)
    
    # === Build Gimbal ===
    print("Building gimbal simulation...")
    gimbal = GimbalSim(pl)
    
    # === Set Camera View ===
    pl.camera_position = [
        (300, -250, 150),  # Camera location
        (0, 0, 50),         # Focal point
        (0, 0, 1)           # Up vector
    ]
    pl.camera.zoom(1.2)
    
    # === Simulation Parameters ===
    duration = 20.0  # seconds
    dt = 0.033       # ~30 FPS
    time_steps = np.arange(0, duration, dt)
    
    # === Trajectory Definition (Sinusoidal Tracking) ===
    def compute_target_angles(t):
        """
        Compute azimuth/elevation tracking trajectory.
        Simulates gimbal following a moving target.
        """
        # Azimuth: slow sinusoid
        az = 30 * np.sin(0.3 * t)
        
        # Elevation: faster sinusoid with offset
        el = 15 * np.sin(0.5 * t) + 20
        
        return az, el
    
    # === Render Initial Frame ===
    az_init, el_init = compute_target_angles(0)
    gimbal.update(az_init, el_init)
    
    # Add text overlay
    text_actor = pl.add_text(
        f"Az: {az_init:.1f}° | El: {el_init:.1f}°",
        position='upper_left',
        font_size=12,
        color='black'
    )
    
    # === Animation Loop ===
    print("Starting simulation...")
    print("Close the window to exit.")
    
    pl.open_gif("gimbal_tracking.gif")  # Optional: save animation
    
    for i, t in enumerate(time_steps):
        # Compute target angles
        az, el = compute_target_angles(t)
        
        # Rebuild gimbal at new orientation
        # (Note: This is a simplified approach. For production, use
        #  proper transform matrices without rebuilding meshes)
        pl.clear()
        gimbal = GimbalSim(pl)
        gimbal.update(az, el)
        
        # Update text
        text_actor = pl.add_text(
            f"Time: {t:.1f}s | Az: {az:.1f}° | El: {el:.1f}°\n"
            f"Tracking Mode: ACTIVE",
            position='upper_left',
            font_size=14,
            color='black',
            font='arial'
        )
        
        # Add performance metrics
        metrics_text = (
            f"Simulation Metrics\n"
            f"─────────────────\n"
            f"Azimuth:     {az:+7.2f}°\n"
            f"Elevation:   {el:+7.2f}°\n"
            f"Frame:       {i+1}/{len(time_steps)}\n"
            f"Status:      NOMINAL"
        )
        pl.add_text(
            metrics_text,
            position='lower_right',
            font_size=10,
            color='darkgreen',
            font='courier'
        )
        
        # Reset camera (keep same view)
        pl.camera_position = [
            (300, -250, 150),
            (0, 0, 50),
            (0, 0, 1)
        ]
        pl.camera.zoom(1.2)
        
        # Render frame
        pl.write_frame()
        
        # Update at ~30 FPS
        if i % 10 == 0:
            print(f"  Frame {i}/{len(time_steps)} | Az={az:.1f}° El={el:.1f}°")
    
    pl.close()
    print("\nSimulation complete!")
    print("Animation saved to: gimbal_tracking.gif")


if __name__ == "__main__":
    main()
