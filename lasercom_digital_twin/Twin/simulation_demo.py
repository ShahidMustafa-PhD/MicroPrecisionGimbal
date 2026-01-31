"""
Simulation Demo - Trajectory Playback and Visualization
========================================================
High-frequency update loop for gimbal tracking demonstrations.

This module handles:
- PyVista plotter setup with PBR lighting
- Environment maps and studio lighting
- Trajectory generation and playback
- Real-time UI overlays and metrics

Author: Expert Python Software Architect
Date: January 31, 2026
"""

import numpy as np
import pyvista as pv
from pathlib import Path
import sys

# Import the digital twin model
from gimbal_model import GimbalModel


class StudioEnvironment:
    """
    Professional studio lighting and environment setup.
    
    Features:
    - PBR (Physically Based Rendering) lighting
    - Environment maps for realistic reflections
    - Shadow casting for depth perception
    """
    
    def __init__(self, plotter: pv.Plotter):
        """
        Initialize studio environment.
        
        Parameters:
        -----------
        plotter : pv.Plotter
            PyVista plotter to configure
        """
        self.plotter = plotter
        
    def setup_lighting(self):
        """Configure professional studio lighting."""
        # Clear default lights
        self.plotter.remove_all_lights()
        
        # === Key Light (main illumination) ===
        key_light = pv.Light(
            position=(400, -300, 300),
            focal_point=(0, 0, 50),
            color='white',
            intensity=0.9,
            positional=True,
            cone_angle=60,
            exponent=20
        )
        self.plotter.add_light(key_light)
        
        # === Fill Light (soften shadows) ===
        fill_light = pv.Light(
            position=(-300, -200, 200),
            focal_point=(0, 0, 50),
            color='#E8F4F8',  # Cool fill
            intensity=0.4,
            positional=True
        )
        self.plotter.add_light(fill_light)
        
        # === Rim Light (edge definition) ===
        rim_light = pv.Light(
            position=(-200, 300, 250),
            focal_point=(0, 0, 50),
            color='#FFF8E1',  # Warm rim
            intensity=0.5,
            positional=True
        )
        self.plotter.add_light(rim_light)
        
        # === Ambient Light (base illumination) ===
        ambient = pv.Light(
            light_type='headlight',
            intensity=0.2
        )
        self.plotter.add_light(ambient)
        
        # Enable shadows
        self.plotter.enable_shadows()
    
    def setup_environment(self):
        """Configure scene environment."""
        # Professional background
        self.plotter.set_background('#F5F5F5')  # Light gray studio background
        
        # Enable anti-aliasing for smooth edges
        self.plotter.enable_anti_aliasing('msaa')
        
        # Add ground plane for context
        ground = pv.Plane(
            center=(0, 0, -100),
            direction=(0, 0, 1),
            i_size=600,
            j_size=600
        )
        self.plotter.add_mesh(
            ground,
            color='#E0E0E0',
            opacity=0.3,
            pbr=True,
            metallic=0.0,
            roughness=0.8,
            name='ground_plane'
        )


class TrajectoryGenerator:
    """
    Generate realistic gimbal tracking trajectories.
    
    Patterns:
    - Sinusoidal scanning
    - Circular conical scan
    - Step-and-dwell acquisition
    - Spiral search pattern
    """
    
    @staticmethod
    def sinusoidal_scan(t: float, az_amp: float = 30.0, el_amp: float = 15.0,
                        az_freq: float = 0.3, el_freq: float = 0.5,
                        el_offset: float = 20.0) -> tuple:
        """
        Sinusoidal scanning pattern (Lissajous).
        
        Parameters:
        -----------
        t : float
            Time in seconds
        az_amp : float
            Azimuth amplitude (degrees)
        el_amp : float
            Elevation amplitude (degrees)
        az_freq : float
            Azimuth frequency (Hz)
        el_freq : float
            Elevation frequency (Hz)
        el_offset : float
            Elevation bias (degrees)
            
        Returns:
        --------
        tuple : (azimuth_deg, elevation_deg)
        """
        az = az_amp * np.sin(2 * np.pi * az_freq * t)
        el = el_amp * np.sin(2 * np.pi * el_freq * t) + el_offset
        return az, el
    
    @staticmethod
    def conical_scan(t: float, cone_angle: float = 20.0, 
                     scan_rate: float = 1.0, el_bias: float = 30.0) -> tuple:
        """
        Conical scan pattern (circular).
        
        Parameters:
        -----------
        t : float
            Time in seconds
        cone_angle : float
            Cone half-angle (degrees)
        scan_rate : float
            Scan frequency (Hz)
        el_bias : float
            Elevation boresight (degrees)
            
        Returns:
        --------
        tuple : (azimuth_deg, elevation_deg)
        """
        theta = 2 * np.pi * scan_rate * t
        az = cone_angle * np.cos(theta)
        el = el_bias + cone_angle * np.sin(theta)
        return az, el
    
    @staticmethod
    def step_dwell(t: float, step_size: float = 10.0, 
                   dwell_time: float = 2.0) -> tuple:
        """
        Step-and-dwell acquisition pattern.
        
        Parameters:
        -----------
        t : float
            Time in seconds
        step_size : float
            Step increment (degrees)
        dwell_time : float
            Dwell duration per step (seconds)
            
        Returns:
        --------
        tuple : (azimuth_deg, elevation_deg)
        """
        step_index = int(t / dwell_time)
        az = (step_index % 5) * step_size - 20
        el = ((step_index // 5) % 3) * step_size + 10
        return az, el
    
    @staticmethod
    def spiral_search(t: float, radius_rate: float = 5.0,
                      angular_rate: float = 2.0) -> tuple:
        """
        Spiral search pattern.
        
        Parameters:
        -----------
        t : float
            Time in seconds
        radius_rate : float
            Radius growth rate (deg/s)
        angular_rate : float
            Angular velocity (rad/s)
            
        Returns:
        --------
        tuple : (azimuth_deg, elevation_deg)
        """
        r = radius_rate * t
        theta = angular_rate * t
        az = r * np.cos(theta)
        el = 20 + r * np.sin(theta)
        return az, el


class SimulationOrchestrator:
    """
    Main simulation controller.
    
    Responsibilities:
    - Initialize plotter and environment
    - Instantiate gimbal model
    - Run high-frequency update loop
    - Render UI overlays
    """
    
    def __init__(self, window_size: tuple = (1920, 1080), 
                 offline_mode: bool = False):
        """
        Initialize simulation orchestrator.
        
        Parameters:
        -----------
        window_size : tuple
            Window dimensions (width, height)
        offline_mode : bool
            If True, render to GIF/video instead of interactive window
        """
        self.window_size = window_size
        self.offline_mode = offline_mode
        
        # Initialize plotter
        self.plotter = pv.Plotter(
            window_size=window_size,
            notebook=False,
            off_screen=offline_mode
        )
        
        # Setup environment
        self.environment = StudioEnvironment(self.plotter)
        self.environment.setup_lighting()
        self.environment.setup_environment()
        
        # Instantiate gimbal model
        print("Building digital twin model...")
        self.gimbal = GimbalModel(self.plotter)
        
        # Trajectory generator
        self.trajectory = TrajectoryGenerator()
        
        # UI elements
        self.text_actors = []
        
    def setup_camera(self):
        """Configure camera view."""
        self.plotter.camera_position = [
            (350, -280, 180),  # Camera location
            (0, 0, 40),         # Focal point
            (0, 0, 1)           # Up vector
        ]
        self.plotter.camera.zoom(1.3)
    
    def add_ui_overlays(self, t: float, az: float, el: float, frame: int, total_frames: int):
        """
        Add real-time UI overlays.
        
        Parameters:
        -----------
        t : float
            Current time (seconds)
        az : float
            Azimuth angle (degrees)
        el : float
            Elevation angle (degrees)
        frame : int
            Current frame number
        total_frames : int
            Total frame count
        """
        # Main status text
        status_text = (
            f"Time: {t:.2f}s | Frame: {frame}/{total_frames}\n"
            f"Azimuth:   {az:+8.3f}°\n"
            f"Elevation: {el:+8.3f}°\n"
            f"Status: TRACKING"
        )
        self.plotter.add_text(
            status_text,
            position='upper_left',
            font_size=12,
            color='black',
            font='courier',
            name='status'
        )
        
        # Performance metrics
        metrics_text = (
            f"═══ SIMULATION METRICS ═══\n"
            f"Update Rate:   {1.0/0.033:.1f} Hz\n"
            f"Gimbal Type:   2-Axis\n"
            f"Control Mode:  Position\n"
            f"Coord Frame:   World (ECI)\n"
            f"═══════════════════════════"
        )
        self.plotter.add_text(
            metrics_text,
            position='lower_right',
            font_size=10,
            color='darkgreen',
            font='courier',
            name='metrics'
        )
        
        # Title banner
        self.plotter.add_text(
            "LaserCom Digital Twin - High-Fidelity Gimbal Visualization",
            position='upper_edge',
            font_size=14,
            color='#2C3E50',
            font='arial',
            name='title'
        )
    
    def run_simulation(self, duration: float = 20.0, 
                       trajectory_type: str = 'sinusoidal',
                       save_animation: bool = True):
        """
        Execute simulation loop.
        
        Parameters:
        -----------
        duration : float
            Simulation duration (seconds)
        trajectory_type : str
            Trajectory pattern: 'sinusoidal', 'conical', 'step_dwell', 'spiral'
        save_animation : bool
            If True, save animation to GIF
        """
        # Simulation parameters
        dt = 0.033  # ~30 FPS
        time_steps = np.arange(0, duration, dt)
        total_frames = len(time_steps)
        
        # Select trajectory generator
        trajectory_funcs = {
            'sinusoidal': self.trajectory.sinusoidal_scan,
            'conical': self.trajectory.conical_scan,
            'step_dwell': self.trajectory.step_dwell,
            'spiral': self.trajectory.spiral_search
        }
        get_trajectory = trajectory_funcs.get(trajectory_type, 
                                              self.trajectory.sinusoidal_scan)
        
        # Setup camera
        self.setup_camera()
        
        # Optional: Open GIF writer
        if save_animation and self.offline_mode:
            output_path = Path(__file__).parent / f"gimbal_{trajectory_type}.gif"
            self.plotter.open_gif(str(output_path))
            print(f"Recording animation to: {output_path}")
        
        print(f"Starting simulation ({trajectory_type} trajectory)...")
        print(f"Duration: {duration}s | Frames: {total_frames} | Update Rate: {1/dt:.1f} Hz")
        
        # === Main Simulation Loop ===
        for i, t in enumerate(time_steps):
            # Compute target angles from trajectory
            az, el = get_trajectory(t)
            
            # Update gimbal pose
            self.gimbal.update_pose(az, el)
            
            # Update UI overlays
            self.add_ui_overlays(t, az, el, i+1, total_frames)
            
            # Render frame
            if self.offline_mode:
                self.plotter.write_frame()
            else:
                self.plotter.render()
            
            # Progress indicator
            if i % 30 == 0:
                print(f"  Frame {i+1}/{total_frames} | "
                      f"Az={az:+7.2f}° El={el:+7.2f}°")
        
        # Close plotter
        if self.offline_mode and save_animation:
            self.plotter.close()
            print(f"\nAnimation saved: {output_path}")
        else:
            print("\nSimulation complete. Close window to exit.")
            self.plotter.show()


def main():
    """Main entry point for simulation demo."""
    
    # === Configuration ===
    DURATION = 1.0  # seconds
    TRAJECTORY = 'sinusoidal'  # Options: 'sinusoidal', 'conical', 'step_dwell', 'spiral'
    SAVE_ANIMATION = True
    OFFLINE_MODE = True  # Set to False for interactive window
    
    # === Run Simulation ===
    print("═" * 60)
    print("  LaserCom Digital Twin - Gimbal Visualization")
    print("  High-Fidelity PyVista Simulation")
    print("═" * 60)
    
    orchestrator = SimulationOrchestrator(
        window_size=(1920, 1080),
        offline_mode=OFFLINE_MODE
    )
    
    orchestrator.run_simulation(
        duration=DURATION,
        trajectory_type=TRAJECTORY,
        save_animation=SAVE_ANIMATION
    )
    
    print("\n✓ Simulation execution complete.")


if __name__ == "__main__":
    main()
