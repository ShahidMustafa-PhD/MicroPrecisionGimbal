"""
Real-Time MuJoCo Visualization for Digital Twin Debugging

This module provides interactive 3D visualization of the physics simulation
with engineering-focused debugging features for control system validation.

Key Features:
-------------
- Real-time display of 2-DOF gimbal mechanism
- Force/torque vector visualization for actuators and disturbances
- Joint limit and contact force indicators
- Configurable camera views and rendering options

Design Philosophy:
------------------
Prioritizes engineering insight over aesthetics. Every visual element serves
a specific debugging purpose (actuator saturation, mechanical compliance,
contact forces, etc.).
"""

import numpy as np
import mujoco
import mujoco.viewer
from typing import Optional, Dict, Any, Callable
import time
import threading


class MuJoCoVisualizer:
    """
    Real-time MuJoCo visualization with engineering debugging features.
    
    This class wraps the MuJoCo viewer and adds custom overlays for
    visualizing forces, torques, and system states during simulation.
    
    Usage:
    ------
    >>> visualizer = MuJoCoVisualizer(model, data)
    >>> visualizer.add_torque_arrow('motor_az', scale=10.0)
    >>> visualizer.add_joint_limit_indicator('gimbal_az')
    >>> 
    >>> # Launch viewer in separate thread
    >>> visualizer.launch(blocking=False)
    >>> 
    >>> # In simulation loop
    >>> while running:
    ...     mujoco.mj_step(model, data)
    ...     visualizer.update(data)
    ...     visualizer.render()
    >>> 
    >>> visualizer.close()
    """
    
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        title: str = "Digital Twin - MuJoCo Visualization",
        width: int = 1280,
        height: int = 720,
        show_forces: bool = True,
        show_contacts: bool = True,
        show_joint_limits: bool = True
    ):
        """
        Initialize MuJoCo visualizer with debugging features.
        
        Parameters
        ----------
        model : mujoco.MjModel
            MuJoCo model instance
        data : mujoco.MjData
            MuJoCo data instance
        title : str
            Window title
        width : int
            Window width in pixels
        height : int
            Window height in pixels
        show_forces : bool
            Display force/torque arrows
        show_contacts : bool
            Display contact forces at joint limits
        show_joint_limits : bool
            Display joint limit zones
        """
        self.model = model
        self.data = data
        self.title = title
        self.width = width
        self.height = height
        
        # Visualization flags
        self.show_forces = show_forces
        self.show_contacts = show_contacts
        self.show_joint_limits = show_joint_limits
        
        # Viewer handle (initialized on launch)
        self.viewer = None
        self.viewer_thread = None
        self.is_running = False
        
        # Custom visualization sites (for force arrows, etc.)
        self.force_arrows = {}  # {name: {'body': body_id, 'scale': float}}
        self.joint_indicators = {}  # {joint_name: {'limits': (min, max)}}
        
        # Performance tracking
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0.0
        
    def add_torque_arrow(
        self,
        actuator_name: str,
        scale: float = 1.0,
        color: tuple = (1.0, 0.0, 0.0, 0.8)
    ):
        """
        Add visual arrow representing actuator torque.
        
        This creates a visual indicator showing the direction and magnitude
        of the torque applied by a motor. Essential for debugging saturation
        and anti-windup behavior.
        
        Parameters
        ----------
        actuator_name : str
            Name of MuJoCo actuator
        scale : float
            Scaling factor for arrow length (length = scale * torque)
        color : tuple
            RGBA color for arrow (red by default for easy identification)
        """
        # Find actuator ID
        try:
            actuator_id = mujoco.mj_name2id(
                self.model, 
                mujoco.mjtObj.mjOBJ_ACTUATOR, 
                actuator_name
            )
        except:
            raise ValueError(f"Actuator '{actuator_name}' not found in model")
        
        # Get joint associated with actuator
        trnid = self.model.actuator_trnid[actuator_id]
        joint_id = trnid[0]  # First element is joint ID
        
        # Store arrow configuration
        self.force_arrows[actuator_name] = {
            'actuator_id': actuator_id,
            'joint_id': joint_id,
            'scale': scale,
            'color': color
        }
    
    def add_disturbance_arrow(
        self,
        body_name: str,
        scale: float = 1.0,
        color: tuple = (0.0, 0.5, 1.0, 0.8)
    ):
        """
        Add visual arrow representing external disturbance forces.
        
        Useful for visualizing wind forces, friction, or other external
        loads that affect pointing accuracy.
        
        Parameters
        ----------
        body_name : str
            Name of MuJoCo body receiving disturbance
        scale : float
            Scaling factor for arrow length
        color : tuple
            RGBA color for arrow (blue by default to distinguish from control)
        """
        try:
            body_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                body_name
            )
        except:
            raise ValueError(f"Body '{body_name}' not found in model")
        
        self.force_arrows[f"disturbance_{body_name}"] = {
            'body_id': body_id,
            'scale': scale,
            'color': color,
            'type': 'disturbance'
        }
    
    def add_joint_limit_indicator(
        self,
        joint_name: str,
        warning_zone: float = 0.1
    ):
        """
        Add visual indicator for joint limits.
        
        Colors the joint region based on proximity to limits:
        - Green: Normal operation
        - Yellow: Within warning_zone of limits
        - Red: At or beyond limits (mechanical stop engaged)
        
        Parameters
        ----------
        joint_name : str
            Name of MuJoCo joint
        warning_zone : float
            Fraction of range to mark as warning (0.1 = 10% from limits)
        """
        try:
            joint_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name
            )
        except:
            raise ValueError(f"Joint '{joint_name}' not found in model")
        
        # Get joint limits from model
        jnt_limited = self.model.jnt_limited[joint_id]
        if not jnt_limited:
            raise ValueError(f"Joint '{joint_name}' does not have limits defined")
        
        jnt_range = self.model.jnt_range[joint_id]
        
        self.joint_indicators[joint_name] = {
            'joint_id': joint_id,
            'limits': (jnt_range[0], jnt_range[1]),
            'warning_zone': warning_zone
        }
    
    def launch(self, blocking: bool = True):
        """
        Launch the MuJoCo viewer window.
        
        Parameters
        ----------
        blocking : bool
            If True, blocks until viewer is closed
            If False, runs viewer in separate thread
        """
        if self.is_running:
            print("Viewer already running")
            return
        
        if blocking:
            self._run_viewer()
        else:
            # Run in separate thread for non-blocking visualization
            self.viewer_thread = threading.Thread(
                target=self._run_viewer,
                daemon=True
            )
            self.viewer_thread.start()
            
            # Wait for viewer to initialize
            time.sleep(0.5)
    
    def _run_viewer(self):
        """Internal method to run viewer loop."""
        self.is_running = True
        
        # Launch passive viewer (user controls camera)
        with mujoco.viewer.launch_passive(
            self.model,
            self.data,
            show_left_ui=True,
            show_right_ui=True
        ) as viewer:
            self.viewer = viewer
            
            # Configure viewer options
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = self.show_contacts
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
            
            # Set camera to good initial view
            viewer.cam.azimuth = 45
            viewer.cam.elevation = -30
            viewer.cam.distance = 2.0
            viewer.cam.lookat[:] = [0, 0, 0.5]
            
            print(f"MuJoCo Viewer launched: {self.title}")
            print("Controls:")
            print("  - Left mouse: Rotate camera")
            print("  - Right mouse: Translate camera")
            print("  - Scroll: Zoom")
            print("  - ESC: Close viewer")
            
            # Viewer loop
            while viewer.is_running():
                # Sync viewer with latest data
                viewer.sync()
                
                # Update custom overlays
                self._render_custom_overlays(viewer)
                
                # Update FPS counter
                self._update_fps()
                
                # Small sleep to prevent CPU spinning
                time.sleep(0.001)
        
        self.is_running = False
        self.viewer = None
    
    def _render_custom_overlays(self, viewer):
        """
        Render custom debugging overlays (force arrows, joint indicators).
        
        This method is called every frame to update visual debugging aids.
        """
        # Render torque arrows
        if self.show_forces:
            for name, arrow_cfg in self.force_arrows.items():
                self._render_force_arrow(viewer, name, arrow_cfg)
        
        # Render joint limit indicators
        if self.show_joint_limits:
            for name, indicator_cfg in self.joint_indicators.items():
                self._render_joint_indicator(viewer, name, indicator_cfg)
    
    def _render_force_arrow(self, viewer, name: str, cfg: dict):
        """
        Render a force/torque arrow using MuJoCo geoms.
        
        Note: This is a simplified implementation. For production use,
        consider using MuJoCo's built-in perturbation forces or
        custom geom creation for more sophisticated arrows.
        """
        if 'actuator_id' in cfg:
            # Actuator torque arrow
            actuator_id = cfg['actuator_id']
            joint_id = cfg['joint_id']
            
            # Get current actuator force/torque
            actuator_force = self.data.actuator_force[actuator_id]
            
            # Get joint position for arrow placement
            joint_pos = self.data.qpos[self.model.jnt_qposadr[joint_id]]
            
            # Arrow length proportional to torque magnitude
            arrow_length = cfg['scale'] * abs(actuator_force)
            
            # Direction based on torque sign
            arrow_dir = 1.0 if actuator_force > 0 else -1.0
            
            # Draw arrow using viewer's add_marker function (if available)
            # Note: Actual implementation depends on MuJoCo viewer API
            # This is a placeholder for the concept
            
        elif cfg.get('type') == 'disturbance':
            # External disturbance arrow
            body_id = cfg['body_id']
            
            # Get external force applied to body (if any)
            # This would come from xfrc_applied
            ext_force = self.data.xfrc_applied[body_id]
            
            # Draw disturbance arrow
            # Placeholder for actual implementation
    
    def _render_joint_indicator(self, viewer, name: str, cfg: dict):
        """
        Render joint limit indicator by coloring joint region.
        
        Colors based on proximity to limits:
        - Green: > warning_zone from limits
        - Yellow: within warning_zone
        - Red: at or beyond limits
        """
        joint_id = cfg['joint_id']
        limits = cfg['limits']
        warning_zone = cfg['warning_zone']
        
        # Get current joint position
        qpos_adr = self.model.jnt_qposadr[joint_id]
        joint_pos = self.data.qpos[qpos_adr]
        
        # Calculate proximity to limits
        range_size = limits[1] - limits[0]
        warning_dist = warning_zone * range_size
        
        lower_warning = limits[0] + warning_dist
        upper_warning = limits[1] - warning_dist
        
        # Determine color based on position
        if joint_pos <= limits[0] or joint_pos >= limits[1]:
            # At or beyond limits - RED
            status_color = (1.0, 0.0, 0.0)
            status_text = "LIMIT"
        elif joint_pos <= lower_warning or joint_pos >= upper_warning:
            # In warning zone - YELLOW
            status_color = (1.0, 1.0, 0.0)
            status_text = "WARNING"
        else:
            # Normal operation - GREEN
            status_color = (0.0, 1.0, 0.0)
            status_text = "OK"
        
        # Overlay status text (if viewer supports text rendering)
        # Placeholder for actual implementation
    
    def _update_fps(self):
        """Update FPS counter for performance monitoring."""
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time
    
    def update(self, data: Optional[mujoco.MjData] = None):
        """
        Update visualization with latest simulation data.
        
        Call this after each mj_step() to keep visualization in sync.
        
        Parameters
        ----------
        data : mujoco.MjData, optional
            Updated data instance (uses self.data if None)
        """
        if data is not None:
            self.data = data
        
        # Sync happens automatically in viewer loop
        # This method is provided for API completeness
    
    def render(self):
        """
        Explicitly trigger a render (useful for step-by-step debugging).
        
        Note: In passive viewer mode, rendering happens automatically.
        This method is provided for compatibility with active rendering modes.
        """
        if self.viewer is not None and self.is_running:
            self.viewer.sync()
    
    def close(self):
        """
        Close the viewer window and cleanup resources.
        """
        self.is_running = False
        
        if self.viewer_thread is not None:
            self.viewer_thread.join(timeout=2.0)
        
        self.viewer = None
        print("MuJoCo Viewer closed")
    
    def set_camera_view(
        self,
        azimuth: float,
        elevation: float,
        distance: float,
        lookat: tuple = (0, 0, 0.5)
    ):
        """
        Set camera viewpoint programmatically.
        
        Useful for capturing consistent screenshots or videos.
        
        Parameters
        ----------
        azimuth : float
            Horizontal rotation angle [degrees]
        elevation : float
            Vertical rotation angle [degrees]
        distance : float
            Distance from target [m]
        lookat : tuple
            Target point (x, y, z) [m]
        """
        if self.viewer is not None:
            self.viewer.cam.azimuth = azimuth
            self.viewer.cam.elevation = elevation
            self.viewer.cam.distance = distance
            self.viewer.cam.lookat[:] = lookat
    
    def capture_frame(self, filepath: str, width: int = 1920, height: int = 1080):
        """
        Capture current frame to image file.
        
        Parameters
        ----------
        filepath : str
            Output image path (e.g., 'frame_001.png')
        width : int
            Image width in pixels
        height : int
            Image height in pixels
        """
        if not self.is_running:
            raise RuntimeError("Viewer must be running to capture frames")
        
        # Render to offscreen buffer
        renderer = mujoco.Renderer(self.model, height, width)
        renderer.update_scene(self.data)
        pixels = renderer.render()
        
        # Save to file (requires PIL/Pillow)
        try:
            from PIL import Image
            img = Image.fromarray(pixels)
            img.save(filepath)
            print(f"Frame saved to {filepath}")
        except ImportError:
            raise ImportError("PIL/Pillow required for frame capture. Install with: pip install Pillow")


def create_standard_visualizer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    show_all_debug: bool = True
) -> MuJoCoVisualizer:
    """
    Create visualizer with standard debugging features enabled.
    
    This convenience function sets up a fully-configured visualizer
    with all debugging aids enabled for typical control system validation.
    
    Parameters
    ----------
    model : mujoco.MjModel
        MuJoCo model instance
    data : mujoco.MjData
        MuJoCo data instance
    show_all_debug : bool
        Enable all debugging visualizations
        
    Returns
    -------
    MuJoCoVisualizer
        Configured visualizer instance
        
    Example
    -------
    >>> visualizer = create_standard_visualizer(model, data)
    >>> visualizer.launch(blocking=False)
    >>> 
    >>> # Run simulation
    >>> for _ in range(1000):
    ...     mujoco.mj_step(model, data)
    ...     visualizer.update(data)
    >>> 
    >>> visualizer.close()
    """
    viz = MuJoCoVisualizer(
        model, 
        data,
        show_forces=show_all_debug,
        show_contacts=show_all_debug,
        show_joint_limits=show_all_debug
    )
    
    # Add standard debugging features
    if show_all_debug:
        # Torque arrows for gimbal motors
        try:
            viz.add_torque_arrow('motor_az', scale=5.0, color=(1.0, 0.0, 0.0, 0.8))
            viz.add_torque_arrow('motor_el', scale=5.0, color=(1.0, 0.5, 0.0, 0.8))
        except ValueError:
            pass  # Motors not found, skip
        
        # Joint limit indicators
        try:
            viz.add_joint_limit_indicator('gimbal_az', warning_zone=0.1)
            viz.add_joint_limit_indicator('gimbal_el', warning_zone=0.1)
        except ValueError:
            pass  # Joints not found, skip
        
        # Disturbance visualization (if disturbance body exists)
        try:
            viz.add_disturbance_arrow('gimbal_platform', scale=10.0)
        except ValueError:
            pass  # Body not found, skip
    
    return viz
