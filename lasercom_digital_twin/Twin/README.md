# Twin - LaserCom Digital Twin Visualization Module

## Overview

High-fidelity 3D visualization of the laser communication gimbal system using PyVista with Physically Based Rendering (PBR). This module provides professional aerospace-grade CAD asset generation and real-time kinematic simulation.

## Architecture

### Modular Design

```
Twin/
├── gimbal_model.py       # Geometry generation (CAD assets)
├── simulation_demo.py    # Trajectory playback and visualization
├── __init__.py           # Package exports
└── README.md             # This file
```

### Class Hierarchy

**gimbal_model.py** (Digital Twin Asset):
- `AzimuthBaseAssembly` - Base housing with internal motor, encoder, MEMS gyro
- `ElevationYokeAssembly` - U-shaped frame with pivot bearings
- `ElevationHousingAssembly` - Telescope barrel with optics and sensor suite
- `CableHarnessAssembly` - Flexible cable harness (spline-based)
- `GimbalModel` - Complete hierarchical model with kinematics

**simulation_demo.py** (Visualization Orchestration):
- `StudioEnvironment` - PBR lighting and environment setup
- `TrajectoryGenerator` - Multiple trajectory patterns
- `SimulationOrchestrator` - Main simulation loop and UI

## Features

### Mechanical Fidelity
- **Hierarchical Kinematics**: Base (fixed) → Yoke (azimuth) → Housing (elevation)
- **Internal Components**: Motors, encoders, MEMS gyro, sensors
- **Material Accuracy**: Anodized aluminum, matte plastics, transparent optics

### Visual Quality (PBR)
- **Physically Based Rendering**: Metallic/roughness workflow
- **Studio Lighting**: Key, fill, rim, and ambient lights
- **Shadow Casting**: Real-time shadows for depth perception
- **Anti-Aliasing**: MSAA for smooth edges

### Trajectory Patterns
1. **Sinusoidal Scan** - Lissajous pattern (default)
2. **Conical Scan** - Circular acquisition
3. **Step-and-Dwell** - Discrete positioning
4. **Spiral Search** - Expanding search pattern

## Usage

### Basic Execution

```python
# Run default simulation (sinusoidal trajectory, 15s)
python simulation_demo.py
```

### Custom Trajectory

```python
from simulation_demo import SimulationOrchestrator

orchestrator = SimulationOrchestrator(
    window_size=(1920, 1080),
    offline_mode=True  # Set False for interactive window
)

orchestrator.run_simulation(
    duration=20.0,
    trajectory_type='conical',  # or 'sinusoidal', 'step_dwell', 'spiral'
    save_animation=True
)
```

### Interactive Mode

```python
# For manual control and interactive exploration
orchestrator = SimulationOrchestrator(offline_mode=False)

# Update gimbal manually
orchestrator.gimbal.update_pose(az_deg=30, el_deg=15)
orchestrator.plotter.show()
```

### Programmatic Control

```python
from gimbal_model import GimbalModel
import pyvista as pv

# Create plotter
pl = pv.Plotter()

# Build gimbal
gimbal = GimbalModel(pl)

# Update pose
gimbal.update_pose(az_deg=45, el_deg=20)

# Get current state
az, el = gimbal.get_current_pose()
print(f"Current: Az={az:.2f}° El={el:.2f}°")

# Render
pl.show()
```

## Engineering Details

### Elevation Assembly Specifications

**Telescope Barrel:**
- Material: Anodized aluminum (metallic=1.0, roughness=0.4)
- Dimensions: 180mm length, 80mm diameter
- Construction: Hollow tube with 8mm wall thickness

**Primary Mirror:**
- Finish: Silver mirror (metallic=1.0, roughness=0.0)
- Diameter: 64mm
- Location: -85mm from origin (rear mount)

**Front Lens:**
- Material: Optical glass (cyan tint)
- Transparency: 70% (opacity=0.3)
- Specular: High (specular=1.0, power=128)

**Sensor Package:**
- Type: Native industrial housing
- Finish: Matte black engineering plastic (roughness=0.9)
- Mounting: Side-mounted bracket
- Components: Detector window, mounting bracket, label annotation

### Coordinate System

- **X-axis**: Forward (optical boresight)
- **Y-axis**: Right
- **Z-axis**: Up (azimuth rotation axis)
- **Origin**: Base mounting interface

### Transform Hierarchy

```
World Frame
├── Base Assembly (fixed)
├── Yoke Assembly (R_z(azimuth))
│   └── Housing Assembly (R_y(elevation))
│       ├── Optics (parented)
│       ├── Sensor Suite (parented)
│       └── Cables (parented)
```

## Output

### Animation Export
- Format: GIF (30 FPS)
- Resolution: 1920x1080
- Filename: `gimbal_{trajectory_type}.gif`
- Location: `lasercom_digital_twin/Twin/`

### UI Overlays
- **Status Panel**: Time, frame count, azimuth, elevation
- **Metrics Panel**: Update rate, gimbal type, control mode
- **Title Banner**: System identification

## Dependencies

```bash
pip install pyvista numpy scipy
```

## Performance

- **Update Rate**: ~30 Hz (33ms per frame)
- **Rendering**: Hardware-accelerated OpenGL
- **Memory**: ~200MB (typical)

## Future Enhancements

- [ ] MuJoCo physics integration
- [ ] Real-time sensor data overlay
- [ ] Multi-gimbal formation visualization
- [ ] VR/AR export support
- [ ] Custom material shaders

## License

Part of the MicroPrecisionGimbal Digital Twin framework.
DO-178C Level B compliance standards.

## Contact

Expert Robotics Simulation Engineer
Date: January 31, 2026
