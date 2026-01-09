# Ground-Based Satellite Laser Communication Terminal
## System Architecture Documentation

---

## 1. System Component Classification

The terminal is fundamentally divided into three operational domains: **Mechanical**, **Optical**, and **Virtual (Computational/Control)**.

| Component / Function | Classification | Description |
|---------------------|----------------|-------------|
| **Coarse Pointing Assembly (CPA)** | Mechanical | The 2-DOF structure (Gimbal) responsible for large-angle tracking. |
| **Azimuth (Az) Joint** | Mechanical | The physical joint providing rotation about the local vertical axis. |
| **Elevation (El) Joint** | Mechanical | The physical joint providing rotation about the local horizontal axis. |
| **Fast Steering Mirror (FSM)** | Mechanical | A small mirror actuated by piezo/voice-coils for high-bandwidth Tip/Tilt. It has physical movement but acts in the optical domain. |
| **Quadrant Detector (QPD)** | Optical | The sensor that converts the received laser spot position into voltage signals. |
| **Laser/Telescope Optics** | Optical | The non-moving parts (lenses, mirrors) that condition the beam and couple it into the system. |
| **K-Mirror/Prism Equivalent** | Optical | The arrangement used for non-mechanical roll compensation (Field Rotation). |
| **Internal Optical Roll Compensation** | Virtual | The control algorithm that calculates the required rotation for the optical elements to stabilize the image/sensor frame. |
| **Line-of-Sight (LOS) Estimation** | Virtual | Kalman Filter/EKF algorithms fusing sensor data to estimate the true pointing error. |
| **Control System Logic (PID/LQR)** | Virtual | All software-implemented control laws driving the motors and the FSM. |

---

## 2. System Block Diagram (Textual Representation)

The system operates under a hierarchical, decoupled control loop: **Coarse (slow)** and **Fine (fast)**.

```
                    [Satellite Ephemeris]
                            |
                            v
                  {Coarse Gimbal Controller}
                            |
                            v
                    (CPA Actuator Motors)
                            |
                            v
                  [2-DOF Az/El Gimbal]
                            |
              +-------------+-------------+
              |                           |
              v                           v
    ((Telescope Optics))    [Optical Roll Compensation / K-Mirror]
                                          |
        ┌─────────────────────────────────┼─────────────────────────────┐
        │                                 |                             │
        │  Coarse Loop (Low Bandwidth)    v                             │
        │                       (Fast Steering Mirror: FSM)             │
        │                                 |                             │
        │                                 v                             │
        │                       (Laser Beam / LOS Error)                │
        │                                 |                             │
        │                                 v                             │
        │                    [Quadrant Detector: QPD Sensor]            │
        │                                 |                             │
        │                                 v                             │
        │                        {Fine FSM Controller}                  │
        │                                 |                             │
        └─────────────────────────────────┼─────────────────────────────┘
                    |                     |
                    v                     v
          [Az/El Encoders & Gyros]   (FSM Feedback)
                    |                     |
                    +─────────┬───────────+
                              v
                   [Sensor Fusion / EKF]
                              |
              +---------------+---------------+
              |                               |
              v                               v
    {Coarse Gimbal Controller}      {Fine FSM Controller}
```

### Control Loop Hierarchy

**Coarse Loop (Low Bandwidth: ~1-10 Hz)**
- Input: Satellite ephemeris, current Az/El position
- Sensors: Az/El encoders, rate gyroscopes
- Actuators: CPA motors (Az/El)
- Purpose: Large-angle tracking, handles slew maneuvers

**Fine Loop (High Bandwidth: ~100-1000 Hz)**
- Input: QPD error signal, LOS estimate
- Sensors: Quadrant Detector (QPD)
- Actuators: Fast Steering Mirror (FSM)
- Purpose: Fine pointing correction, vibration isolation, disturbance rejection

---

## 3. Coordinate Frame Definitions

The system relies on at least **five primary coordinate frames** to map mechanical movement to optical pointing.

| Frame Name | Acronym | Definition | Purpose |
|-----------|---------|------------|---------|
| **Inertial/Geocentric** | I-Frame | A non-rotating, non-accelerating frame used for satellite ephemeris calculations (e.g., ECEF or J2000). | Reference for target trajectory. |
| **Site/Local Vertical** | S-Frame | Origin at the CPA pivot point; Z-axis is local vertical (Up), X-axis is North/East. | Reference for Azimuth control (Earth-fixed). |
| **Gimbal/Mechanical** | M-Frame | Origin at the CPA pivot point; Az-axis is along Z of S-Frame, El-axis rotates with the structure. | The frame in which physical motor commands are applied. |
| **Optical/Line-of-Sight** | O-Frame | Origin on the beam path; Z-axis is collinear with the desired LOS vector. X/Y axes define the optical focal plane. | The frame where the pointing error (Tip/Tilt) is measured and corrected by the FSM. |
| **Sensor/QPD** | Q-Frame | Origin at the center of the QPD; axes are aligned with the QPD's four quadrants. | The frame where the raw voltage error signal is generated. Crucially, the K-Mirror compensates for rotation between the M-Frame and the Q-Frame. |

### Frame Transformation Chain

```
I-Frame → S-Frame → M-Frame → O-Frame → Q-Frame
  (Target)  (Site)   (Gimbal)  (Optical)  (Sensor)
```

**Key Transformations:**
- **I → S**: Geodetic transformation (ECEF, lat/lon/alt)
- **S → M**: Az/El rotation (Euler angles)
- **M → O**: Optical alignment + FSM Tip/Tilt
- **O → Q**: K-Mirror roll compensation

---

## 4. Assumptions and Exclusions

### Assumptions (For Initial Digital Twin)

| Assumption | Description |
|-----------|-------------|
| **Ideal Site** | The terminal site is stationary and the S-Frame is perfectly aligned with the local gravity vector. |
| **Rigid Structure** | The CPA structure is initially modeled as a rigid body. Structural compliance will be modeled using MuJoCo spring/damper elements but not via full flexible body dynamics. |
| **Small Angle Optics** | All fine pointing (FSM/QPD) occurs in the small-angle approximation. |
| **Decoupled LOS** | The LOS error is modeled as linear Tip/Tilt in the O-Frame. Cross-coupling in the optics is a perturbation, not the primary physics. |
| **Simplified FSM** | FSM dynamics are modeled by a reduced-order, linearized transfer function (e.g., 2nd-order system) for high bandwidth, with hysteresis and saturation added as non-linearities. |

### Exclusions (Out of Scope for Initial System Definition)

| Exclusion | Rationale |
|-----------|-----------|
| **Atmospheric Effects** | Turbulence, beam wander, and scintillations are excluded from the initial geometric optics model. |
| **Relativistic/Non-Inertial Effects** | High-precision time-of-flight or relativistic light bending are excluded. |
| **Thermal Modeling** | Full heat-transfer simulation is excluded, though the effect of thermal variation on parameters (e.g., misalignment) will be included. |
| **Communication Protocol** | The modeling of the actual data transmission/reception (the laser communication layer itself) is excluded; only the pointing link is considered. |

---
##
## 5. System Design Philosophy

### Hierarchical Control Architecture

The system employs a **two-stage control architecture**:

1. **Coarse Loop**: Handles large-angle slewing and tracking using the mechanical gimbal
2. **Fine Loop**: Provides high-bandwidth disturbance rejection using the FSM

This separation allows:
- Independent tuning and optimization of each control loop
- Clear responsibility boundaries between subsystems
- Graceful degradation (gimbal can operate without FSM)

### Optical Roll Compensation Strategy

**No Mechanical Roll Axis**: The system uses **internal optical roll compensation** via K-Mirror or equivalent optical elements. This approach:
- Reduces mechanical complexity (2-DOF vs 3-DOF)
- Eliminates cable wrap issues
- Provides faster roll compensation bandwidth
- Requires computational de-rotation of sensor frame

### Sensor Fusion and State Estimation

The system employs an **Extended Kalman Filter (EKF)** to fuse:
- Gimbal encoders (position)
- Rate gyroscopes (angular velocity)
- QPD measurements (line-of-sight error)

This provides a unified state estimate for both control loops.

---

## 6. Key Performance Requirements

### Pointing Accuracy

| Requirement | Target Value |
|------------|--------------|
| Coarse Pointing (Gimbal) | < 100 µrad RMS |
| Fine Pointing (FSM) | < 5 µrad RMS |
| Total System Pointing | < 5 µrad RMS |

### Bandwidth

| Subsystem | Bandwidth |
|-----------|-----------|
| Coarse Loop | 1-10 Hz |
| Fine Loop | 100-1000 Hz |
| Sensor Fusion | 100 Hz (minimum) |

### Range of Motion

| Axis | Range |
|------|-------|
| Azimuth | 0° to 360° (continuous) |
| Elevation | 10° to 90° |
| FSM Tip/Tilt | ±1° (optical) |

---

**Document Version:** 1.0  
**Last Updated:** January 8, 2026  
**Status:** Baseline Architecture Definition
