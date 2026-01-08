"""
Demo: Disturbance and Fault Injection Systems

This script demonstrates the usage of environmental disturbances and
fault injection for the digital twin.
"""

import numpy as np
import matplotlib.pyplot as plt
from lasercom_digital_twin.core.disturbances.disturbance_models import (
    EnvironmentalDisturbances
)
from lasercom_digital_twin.core.simulation.fault_injector import (
    create_fault_injector
)


def demo_disturbances():
    """Demonstrate environmental disturbance generation."""
    print("=" * 70)
    print("DEMO 1: Environmental Disturbances")
    print("=" * 70)
    
    # Configure disturbances
    config = {
        'wind_rms': 0.5,  # N·m
        'wind_correlation_time': 2.0,  # s
        'wind_enabled': True,
        
        'vibration_psd': 1e-6,  # (m/s²)²/Hz
        'vibration_freq_low': 10.0,  # Hz
        'vibration_freq_high': 100.0,  # Hz
        'vibration_enabled': True,
        
        'structural_noise_std': 0.01,  # N·m
        'structural_freq_low': 100.0,  # Hz
        'structural_freq_high': 500.0,  # Hz
        'structural_enabled': True,
        
        'seed': 42
    }
    
    disturbances = EnvironmentalDisturbances(config)
    
    # Generate 5 seconds of disturbances
    dt = 0.001
    n_steps = 5000
    time = []
    wind_az = []
    wind_el = []
    vib_z = []
    structural_az = []
    total_az = []
    
    print(f"\nGenerating {n_steps} steps at dt={dt}s...")
    
    for i in range(n_steps):
        t = i * dt
        state = disturbances.step(
            dt=dt,
            gimbal_az=0.1,
            gimbal_el=0.5
        )
        
        time.append(t)
        wind_az.append(state.wind_torque_az)
        wind_el.append(state.wind_torque_el)
        vib_z.append(state.vibration_accel_z)
        structural_az.append(state.structural_torque_az)
        total_az.append(state.total_torque_az)
    
    # Print statistics
    print(f"\nDisturbance Statistics:")
    print(f"  Wind Az:        mean={np.mean(wind_az):.4f} N·m, "
          f"std={np.std(wind_az):.4f} N·m")
    print(f"  Wind El:        mean={np.mean(wind_el):.4f} N·m, "
          f"std={np.std(wind_el):.4f} N·m")
    print(f"  Vibration Z:    mean={np.mean(vib_z):.6f} m/s², "
          f"std={np.std(vib_z):.6f} m/s²")
    print(f"  Structural Az:  mean={np.mean(structural_az):.6f} N·m, "
          f"std={np.std(structural_az):.6f} N·m")
    print(f"  Total Torque Az: min={np.min(total_az):.4f} N·m, "
          f"max={np.max(total_az):.4f} N·m")
    
    # Create plots
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    
    # Wind torques
    axes[0].plot(time, wind_az, label='Wind Az', alpha=0.7)
    axes[0].plot(time, wind_el, label='Wind El', alpha=0.7)
    axes[0].set_ylabel('Wind Torque [N·m]')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Environmental Disturbances')
    
    # Vibration
    axes[1].plot(time, vib_z, label='Vertical Vibration', alpha=0.7)
    axes[1].set_ylabel('Acceleration [m/s²]')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Structural noise + Total
    axes[2].plot(time, structural_az, label='Structural Noise', alpha=0.5)
    axes[2].plot(time, total_az, label='Total Torque Az', linewidth=2)
    axes[2].set_xlabel('Time [s]')
    axes[2].set_ylabel('Torque [N·m]')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('disturbance_demo.png', dpi=150)
    print(f"\n  Saved plot to: disturbance_demo.png")
    
    return disturbances


def demo_fault_injection():
    """Demonstrate fault injection framework."""
    print("\n" + "=" * 70)
    print("DEMO 2: Fault Injection")
    print("=" * 70)
    
    # Define custom fault schedule
    config = {
        'seed': 42,
        'faults': [
            {
                'type': 'sensor_dropout',
                'target': 'gyro_az',
                'start_time': 2.0,
                'duration': 0.5
            },
            {
                'type': 'sensor_bias',
                'target': 'encoder_az',
                'start_time': 3.0,
                'duration': 2.0,
                'parameters': {'bias_value': 1e-4}  # 100 µrad
            },
            {
                'type': 'backlash_growth',
                'target': 'az',
                'start_time': 5.0,
                'duration': 3.0,
                'parameters': {'growth_rate': 0.2}  # 20% per second
            },
            {
                'type': 'fsm_saturation',
                'start_time': 7.0,
                'duration': 1.5,
                'parameters': {
                    'magnitude': 600e-6,  # 600 µrad
                    'type': 'step'
                }
            }
        ]
    }
    
    injector = create_fault_injector('custom', **config)
    
    print(f"\nConfigured {len(injector.fault_events)} fault events:")
    for i, event in enumerate(injector.fault_events, 1):
        print(f"  {i}. {event.fault_type.value:20s} "
              f"@ t={event.start_time:.1f}s "
              f"(duration={event.duration}s)")
    
    # Simulate and track faults
    dt = 0.01
    duration = 10.0
    n_steps = int(duration / dt)
    
    time = []
    sensor_failed = []
    bias_value = []
    backlash_scale = []
    fsm_saturation = []
    
    print(f"\nSimulating fault timeline...")
    
    for i in range(n_steps):
        t = i * dt
        
        time.append(t)
        sensor_failed.append(
            1 if injector.is_sensor_failed('gyro_az', t) else 0
        )
        bias_value.append(injector.get_sensor_bias('encoder_az', t))
        backlash_scale.append(injector.get_backlash_scale('az', t))
        
        sat_dist = injector.get_fsm_saturation_disturbance(t)
        fsm_saturation.append(
            sat_dist['los_error_x'] * 1e6 if sat_dist else 0.0
        )
    
    # Print fault summary
    print(f"\nFault Summary:")
    sensor_dropout_duration = np.sum(sensor_failed) * dt
    print(f"  Gyro dropout:    {sensor_dropout_duration:.2f}s total")
    print(f"  Encoder bias:    max={np.max(bias_value)*1e6:.1f} µrad")
    print(f"  Backlash growth: max scale={np.max(backlash_scale):.2f}x")
    print(f"  FSM saturation:  max error={np.max(fsm_saturation):.1f} µrad")
    
    # Create plots
    fig, axes = plt.subplots(4, 1, figsize=(10, 10))
    
    # Sensor dropout
    axes[0].fill_between(time, 0, sensor_failed, alpha=0.5, color='red')
    axes[0].set_ylabel('Gyro Failed')
    axes[0].set_ylim([-0.1, 1.1])
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Fault Injection Timeline')
    
    # Sensor bias
    axes[1].plot(time, np.array(bias_value) * 1e6, linewidth=2)
    axes[1].set_ylabel('Encoder Bias [µrad]')
    axes[1].grid(True, alpha=0.3)
    
    # Backlash growth
    axes[2].plot(time, backlash_scale, linewidth=2, color='orange')
    axes[2].axhline(1.0, color='k', linestyle='--', alpha=0.3)
    axes[2].set_ylabel('Backlash Scale')
    axes[2].grid(True, alpha=0.3)
    
    # FSM saturation test
    axes[3].plot(time, fsm_saturation, linewidth=2, color='purple')
    axes[3].set_xlabel('Time [s]')
    axes[3].set_ylabel('FSM Test Error [µrad]')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fault_injection_demo.png', dpi=150)
    print(f"\n  Saved plot to: fault_injection_demo.png")
    
    return injector


def demo_composite_scenarios():
    """Demonstrate pre-configured composite scenarios."""
    print("\n" + "=" * 70)
    print("DEMO 3: Composite Fault Scenarios")
    print("=" * 70)
    
    scenarios = [
        ('none', 'No faults'),
        ('sensor_degradation', 'Gradual sensor degradation'),
        ('mechanical_wear', 'Progressive mechanical wear'),
        ('mission_stress', 'Comprehensive stress test')
    ]
    
    print("\nAvailable pre-configured scenarios:")
    for name, description in scenarios:
        injector = create_fault_injector(name, seed=42)
        n_faults = len(injector.fault_events)
        print(f"  {name:25s}: {description:40s} ({n_faults} faults)")
    
    # Demo mission stress scenario
    print("\nMission Stress Test Details:")
    injector = create_fault_injector('mission_stress', seed=42)
    
    for event in injector.fault_events:
        print(f"  t={event.start_time:5.1f}s: {event.fault_type.value:20s} "
              f"target={event.target or 'N/A':10s} "
              f"duration={event.duration}s")


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("DISTURBANCE AND FAULT INJECTION DEMONSTRATION")
    print("=" * 70)
    
    # Run demos
    disturbances = demo_disturbances()
    injector = demo_fault_injection()
    demo_composite_scenarios()
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - disturbance_demo.png")
    print("  - fault_injection_demo.png")
    print("\nNext steps:")
    print("  1. Integrate disturbances into simulation_runner.py")
    print("  2. Apply faults to sensor/actuator models")
    print("  3. Run robustness validation tests")
    print("  4. See docs/DISTURBANCE_FAULT_INTEGRATION.md for details")
    print()
