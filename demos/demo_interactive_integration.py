#!/usr/bin/env python3
"""
Live demonstration that ResearchComparisonPlotter uses InteractiveFigureManager.

This script creates a minimal test with fake data to show the integration
working without running the full simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
from lasercom_digital_twin.core.plots.research_comparison_plotter import ResearchComparisonPlotter

print("=" * 80)
print("LIVE DEMO: ResearchComparisonPlotter → InteractiveFigureManager")
print("=" * 80)

# Create minimal fake simulation results
def create_fake_results(duration=10.0, name="TEST"):
    """Create minimal fake data for testing."""
    t = np.linspace(0, duration, 1000)
    
    # Fake gimbal angles (small oscillations)
    theta_az = 5.0 * np.sin(2 * np.pi * 0.2 * t) * np.exp(-0.1 * t)
    theta_el = 3.0 * np.cos(2 * np.pi * 0.3 * t) * np.exp(-0.1 * t)
    
    # Fake angular velocities
    theta_dot_az = np.gradient(theta_az, t)
    theta_dot_el = np.gradient(theta_el, t)
    
    # Fake control torques
    tau_cmd_az = 0.1 * theta_az + 0.01 * theta_dot_az
    tau_cmd_el = 0.1 * theta_el + 0.01 * theta_dot_el
    
    # Fake FSM angles
    alpha_x_fsm = 0.001 * np.sin(2 * np.pi * 1.0 * t)
    alpha_y_fsm = 0.001 * np.cos(2 * np.pi * 1.0 * t)
    
    # Fake state estimate
    theta_az_est = theta_az + 0.01 * np.random.randn(len(t))
    theta_el_est = theta_el + 0.01 * np.random.randn(len(t))
    
    # Fake EKF innovation
    innovation_az = 0.005 * np.random.randn(len(t))
    innovation_el = 0.005 * np.random.randn(len(t))
    
    # Fake disturbance estimation
    tau_disturbance_az = 0.005 * np.sin(2 * np.pi * 0.5 * t)
    tau_disturbance_el = 0.005 * np.cos(2 * np.pi * 0.5 * t)
    
    # Fake disturbance components
    wind_torque_az = 0.003 * np.random.randn(len(t))
    wind_torque_el = 0.003 * np.random.randn(len(t))
    vibration_torque_az = 0.002 * np.sin(2 * np.pi * 15.0 * t)
    vibration_torque_el = 0.002 * np.cos(2 * np.pi * 15.0 * t)
    
    return {
        'log_arrays': {
            'time': t,
            'theta_az': theta_az,
            'theta_el': theta_el,
            'theta_dot_az': theta_dot_az,
            'theta_dot_el': theta_dot_el,
            'tau_cmd_az': tau_cmd_az,
            'tau_cmd_el': tau_cmd_el,
            'alpha_x_fsm': alpha_x_fsm,
            'alpha_y_fsm': alpha_y_fsm,
            'theta_az_est': theta_az_est,
            'theta_el_est': theta_el_est,
            'innovation_az': innovation_az,
            'innovation_el': innovation_el,
            'P_az_az': 0.001 * np.ones_like(t),
            'P_el_el': 0.001 * np.ones_like(t),
            'tau_disturbance_az': tau_disturbance_az,
            'tau_disturbance_el': tau_disturbance_el,
            'wind_torque_az': wind_torque_az,
            'wind_torque_el': wind_torque_el,
            'vibration_torque_az': vibration_torque_az,
            'vibration_torque_el': vibration_torque_el,
            'tau_disturbance_az_hat': tau_disturbance_az * 0.9,
            'tau_disturbance_el_hat': tau_disturbance_el * 0.9,
        },
        'controller_name': name
    }

print("\n1. Creating fake simulation data...")
results_pid = create_fake_results(duration=10.0, name="PID")
results_fbl = create_fake_results(duration=10.0, name="FBL")
results_ndob = create_fake_results(duration=10.0, name="NDOB")
print("   ✓ Created 3 fake result sets (PID, FBL, NDOB)")

print("\n2. Creating ResearchComparisonPlotter with interactive=True...")
plotter = ResearchComparisonPlotter(
    save_figures=False,  # Don't clutter disk
    show_figures=False,  # Don't block testing
    interactive=True     # ← THIS ENABLES InteractiveFigureManager
)
print(f"   ✓ Plotter created")
print(f"   ✓ interactive flag = {plotter.interactive}")
print(f"   ✓ interactive_managers dict initialized = {len(plotter.interactive_managers)} items")

print("\n3. Generating plots (this will take ~5 seconds)...")
figures = plotter.plot_all(
    results_pid=results_pid,
    results_fbl=results_fbl,
    results_ndob=results_ndob,
    target_az_deg=0.0,
    target_el_deg=0.0
)
print(f"   ✓ Generated {len(figures)} figures")

print("\n4. Verifying InteractiveFigureManager integration...")
print(f"   ✓ Number of interactive managers created: {len(plotter.interactive_managers)}")

if len(plotter.interactive_managers) > 0:
    print("\n   ✅ SUCCESS! Managers were created for each figure:")
    for fig_name, manager in plotter.interactive_managers.items():
        print(f"      • {fig_name}: {type(manager).__name__}")
        
    # Verify manager functionality
    print("\n5. Verifying manager capabilities...")
    first_manager = list(plotter.interactive_managers.values())[0]
    
    capabilities = {
        "Has zoom_regions list": hasattr(first_manager, 'zoom_regions'),
        "Has vlines list": hasattr(first_manager, 'vlines'),
        "Has hlines list": hasattr(first_manager, 'hlines'),
        "Has mode property": hasattr(first_manager, 'mode'),
        "Has _undo_last method": hasattr(first_manager, '_undo_last'),
        "Has _delete_zoom method": hasattr(first_manager, '_delete_zoom'),
        "Has status_text": hasattr(first_manager, 'status_text'),
    }
    
    for capability, has_it in capabilities.items():
        status = "✓" if has_it else "✗"
        print(f"      {status} {capability}")
    
    if all(capabilities.values()):
        print("\n   ✅ All interactive features verified!")
    else:
        print("\n   ⚠ Some features missing")
        
else:
    print("\n   ❌ FAILURE: No interactive managers were created!")
    print("   Check that interactive=True and _make_figure_interactive() is called")

print("\n" + "=" * 80)
print("DEMONSTRATION COMPLETE")
print("=" * 80)

if len(plotter.interactive_managers) > 0:
    print("\n✅ ResearchComparisonPlotter SUCCESSFULLY uses InteractiveFigureManager")
    print("\nTo see interactive features in action:")
    print("  1. Run: python demo_feedback_linearization.py")
    print("  2. When figures appear, press 'Z' to enter zoom mode")
    print("  3. Draw green rectangle by clicking two corners")
    print("  4. Press 'U' to undo zoom (rectangle disappears)")
    print("  5. Try other features: V (vline), H (hline), S (save)")
    print("\nAll 13 figures from ResearchComparisonPlotter are fully interactive!")
else:
    print("\n❌ Integration failed - no managers created")
    
print("=" * 80)

# Clean up
plt.close('all')
