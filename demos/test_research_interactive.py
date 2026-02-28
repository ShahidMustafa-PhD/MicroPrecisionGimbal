#!/usr/bin/env python3
"""
Test script to verify ResearchComparisonPlotter interactive features.

This creates minimal test data to verify the interactive plotting
capabilities are properly integrated.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import the enhanced plotter
from lasercom_digital_twin.core.plots.research_comparison_plotter import (
    ResearchComparisonPlotter
)

def create_test_data(duration=10.0, dt=0.01):
    """Create minimal test data for verification."""
    t = np.arange(0, duration, dt)
    
    # Simple step response simulation
    target_az = np.deg2rad(10.0)
    target_el = np.deg2rad(15.0)
    
    # Different settling behaviors for each controller
    tau_pid = 2.0
    tau_fbl = 1.0
    tau_ndob = 0.5
    
    q_az_pid = target_az * (1 - np.exp(-t/tau_pid))
    q_az_fbl = target_az * (1 - np.exp(-t/tau_fbl))
    q_az_ndob = target_az * (1 - np.exp(-t/tau_ndob))
    
    q_el_pid = target_el * (1 - np.exp(-t/tau_pid))
    q_el_fbl = target_el * (1 - np.exp(-t/tau_fbl))
    q_el_ndob = target_el * (1 - np.exp(-t/tau_ndob))
    
    # Velocities (derivatives)
    qd_az_pid = (target_az / tau_pid) * np.exp(-t/tau_pid)
    qd_az_fbl = (target_az / tau_fbl) * np.exp(-t/tau_fbl)
    qd_az_ndob = (target_az / tau_ndob) * np.exp(-t/tau_ndob)
    
    qd_el_pid = (target_el / tau_pid) * np.exp(-t/tau_pid)
    qd_el_fbl = (target_el / tau_fbl) * np.exp(-t/tau_fbl)
    qd_el_ndob = (target_el / tau_ndob) * np.exp(-t/tau_ndob)
    
    # Simple torques
    torque_az_pid = 0.1 * qd_az_pid
    torque_az_fbl = 0.08 * qd_az_fbl
    torque_az_ndob = 0.06 * qd_az_ndob
    
    torque_el_pid = 0.1 * qd_el_pid
    torque_el_fbl = 0.08 * qd_el_fbl
    torque_el_ndob = 0.06 * qd_el_ndob
    
    # Package into results format
    def make_results(q_az, q_el, qd_az, qd_el, tau_az, tau_el):
        return {
            'log_arrays': {
                'time': t,
                'q_az': q_az,
                'q_el': q_el,
                'qd_az': qd_az,
                'qd_el': qd_el,
                'torque_az': tau_az,
                'torque_el': tau_el,
                'target_az': np.full_like(t, target_az),
                'target_el': np.full_like(t, target_el),
                # Add dummy data for other required fields
                'ekf_q_az': q_az + np.random.normal(0, 1e-6, len(t)),
                'ekf_q_el': q_el + np.random.normal(0, 1e-6, len(t)),
                'ekf_qd_az': qd_az + np.random.normal(0, 1e-6, len(t)),
                'ekf_qd_el': qd_el + np.random.normal(0, 1e-6, len(t)),
                'fsm_az': np.zeros_like(t),
                'fsm_el': np.zeros_like(t),
                'fsm_daz': np.zeros_like(t),
                'fsm_del': np.zeros_like(t),
                'los_error_x': np.random.normal(0, 1e-6, len(t)),
                'los_error_y': np.random.normal(0, 1e-6, len(t)),
                'innovation_az': np.random.normal(0, 1e-6, len(t)),
                'innovation_el': np.random.normal(0, 1e-6, len(t)),
                'P_az_az': np.ones_like(t) * 1e-6,
                'P_el_el': np.ones_like(t) * 1e-6,
                'dist_wind_az': np.zeros_like(t),
                'dist_wind_el': np.zeros_like(t),
                'dist_vib_az': np.zeros_like(t),
                'dist_vib_el': np.zeros_like(t),
                'dist_struct_az': np.zeros_like(t),
                'dist_struct_el': np.zeros_like(t),
            }
        }
    
    results_pid = make_results(q_az_pid, q_el_pid, qd_az_pid, qd_el_pid, torque_az_pid, torque_el_pid)
    results_fbl = make_results(q_az_fbl, q_el_fbl, qd_az_fbl, qd_el_fbl, torque_az_fbl, torque_el_fbl)
    results_ndob = make_results(q_az_ndob, q_el_ndob, qd_az_ndob, qd_el_ndob, torque_az_ndob, torque_el_ndob)
    
    # Add NDOB-specific data
    results_ndob['log_arrays']['d_hat_ndob_az'] = 0.01 * np.sin(2*np.pi*t)
    results_ndob['log_arrays']['d_hat_ndob_el'] = 0.01 * np.cos(2*np.pi*t)
    
    return results_pid, results_fbl, results_ndob, target_az, target_el


def main():
    """Run interactive plotting test."""
    print("="*70)
    print("  TESTING INTERACTIVE RESEARCH COMPARISON PLOTTER")
    print("="*70)
    print()
    print("Creating test data...")
    
    # Generate test data
    results_pid, results_fbl, results_ndob, target_az, target_el = create_test_data()
    
    print("[OK] Test data created")
    print()
    print("Initializing ResearchComparisonPlotter with interactive=True...")
    
    # Create plotter with interactive mode
    plotter = ResearchComparisonPlotter(
        save_figures=False,  # Don't save during test
        show_figures=True,
        interactive=True     # ENABLE INTERACTIVE FEATURES
    )
    
    print("[OK] Plotter initialized")
    print()
    print("Generating sample figures (position, error, torque, velocity)...")
    print()
    
    # Store inputs for plotting
    plotter._results_pid = results_pid
    plotter._results_fbl = results_fbl
    plotter._results_ndob = results_ndob
    plotter._target_az_deg = np.rad2deg(target_az)
    plotter._target_el_deg = np.rad2deg(target_el)
    plotter._target_az_rad = target_az
    plotter._target_el_rad = target_el
    plotter._t_pid = results_pid['log_arrays']['time']
    plotter._t_fbl = results_fbl['log_arrays']['time']
    plotter._t_ndob = results_ndob['log_arrays']['time']
    
    # Generate only figures that work with minimal data
    plotter.figures['fig1_position'] = plotter._plot_position_tracking()
    plotter.figures['fig2_error'] = plotter._plot_tracking_error()
    plotter.figures['fig3_torque'] = plotter._plot_control_torques()
    plotter.figures['fig4_velocity'] = plotter._plot_velocities()
    
    # Make figures interactive
    if plotter.interactive:
        print("\n[OK] Enhancing figures with interactive capabilities...")
        print("     - Zoom regions (Z key)")
        print("     - Vertical/horizontal lines (V/H keys)")
        print("     - Mouse-based selection and deletion")
        print("     - Professional annotation tools")
        print("     - Press ? in any figure for full help")
        
        fig_axes_map = {
            'fig1_position': plotter.figures['fig1_position'].get_axes(),
            'fig2_error': plotter.figures['fig2_error'].get_axes(),
            'fig3_torque': plotter.figures['fig3_torque'].get_axes(),
            'fig4_velocity': plotter.figures['fig4_velocity'].get_axes(),
        }
        
        for fig_name, axes in fig_axes_map.items():
            if fig_name in plotter.figures:
                manager = plotter._make_figure_interactive(
                    plotter.figures[fig_name],
                    axes,
                    fig_name
                )
                if manager:
                    plotter.interactive_managers[fig_name] = manager
        
        print(f"[OK] Made {len(plotter.interactive_managers)} figures interactive")
    
    # Show figures
    plt.show()
    
    print()
    print("="*70)
    print("  INTERACTIVE FEATURES AVAILABLE IN ALL FIGURES:")
    print("="*70)
    print()
    print("  KEYBOARD SHORTCUTS:")
    print("    Z  = Create zoom region (click 2 corners)")
    print("    V  = Add vertical line")
    print("    H  = Add horizontal line")
    print("    M  = Move mode (drag annotations)")
    print("    D  = Delete mode")
    print("    U  = Undo last action")
    print("    S  = Save figure")
    print("    ?  = Show help")
    print()
    print("  MOUSE CONTROLS:")
    print("    Left-click    = Place/select annotation")
    print("    Right-click   = Quick delete")
    print("    Drag in inset = Reposition zoom window")
    print()
    print("  ZOOM DELETION (3 METHODS):")
    print("    1. Press U after creating zoom (undo)")
    print("    2. Click rectangle (turns orange) → Press DELETE")
    print("    3. Right-click inside rectangle (instant)")
    print()
    print("="*70)
    print("  Close figure windows when done testing")
    print("="*70)


if __name__ == "__main__":
    main()
