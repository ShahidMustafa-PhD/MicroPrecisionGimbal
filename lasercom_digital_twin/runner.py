#!/usr/bin/env python3
"""
Command-line runner for the Lasercom Digital Twin simulation.

This script acts as the primary entry point for the simulation framework.
It handles argument parsing, configuration loading, and instantiates the 
DigitalTwinRunner with the correct context.

Usage:
    python -m lasercom_digital_twin.runner --fidelity L4 --duration 60 --visualize
"""

import argparse
import json
import sys
import os
from pathlib import Path

# Ensure the project root is in sys.path to resolve internal modules correctly
# This allows running as a script even if not installed as a package
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from lasercom_digital_twin.core.simulation.simulation_runner import DigitalTwinRunner, SimulationConfig
except ImportError as e:
    print(f"Import Error: {e}")
    print("Ensure you are running from the project root or have installed the package.")
    print("Example: python -m lasercom_digital_twin.runner ...")
    sys.exit(1)

def load_fidelity_config(fidelity_level: str) -> dict:
    """Load specific fidelity parameters from the master config file."""
    # Config is at project root / config / fidelity_levels.json
    config_path = Path(__file__).resolve().parent.parent / "config" / "fidelity_levels.json"
    
    if not config_path.exists():
        print(f"Configuration Error: Config file not found at {config_path}")
        # Return empty dict fallback or validation error in production
        print("Using default internal parameters.")
        return {}
    
    try:
        with open(config_path, 'r') as f:
            full_config = json.load(f)
            configs = full_config.get("fidelity_levels", {})
        
        if fidelity_level not in configs:
            print(f"Error: Fidelity level '{fidelity_level}' not defined in config.")
            print(f"Available levels: {list(configs.keys())}")
            sys.exit(1)
            
        return configs[fidelity_level]
    except json.JSONDecodeError:
        print(f"Error: Failed to parse JSON config at {config_path}")
        sys.exit(1)

def map_config_to_kwargs(json_config: dict) -> dict:
    """Map JSON fidelity config structure to SimulationConfig flattened kwargs."""
    params = json_config.get("parameters", {})
    sim = params.get("simulation", {})
    
    kwargs = {}
    
    # Timing mappings
    if "dt" in sim:
        kwargs["dt_sim"] = sim["dt"]
        kwargs["dt_fine"] = sim["dt"] 
        # Heuristic: Coarse loop is slower
        kwargs["dt_coarse"] = sim["dt"] * 10
        kwargs["dt_encoder"] = sim["dt"]
        kwargs["dt_gyro"] = sim["dt"]
        kwargs["dt_qpd"] = sim["dt"]
        kwargs["log_period"] = sim["dt"]

    # Component Configs
    parts = {
        "motor_config": ["actuators", "coarse_gimbal"],
        "fsm_config": ["actuators", "fsm"],
        "qpd_config": ["sensors", "qpd"],
        "gyro_config": ["sensors", "imu"],
        "encoder_config": ["sensors", "encoders"],
        "estimator_config": ["estimator"],
        "coarse_controller_config": ["controller"],
        "fsm_controller_config": ["controller"] 
    }
    
    for key, path in parts.items():
        val = params
        for p in path:
            val = val.get(p, {})
        kwargs[key] = val
        
    return kwargs

def main():
    parser = argparse.ArgumentParser(
        description="Lasercom Digital Twin - High-Fidelity Optical Pointing Simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--fidelity", 
        type=str, 
        default="L1",
        choices=["L1", "L2", "L3", "L4"],
        help="Simulation fidelity level (L1=Ideal, L4=Production w/ Faults)"
    )
    parser.add_argument(
        "--duration", 
        type=float, 
        default=30.0,
        help="Duration of the simulation in seconds"
    )
    parser.add_argument(
        "--visualize", 
        action="store_true",
        help="Enable real-time MuJoCo visualization (requires GUI environment)"
    )
    parser.add_argument(
        "--headless", 
        action="store_true",
        help="Force headless mode (overrides visualization)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic execution"
    )

    args = parser.parse_args()

    print("=" * 60)
    print(f"Initializing Lasercom Digital Twin (Fidelity: {args.fidelity})")
    print("=" * 60)

    # 1. Load Configuration
    raw_config = load_fidelity_config(args.fidelity)
    config_dict = map_config_to_kwargs(raw_config)
    
    # Override config with CLI arguments where applicable
    # Note: simulation_duration is used in run_simulation(), not in SimulationConfig
    config_dict["seed"] = args.seed
    config_dict["enable_visualization"] = args.visualize and not args.headless

    # 2. Instantiate Simulation
    try:
        print("Instantiate Simulation")
        # Create config object (handling potential missing keys via defaults)
        sim_config = SimulationConfig(**config_dict)
        runner = DigitalTwinRunner(sim_config)
        
        # 3. Run Simulation (Runner handles visualization internally)
        results = runner.run_simulation(duration=args.duration)

        # 4. Output Results
        print("\n" + "="*30)
        print(" PERFORMANCE SUMMARY")
        print("="*30)
        print(f"RMS Pointing Error: {results.get('los_error_rms', 0.0)*1e6:.2f} µrad")
        print(f"FSM Saturation:     {results.get('fsm_saturation_percent', 0.0):.2f}%")
        print(f"Execution Time:     {results.get('duration', 0.0):.2f} s")
        print("="*30 + "\n")

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nCRITICAL FAILURE: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
