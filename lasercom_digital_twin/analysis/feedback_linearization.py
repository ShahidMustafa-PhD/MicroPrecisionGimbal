"""
NdobFbl MVC Model - Encapsulates the 3-Way Controller Comparison (PID vs FBL vs FBL+NDOB).
"""

import json
import re
import os
import copy
import numpy as np
from typing import Dict
from lasercom_digital_twin.core.simulation.simulation_runner import DigitalTwinRunner, SimulationConfig
from lasercom_digital_twin.core.plots.metrics_utils import compute_tracking_metrics
from lasercom_digital_twin.core.plots.research_comparison_plotter import ResearchComparisonPlotter


class NdobFbl:
    """
    Model class that runs the complete 3-way analysis for the GUI.
    """

    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_path = os.path.join(base_dir, 'data', 'config.jsonc')
        if not os.path.exists(config_path):
            config_path = os.path.join('lasercom_digital_twin', 'data', 'config.jsonc')
        
        self.default_config = self._load_commented_json(config_path)

    def _load_commented_json(self, filepath: str) -> dict:
        """Loads a JSON file, stripping C-style (//) comments first."""
        if not os.path.exists(filepath):
            return {}
            
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Regular expression to remove // comments, ignoring // inside strings
        comment_re = re.compile(
            r'(^)?[^\S\n]*/(?:\*(.*?)\*/[^\S\n]*|/[^\n]*)($)?',
            re.DOTALL | re.MULTILINE
        )
        content_no_comments = comment_re.sub('', content)
        return json.loads(content_no_comments)

    def run_analysis(self, config: dict, progress_callback=None) -> dict:
        """
        Executes the three-way simulation, computes metrics, and generates plot figures.

        Parameters
        ----------
        config : dict
            The flat dictionary from ControlPanelWidget.get_simulation_config().
        progress_callback : callable, optional
            A function that accepts a string summarizing the current status.
            
        Returns
        -------
        dict
            Contains "metrics" and "figures".
        """
        
        # Overlay the GUI overrides onto the default JSON configuration
        merged_config = dict(self.default_config)
        merged_config.update(config)

        duration = merged_config.get("_gui_duration", merged_config.get("duration", 5.0))

        # ---------------------------------------------------------------------
        # 1. PID Simulation
        # ---------------------------------------------------------------------
        if progress_callback:
            progress_callback("Running Baseline PID Simulation...")
            
        cfg_pid = self._build_pid_config(merged_config)
        runner_pid = DigitalTwinRunner(cfg_pid)
        results_pid = runner_pid.run_simulation(duration=duration)

        # ---------------------------------------------------------------------
        # 2. FBL Simulation
        # ---------------------------------------------------------------------
        if progress_callback:
            progress_callback("Running Feedback Linearization (FBL) Simulation...")

        cfg_fbl = self._build_fbl_config(merged_config)
        runner_fbl = DigitalTwinRunner(cfg_fbl)
        results_fbl = runner_fbl.run_simulation(duration=duration)

        # ---------------------------------------------------------------------
        # 3. FBL + NDOB Simulation
        # ---------------------------------------------------------------------
        if progress_callback:
            progress_callback("Running FBL + NDOB Simulation...")

        cfg_ndob = self._build_ndob_config(merged_config)
        runner_ndob = DigitalTwinRunner(cfg_ndob)
        results_ndob = runner_ndob.run_simulation(duration=duration)

        # ---------------------------------------------------------------------
        # Analysis & Metrics
        # ---------------------------------------------------------------------
        if progress_callback:
            progress_callback("Computing metrics and generating figures...")

        target_az_rad = merged_config.get("target_az", np.deg2rad(merged_config.get("target_az_deg", 0.0)))
        target_el_rad = merged_config.get("target_el", np.deg2rad(merged_config.get("target_el_deg", 0.0)))

        metrics = self._compute_metrics(results_pid, results_fbl, results_ndob, target_az_rad, target_el_rad)

        target_az_deg = np.rad2deg(target_az_rad)
        target_el_deg = np.rad2deg(target_el_rad)

        figures = self._generate_figures(results_pid, results_fbl, results_ndob, target_az_deg, target_el_deg)

        if progress_callback:
            progress_callback("Analysis Complete.")

        return {
            "metrics": metrics,
            "figures": figures
        }

    def _build_pid_config(self, base_cfg: dict) -> SimulationConfig:
        """Builds the SimulationConfig for standard PID based on GUI inputs."""
        # Use deepcopy to ensure nested dicts are independent
        cfg_data = copy.deepcopy(base_cfg)
        
        excluded_keys = ["_gui_duration", "duration"]
        cfg_args = {k: v for k, v in cfg_data.items() if k not in excluded_keys}
        cfg = SimulationConfig(**cfg_args)
        
        cfg.use_feedback_linearization = False
        cfg.enable_plotting = False
        
        # Fast handover gating match legacy script
        if not hasattr(cfg, 'qpd_config') or not isinstance(cfg.qpd_config, dict):
            cfg.qpd_config = {}
        cfg.qpd_config['linear_range'] = 0.008
        
        return cfg

    def _build_fbl_config(self, base_cfg: dict) -> SimulationConfig:
        """Builds the SimulationConfig for standard FBL based on GUI inputs."""
        cfg_data = copy.deepcopy(base_cfg)
        
        excluded_keys = ["_gui_duration", "duration"]
        cfg_args = {k: v for k, v in cfg_data.items() if k not in excluded_keys}
        cfg = SimulationConfig(**cfg_args)
        
        cfg.use_feedback_linearization = True
        cfg.use_direct_state_feedback = False
        cfg.enable_plotting = False
        cfg.enable_visualization = False
        
        if not hasattr(cfg, 'qpd_config') or not isinstance(cfg.qpd_config, dict):
            cfg.qpd_config = {}
        cfg.qpd_config['linear_range'] = 0.008
        
        # Ensure NDOB is disabled for baseline FBL test
        if not hasattr(cfg, 'ndob_config') or not isinstance(cfg.ndob_config, dict):
            cfg.ndob_config = {}
        cfg.ndob_config['enable'] = False
        
        # FBL requires integral enabled for this specific test case
        if not hasattr(cfg, 'feedback_linearization_config') or not isinstance(cfg.feedback_linearization_config, dict):
            cfg.feedback_linearization_config = {}
        cfg.feedback_linearization_config['enable_integral'] = True
        
        return cfg

    def _build_ndob_config(self, base_cfg: dict) -> SimulationConfig:
        """Builds the SimulationConfig for FBL+NDOB based on GUI inputs."""
        # Start from FBL config
        cfg = self._build_fbl_config(base_cfg)
        
        # Reset targets to base base_cfg values to avoid mutation from previous test runs
        cfg.target_az = base_cfg.get("target_az", 0.0)
        cfg.target_el = base_cfg.get("target_el", 0.0)
        
        # Enable NDOB
        cfg.ndob_config['enable'] = True
        
        # Disable integral action; let NDOB handle steady state error (as requested)
        cfg.feedback_linearization_config['enable_integral'] = False
        cfg.feedback_linearization_config['enable_disturbance_compensation'] = False
            
        return cfg

    def _compute_metrics(self, results_pid, results_fbl, results_ndob, target_az_rad, target_el_rad) -> dict:
        """Process tracking metrics and return a formatted dictionary."""
        metrics_pid = compute_tracking_metrics(results_pid, target_az_rad, target_el_rad)
        metrics_fbl = compute_tracking_metrics(results_fbl, target_az_rad, target_el_rad)
        metrics_ndob = compute_tracking_metrics(results_ndob, target_az_rad, target_el_rad)
        
        # Calculate handover threshold checks
        thresh_deg = 0.8
        pid_pass = abs(np.rad2deg(metrics_pid['ss_error_az'])) <= thresh_deg
        fbl_pass = abs(np.rad2deg(metrics_fbl['ss_error_az'])) <= thresh_deg
        ndob_pass = abs(np.rad2deg(metrics_ndob['ss_error_az'])) <= thresh_deg

        # Torque efforts
        t_pid = np.sqrt(results_pid['torque_rms_az']**2 + results_pid['torque_rms_el']**2)
        t_fbl = np.sqrt(results_fbl['torque_rms_az']**2 + results_fbl['torque_rms_el']**2)
        t_ndob = np.sqrt(results_ndob['torque_rms_az']**2 + results_ndob['torque_rms_el']**2)

        return {
            "pid": {
                "settling_time_az": metrics_pid['settling_time_az'],
                "settling_time_el": metrics_pid['settling_time_el'],
                "ss_error_az": metrics_pid['ss_error_az'],
                "ss_error_el": metrics_pid['ss_error_el'],
                "los_error_rms": results_pid['los_error_rms'],
                "los_error_final": results_pid['los_error_final'],
                "torque_rms": t_pid,
                "fsm_saturation_pct": results_pid.get('fsm_saturation_pct', 0.0)
            },
            "fbl": {
                "settling_time_az": metrics_fbl['settling_time_az'],
                "settling_time_el": metrics_fbl['settling_time_el'],
                "ss_error_az": metrics_fbl['ss_error_az'],
                "ss_error_el": metrics_fbl['ss_error_el'],
                "los_error_rms": results_fbl['los_error_rms'],
                "los_error_final": results_fbl['los_error_final'],
                "torque_rms": t_fbl,
                "fsm_saturation_pct": results_fbl.get('fsm_saturation_pct', 0.0)
            },
            "ndob": {
                "settling_time_az": metrics_ndob['settling_time_az'],
                "settling_time_el": metrics_ndob['settling_time_el'],
                "ss_error_az": metrics_ndob['ss_error_az'],
                "ss_error_el": metrics_ndob['ss_error_el'],
                "los_error_rms": results_ndob['los_error_rms'],
                "los_error_final": results_ndob['los_error_final'],
                "torque_rms": t_ndob,
                "fsm_saturation_pct": results_ndob.get('fsm_saturation_pct', 0.0)
            },
            "handover": {
                "pid_pass": pid_pass,
                "fbl_pass": fbl_pass,
                "ndob_pass": ndob_pass
            },
            "raw": {
                "results_pid": results_pid,
                "results_fbl": results_fbl,
                "results_ndob": results_ndob
            }
        }

    def _generate_figures(self, results_pid, results_fbl, results_ndob, target_az_deg, target_el_deg) -> Dict:
        """Generate research figures without blocking or showing external windows."""
        # show_figures=False -> Intercepts plt.show()
        # interactive=False -> No interactive GUI elements (keeps it clean)
        import matplotlib
        matplotlib.use("Agg")  # CRITICAL: Force non-interactive thread-safe backend
        plotter = ResearchComparisonPlotter(
            save_figures=True,
            show_figures=False,  # CRITICAL: keep False. GUI handles displaying.
            interactive=False
        )
        figures = plotter.plot_all(results_pid, results_fbl, results_ndob, target_az_deg, target_el_deg)
        return figures
