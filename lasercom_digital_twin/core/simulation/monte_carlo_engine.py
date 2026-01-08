"""
Monte Carlo Simulation Engine for Digital Twin Uncertainty Analysis

This module implements industrial-grade Monte Carlo analysis for evaluating
system performance under parameter uncertainty. Enables batch execution with
systematic parameter randomization according to specified distributions.

Design Philosophy:
-----------------
- **Deterministic per run:** Each MC run uses a unique seed for reproducibility
- **Comprehensive uncertainty:** Covers actuator, sensor, and structural parameters  
- **Parallel-ready:** Architecture supports future parallelization
- **Standards compliance:** Follows aerospace verification practices (MIL-STD-1540E)

Typical Use Case:
----------------
Evaluate performance margin when component parameters vary within manufacturing
tolerances. Example: Motor torque constant K_t varies ±5%, sensor bias ±10 µrad.
Run 100+ iterations to establish statistical confidence in performance metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import warnings
from pathlib import Path
import json


class DistributionType(Enum):
    """Statistical distributions for parameter uncertainty."""
    UNIFORM = "uniform"
    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    TRUNCATED_NORMAL = "truncated_normal"


@dataclass
class ParameterUncertainty:
    """
    Definition of uncertain parameter with statistical distribution.
    
    Attributes
    ----------
    name : str
        Parameter name (e.g., 'motor_az.K_t', 'gyro_az.noise_std')
    nominal : float
        Nominal/mean value
    distribution : DistributionType
        Statistical distribution type
    uncertainty : float
        Uncertainty magnitude (interpretation depends on distribution)
    bounds : Optional[Tuple[float, float]]
        Physical bounds [min, max] for parameter
    """
    name: str
    nominal: float
    distribution: DistributionType
    uncertainty: float  # ±% for normal, range for uniform
    bounds: Optional[Tuple[float, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MonteCarloConfig:
    """
    Configuration for Monte Carlo batch execution.
    
    Attributes
    ----------
    n_runs : int
        Number of Monte Carlo iterations
    base_seed : int
        Base random seed (each run uses base_seed + run_index)
    parameter_uncertainties : List[ParameterUncertainty]
        List of parameters to randomize
    simulation_duration : float
        Duration of each simulation run [s]
    save_telemetry : bool
        Whether to save full telemetry for each run
    output_dir : Optional[Path]
        Directory for saving results
    """
    n_runs: int = 100
    base_seed: int = 42
    parameter_uncertainties: List[ParameterUncertainty] = field(default_factory=list)
    simulation_duration: float = 10.0
    save_telemetry: bool = False
    output_dir: Optional[Path] = None
    parallel: bool = False  # Reserved for future parallelization
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MonteCarloRun:
    """Results from a single Monte Carlo run."""
    run_id: int
    seed: int
    parameters: Dict[str, float]
    metrics: Any  # PerformanceMetrics object
    telemetry: Optional[Dict[str, List[float]]] = None
    success: bool = True
    error_message: str = ""
    execution_time: float = 0.0


@dataclass
class MonteCarloResults:
    """Aggregated results from Monte Carlo batch."""
    runs: List[MonteCarloRun]
    summary_statistics: pd.DataFrame
    config: MonteCarloConfig
    total_execution_time: float = 0.0
    n_successful: int = 0
    n_failed: int = 0


class ParameterRandomizer:
    """
    Utility class for randomizing parameters with specified distributions.
    
    Handles sampling from various distributions with proper bounds enforcement.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize randomizer with seed.
        
        Parameters
        ----------
        seed : int
            Random seed for reproducibility
        """
        self.rng = np.random.default_rng(seed)
    
    def sample(self, param: ParameterUncertainty) -> float:
        """
        Sample parameter value from specified distribution.
        
        Parameters
        ----------
        param : ParameterUncertainty
            Parameter definition with distribution
            
        Returns
        -------
        float
            Sampled parameter value
        """
        if param.distribution == DistributionType.UNIFORM:
            # Uniform: uncertainty is ±range around nominal
            delta = param.nominal * param.uncertainty / 100.0
            value = self.rng.uniform(param.nominal - delta, param.nominal + delta)
            
        elif param.distribution == DistributionType.NORMAL:
            # Normal: uncertainty is ±1σ as % of nominal
            sigma = param.nominal * param.uncertainty / 100.0
            value = self.rng.normal(param.nominal, sigma)
            
        elif param.distribution == DistributionType.TRUNCATED_NORMAL:
            # Truncated normal: 3σ within bounds
            sigma = param.nominal * param.uncertainty / 100.0
            value = self.rng.normal(param.nominal, sigma)
            
            # Truncate to bounds (if specified)
            if param.bounds is not None:
                value = np.clip(value, param.bounds[0], param.bounds[1])
                
        elif param.distribution == DistributionType.LOGNORMAL:
            # Lognormal: for strictly positive parameters
            mu = np.log(param.nominal)
            sigma = param.uncertainty / 100.0  # Relative std
            value = self.rng.lognormal(mu, sigma)
            
        else:
            raise ValueError(f"Unknown distribution: {param.distribution}")
        
        # Enforce bounds if specified
        if param.bounds is not None:
            value = np.clip(value, param.bounds[0], param.bounds[1])
        
        return value
    
    def sample_batch(
        self,
        uncertainties: List[ParameterUncertainty]
    ) -> Dict[str, float]:
        """
        Sample all parameters at once.
        
        Parameters
        ----------
        uncertainties : List[ParameterUncertainty]
            List of parameter definitions
            
        Returns
        -------
        Dict[str, float]
            Dictionary mapping parameter names to sampled values
        """
        return {param.name: self.sample(param) for param in uncertainties}


class MonteCarloEngine:
    """
    Monte Carlo simulation engine for uncertainty quantification.
    
    This class orchestrates batch execution of the digital twin with
    systematic parameter randomization, enabling evaluation of performance
    margins under manufacturing tolerances and environmental variations.
    
    Usage:
    ------
    >>> from lasercom_digital_twin.core.simulation.simulation_runner import (
    ...     DigitalTwinRunner, SimulationConfig
    ... )
    >>> from lasercom_digital_twin.core.simulation.performance_analyzer import (
    ...     PerformanceAnalyzer
    ... )
    >>> 
    >>> # Define uncertainties
    >>> uncertainties = [
    ...     ParameterUncertainty('motor_az.K_t', 0.1, DistributionType.NORMAL, 5.0),
    ...     ParameterUncertainty('gyro_az.noise_std', 1e-5, DistributionType.UNIFORM, 10.0)
    ... ]
    >>> 
    >>> # Configure Monte Carlo
    >>> mc_config = MonteCarloConfig(
    ...     n_runs=100,
    ...     parameter_uncertainties=uncertainties,
    ...     simulation_duration=10.0
    ... )
    >>> 
    >>> # Create engine
    >>> engine = MonteCarloEngine(mc_config, PerformanceAnalyzer())
    >>> 
    >>> # Run batch
    >>> results = engine.run_batch(simulation_factory)
    >>> 
    >>> # Analyze results
    >>> print(engine.generate_report(results))
    """
    
    def __init__(
        self,
        config: MonteCarloConfig,
        analyzer: 'PerformanceAnalyzer',
        verbose: bool = True
    ):
        """
        Initialize Monte Carlo engine.
        
        Parameters
        ----------
        config : MonteCarloConfig
            Monte Carlo configuration
        analyzer : PerformanceAnalyzer
            Performance analyzer for computing metrics
        verbose : bool
            Print progress messages
        """
        self.config = config
        self.analyzer = analyzer
        self.verbose = verbose
        
        # Create output directory if specified
        if config.output_dir is not None:
            config.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _apply_parameters_to_config(
        self,
        base_config: Dict[str, Any],
        parameters: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Apply randomized parameters to simulation configuration.
        
        Handles nested dictionary paths (e.g., 'motor_az.K_t').
        
        Parameters
        ----------
        base_config : Dict[str, Any]
            Base configuration dictionary
        parameters : Dict[str, float]
            Dictionary of parameter names to values
            
        Returns
        -------
        Dict[str, Any]
            Modified configuration with randomized parameters
        """
        import copy
        config = copy.deepcopy(base_config)
        
        for param_path, value in parameters.items():
            # Parse nested path (e.g., 'motor_az.K_t' → ['motor_az', 'K_t'])
            keys = param_path.split('.')
            
            # Navigate to nested dictionary
            target = config
            for key in keys[:-1]:
                if key not in target:
                    target[key] = {}
                target = target[key]
            
            # Set value
            target[keys[-1]] = value
        
        return config
    
    def run_batch(
        self,
        simulation_factory: Callable[[Dict, int], Tuple[Any, Dict]]
    ) -> MonteCarloResults:
        """
        Execute batch of Monte Carlo simulations.
        
        The simulation_factory is responsible for:
        1. Creating a DigitalTwinRunner with modified config
        2. Running the simulation
        3. Returning (runner, telemetry) tuple
        
        Parameters
        ----------
        simulation_factory : Callable[[Dict, int], Tuple[Any, Dict]]
            Factory function: (config_dict, seed) → (runner, telemetry)
            
            Example:
            ```python
            def factory(config, seed):
                sim_config = SimulationConfig(seed=seed, **config)
                runner = DigitalTwinRunner(sim_config)
                telemetry = runner.run_simulation(duration=10.0)
                return runner, telemetry
            ```
            
        Returns
        -------
        MonteCarloResults
            Aggregated results from all runs
        """
        if self.verbose:
            print("=" * 70)
            print("MONTE CARLO BATCH EXECUTION")
            print("=" * 70)
            print(f"Number of runs:       {self.config.n_runs}")
            print(f"Parameters varied:    {len(self.config.parameter_uncertainties)}")
            print(f"Simulation duration:  {self.config.simulation_duration:.2f} s")
            print(f"Base seed:            {self.config.base_seed}")
            print("=" * 70)
        
        runs: List[MonteCarloRun] = []
        start_time_batch = time.time()
        
        for run_id in range(self.config.n_runs):
            if self.verbose and (run_id % 10 == 0 or run_id == self.config.n_runs - 1):
                print(f"Progress: {run_id + 1}/{self.config.n_runs} "
                      f"({100*(run_id+1)/self.config.n_runs:.1f}%)")
            
            # Generate unique seed for this run
            run_seed = self.config.base_seed + run_id
            
            # Randomize parameters
            randomizer = ParameterRandomizer(seed=run_seed)
            parameters = randomizer.sample_batch(self.config.parameter_uncertainties)
            
            # Create configuration with randomized parameters
            base_config = {
                'simulation_duration': self.config.simulation_duration,
                'seed': run_seed
            }
            run_config = self._apply_parameters_to_config(base_config, parameters)
            
            # Execute simulation
            start_time_run = time.time()
            success = True
            error_msg = ""
            telemetry = None
            metrics = None
            
            try:
                runner, telemetry = simulation_factory(run_config, run_seed)
                
                # Compute metrics
                metrics = self.analyzer.analyze(telemetry)
                
            except Exception as e:
                success = False
                error_msg = str(e)
                warnings.warn(f"Run {run_id} failed: {error_msg}")
            
            execution_time = time.time() - start_time_run
            
            # Store results
            run = MonteCarloRun(
                run_id=run_id,
                seed=run_seed,
                parameters=parameters,
                metrics=metrics,
                telemetry=telemetry if self.config.save_telemetry else None,
                success=success,
                error_message=error_msg,
                execution_time=execution_time
            )
            runs.append(run)
            
            # Save individual run if requested
            if self.config.output_dir is not None and success:
                self._save_run(run, run_id)
        
        total_time = time.time() - start_time_batch
        
        # Aggregate results
        results = self._aggregate_results(runs, total_time)
        
        if self.verbose:
            print("=" * 70)
            print(f"Batch complete: {results.n_successful}/{self.config.n_runs} successful")
            print(f"Total time: {total_time:.2f} s ({total_time/self.config.n_runs:.2f} s/run)")
            print("=" * 70)
        
        # Save summary
        if self.config.output_dir is not None:
            self._save_summary(results)
        
        return results
    
    def _aggregate_results(
        self,
        runs: List[MonteCarloRun],
        total_time: float
    ) -> MonteCarloResults:
        """
        Aggregate metrics across all runs into summary statistics.
        
        Parameters
        ----------
        runs : List[MonteCarloRun]
            List of individual run results
        total_time : float
            Total execution time [s]
            
        Returns
        -------
        MonteCarloResults
            Aggregated results with summary statistics
        """
        # Filter successful runs
        successful_runs = [r for r in runs if r.success and r.metrics is not None]
        n_successful = len(successful_runs)
        n_failed = len(runs) - n_successful
        
        if n_successful == 0:
            warnings.warn("No successful runs to aggregate")
            return MonteCarloResults(
                runs=runs,
                summary_statistics=pd.DataFrame(),
                config=self.config,
                total_execution_time=total_time,
                n_successful=0,
                n_failed=n_failed
            )
        
        # Extract metrics into DataFrame
        metrics_list = []
        for run in successful_runs:
            df = self.analyzer.to_dataframe(run.metrics)
            df['run_id'] = run.run_id
            df['seed'] = run.seed
            
            # Add parameter values
            for param_name, param_value in run.parameters.items():
                df[f'param_{param_name}'] = param_value
            
            metrics_list.append(df)
        
        all_metrics = pd.concat(metrics_list, ignore_index=True)
        
        # Compute summary statistics
        metrics_cols = [
            'rms_pointing_error', 'peak_pointing_error',
            'fsm_saturation_pct', 'fsm_rms_command',
            'steady_state_error', 'jitter_rms'
        ]
        
        summary = {}
        for col in metrics_cols:
            if col in all_metrics.columns:
                summary[f'{col}_mean'] = all_metrics[col].mean()
                summary[f'{col}_std'] = all_metrics[col].std()
                summary[f'{col}_min'] = all_metrics[col].min()
                summary[f'{col}_max'] = all_metrics[col].max()
                summary[f'{col}_median'] = all_metrics[col].median()
                summary[f'{col}_p95'] = all_metrics[col].quantile(0.95)
        
        # Pass/fail statistics
        if 'meets_rms_req' in all_metrics.columns:
            summary['pass_rate_rms'] = all_metrics['meets_rms_req'].sum() / n_successful * 100
            summary['pass_rate_peak'] = all_metrics['meets_peak_req'].sum() / n_successful * 100
            summary['pass_rate_sat'] = all_metrics['meets_sat_req'].sum() / n_successful * 100
        
        summary_df = pd.DataFrame([summary])
        
        return MonteCarloResults(
            runs=runs,
            summary_statistics=summary_df,
            config=self.config,
            total_execution_time=total_time,
            n_successful=n_successful,
            n_failed=n_failed
        )
    
    def _save_run(self, run: MonteCarloRun, run_id: int) -> None:
        """Save individual run results to disk."""
        if self.config.output_dir is None:
            return
        
        run_dir = self.config.output_dir / f"run_{run_id:04d}"
        run_dir.mkdir(exist_ok=True)
        
        # Save parameters
        with open(run_dir / "parameters.json", 'w') as f:
            json.dump(run.parameters, f, indent=2)
        
        # Save metrics
        if run.metrics is not None:
            df = self.analyzer.to_dataframe(run.metrics)
            df.to_csv(run_dir / "metrics.csv", index=False)
        
        # Save telemetry if available
        if run.telemetry is not None and self.config.save_telemetry:
            telemetry_df = pd.DataFrame(run.telemetry)
            telemetry_df.to_csv(run_dir / "telemetry.csv", index=False)
    
    def _save_summary(self, results: MonteCarloResults) -> None:
        """Save aggregated summary to disk."""
        if self.config.output_dir is None:
            return
        
        # Save summary statistics
        results.summary_statistics.to_csv(
            self.config.output_dir / "summary_statistics.csv",
            index=False
        )
        
        # Save configuration
        config_dict = {
            'n_runs': self.config.n_runs,
            'base_seed': self.config.base_seed,
            'simulation_duration': self.config.simulation_duration,
            'n_successful': results.n_successful,
            'n_failed': results.n_failed,
            'total_execution_time': results.total_execution_time,
            'parameters': [
                {
                    'name': p.name,
                    'nominal': p.nominal,
                    'distribution': p.distribution.value,
                    'uncertainty': p.uncertainty
                }
                for p in self.config.parameter_uncertainties
            ]
        }
        
        with open(self.config.output_dir / "config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def generate_report(self, results: MonteCarloResults) -> str:
        """
        Generate comprehensive Monte Carlo analysis report.
        
        Parameters
        ----------
        results : MonteCarloResults
            Aggregated MC results
            
        Returns
        -------
        str
            Formatted report text
        """
        report = []
        report.append("=" * 70)
        report.append("MONTE CARLO ANALYSIS REPORT")
        report.append("=" * 70)
        report.append("")
        
        # Execution summary
        report.append("EXECUTION SUMMARY:")
        report.append(f"  Total runs:            {self.config.n_runs}")
        report.append(f"  Successful:            {results.n_successful} "
                     f"({100*results.n_successful/self.config.n_runs:.1f}%)")
        report.append(f"  Failed:                {results.n_failed}")
        report.append(f"  Total time:            {results.total_execution_time:.2f} s")
        report.append(f"  Time per run:          {results.total_execution_time/self.config.n_runs:.2f} s")
        report.append("")
        
        # Parameter uncertainties
        report.append("PARAMETER UNCERTAINTIES:")
        for param in self.config.parameter_uncertainties:
            report.append(f"  {param.name:30s}: {param.nominal:10.4g} ± {param.uncertainty:5.1f}% "
                         f"[{param.distribution.value}]")
        report.append("")
        
        # Primary performance metrics
        summary = results.summary_statistics
        if not summary.empty:
            report.append("PERFORMANCE STATISTICS (µrad):")
            report.append(f"  RMS Pointing Error:")
            report.append(f"    Mean:              {summary['rms_pointing_error_mean'].iloc[0]:10.2f}")
            report.append(f"    Std Dev:           {summary['rms_pointing_error_std'].iloc[0]:10.2f}")
            report.append(f"    Min:               {summary['rms_pointing_error_min'].iloc[0]:10.2f}")
            report.append(f"    Max:               {summary['rms_pointing_error_max'].iloc[0]:10.2f}")
            report.append(f"    Median:            {summary['rms_pointing_error_median'].iloc[0]:10.2f}")
            report.append(f"    95th percentile:   {summary['rms_pointing_error_p95'].iloc[0]:10.2f}")
            report.append("")
            
            report.append(f"  Peak Pointing Error:")
            report.append(f"    Mean:              {summary['peak_pointing_error_mean'].iloc[0]:10.2f}")
            report.append(f"    Max:               {summary['peak_pointing_error_max'].iloc[0]:10.2f}")
            report.append("")
            
            # FSM saturation
            report.append(f"  FSM Saturation (%):")
            report.append(f"    Mean:              {summary['fsm_saturation_pct_mean'].iloc[0]:10.2f}")
            report.append(f"    Max:               {summary['fsm_saturation_pct_max'].iloc[0]:10.2f}")
            report.append("")
            
            # Pass rates
            if 'pass_rate_rms' in summary.columns:
                report.append("PASS RATES:")
                report.append(f"  RMS Requirement:       {summary['pass_rate_rms'].iloc[0]:6.1f}%")
                report.append(f"  Peak Requirement:      {summary['pass_rate_peak'].iloc[0]:6.1f}%")
                report.append(f"  Saturation Req:        {summary['pass_rate_sat'].iloc[0]:6.1f}%")
                report.append("")
        
        # Performance margin
        if not summary.empty and 'rms_pointing_error_max' in summary.columns:
            worst_case = summary['rms_pointing_error_max'].iloc[0]
            requirement = self.analyzer.rms_requirement
            margin = (requirement - worst_case) / requirement * 100
            
            report.append("PERFORMANCE MARGIN:")
            report.append(f"  Requirement:           {requirement:.2f} µrad")
            report.append(f"  Worst case (95th %):   {summary['rms_pointing_error_p95'].iloc[0]:.2f} µrad")
            report.append(f"  Worst case (max):      {worst_case:.2f} µrad")
            report.append(f"  Margin:                {margin:+.1f}%")
            report.append("")
        
        report.append("=" * 70)
        
        return "\n".join(report)


def create_default_uncertainties() -> List[ParameterUncertainty]:
    """
    Create default set of parameter uncertainties for typical analysis.
    
    Returns
    -------
    List[ParameterUncertainty]
        Standard uncertainty definitions
    """
    uncertainties = [
        # Motor parameters
        ParameterUncertainty(
            name='motor_az.K_t',
            nominal=0.1,  # N·m/A
            distribution=DistributionType.NORMAL,
            uncertainty=5.0,  # ±5%
            metadata={'description': 'Motor torque constant'}
        ),
        ParameterUncertainty(
            name='motor_az.R',
            nominal=1.0,  # Ω
            distribution=DistributionType.NORMAL,
            uncertainty=3.0,  # ±3%
            metadata={'description': 'Motor resistance'}
        ),
        
        # Sensor parameters
        ParameterUncertainty(
            name='gyro_az.noise_std',
            nominal=1e-5,  # rad/s
            distribution=DistributionType.LOGNORMAL,
            uncertainty=20.0,  # Relative std
            bounds=(5e-6, 5e-5),
            metadata={'description': 'Gyro noise density'}
        ),
        ParameterUncertainty(
            name='gyro_az.bias',
            nominal=0.0,  # rad/s
            distribution=DistributionType.UNIFORM,
            uncertainty=100.0,  # ±100% around zero → uniform range
            bounds=(-1e-4, 1e-4),
            metadata={'description': 'Gyro bias offset'}
        ),
        ParameterUncertainty(
            name='encoder_az.resolution',
            nominal=1e-6,  # rad
            distribution=DistributionType.UNIFORM,
            uncertainty=10.0,
            metadata={'description': 'Encoder resolution'}
        ),
        
        # FSM parameters
        ParameterUncertainty(
            name='fsm.bandwidth',
            nominal=100.0,  # Hz
            distribution=DistributionType.NORMAL,
            uncertainty=10.0,
            bounds=(50.0, 150.0),
            metadata={'description': 'FSM mechanical bandwidth'}
        ),
        
        # Structural parameters
        ParameterUncertainty(
            name='gimbal_az.backlash',
            nominal=1e-5,  # rad
            distribution=DistributionType.LOGNORMAL,
            uncertainty=50.0,
            bounds=(5e-6, 5e-5),
            metadata={'description': 'Gimbal backlash'}
        ),
        ParameterUncertainty(
            name='gimbal_az.friction',
            nominal=0.1,  # N·m·s/rad
            distribution=DistributionType.NORMAL,
            uncertainty=20.0,
            bounds=(0.05, 0.3),
            metadata={'description': 'Viscous friction coefficient'}
        ),
    ]
    
    return uncertainties
