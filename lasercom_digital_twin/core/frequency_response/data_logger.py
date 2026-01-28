"""
Frequency Response Data Logger

This module provides industrial-grade data logging and persistence for
frequency response analysis results. Features include:

- JSON-based storage with full metadata
- Reproducibility tracking (configuration, timestamps, versions)
- Incremental logging during long sweeps
- Export to CSV for external analysis tools
- Integrity verification via checksums

Data Schema
-----------
The JSON output follows a hierarchical structure:

{
    "metadata": {
        "timestamp": "2026-01-28T12:00:00Z",
        "version": "1.0.0",
        "config": { ... },
        "checksums": { ... }
    },
    "controllers": {
        "PID": {
            "axis": "az",
            "frequencies_hz": [...],
            "closed_loop_gain_db": [...],
            "closed_loop_phase_deg": [...],
            "sensitivity_gain_db": [...],
            "coherence": [...],
            "metrics": {
                "bandwidth_hz": 12.5,
                "peak_sensitivity": 1.8,
                ...
            }
        },
        ...
    }
}

Author: Dr. S. Shahid Mustafa
Date: January 28, 2026
"""

import json
import numpy as np
import hashlib
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import csv

# Local imports
from .frequency_response_analyzer import FrequencyResponseData, ControllerType


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays."""
    
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, ControllerType):
            return obj.name
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


@dataclass
class LoggerConfig:
    """
    Configuration for frequency response data logger.
    
    Attributes
    ----------
    output_dir : Path
        Directory for saved data files
    base_filename : str
        Base name for output files
    save_json : bool
        Save JSON format
    save_csv : bool
        Save CSV format for each controller
    include_raw_points : bool
        Include individual FrequencyPoint data
    include_checksums : bool
        Add integrity checksums to metadata
    pretty_print : bool
        Format JSON with indentation
    version : str
        Data format version string
    """
    output_dir: Path = field(default_factory=lambda: Path('frequency_response_data'))
    base_filename: str = 'freq_response'
    save_json: bool = True
    save_csv: bool = True
    include_raw_points: bool = False
    include_checksums: bool = True
    pretty_print: bool = True
    version: str = '1.0.0'


class FrequencyResponseLogger:
    """
    Data Logger for Frequency Response Analysis.
    
    Provides persistent storage of frequency response data with full
    traceability and reproducibility metadata. Supports both JSON for
    archival and CSV for external analysis tools (MATLAB, Excel, etc.)
    
    Example Usage
    -------------
    >>> logger = FrequencyResponseLogger(LoggerConfig(output_dir=Path('data')))
    >>> 
    >>> # Add analysis results
    >>> logger.add_result(ControllerType.PID, pid_data)
    >>> logger.add_result(ControllerType.FBL, fbl_data)
    >>> logger.add_result(ControllerType.FBL_NDOB, ndob_data)
    >>> 
    >>> # Add sweep configuration for reproducibility
    >>> logger.set_sweep_config(sweep_config)
    >>> 
    >>> # Save all data
    >>> logger.save()
    
    Parameters
    ----------
    config : LoggerConfig
        Logger configuration
    """
    
    def __init__(self, config: Optional[LoggerConfig] = None):
        self.config = config or LoggerConfig()
        self._results: Dict[ControllerType, FrequencyResponseData] = {}
        self._sweep_config: Optional[Dict] = None
        self._custom_metadata: Dict[str, Any] = {}
        self._start_time = datetime.now()
    
    def add_result(
        self,
        controller_type: ControllerType,
        data: FrequencyResponseData
    ) -> None:
        """
        Add frequency response data for a controller.
        
        Parameters
        ----------
        controller_type : ControllerType
            Controller identifier
        data : FrequencyResponseData
            Frequency response data
        """
        self._results[controller_type] = data
    
    def add_results_dict(
        self,
        results: Dict[ControllerType, FrequencyResponseData]
    ) -> None:
        """
        Add multiple results at once.
        
        Parameters
        ----------
        results : Dict
            Dictionary of controller type to frequency response data
        """
        self._results.update(results)
    
    def set_sweep_config(self, config: Any) -> None:
        """
        Set sweep configuration for reproducibility tracking.
        
        Parameters
        ----------
        config : Any
            Sweep configuration (dataclass or dict)
        """
        if hasattr(config, '__dataclass_fields__'):
            self._sweep_config = asdict(config)
        elif isinstance(config, dict):
            self._sweep_config = config
        else:
            self._sweep_config = {'raw': str(config)}
    
    def add_metadata(self, key: str, value: Any) -> None:
        """
        Add custom metadata.
        
        Parameters
        ----------
        key : str
            Metadata key
        value : Any
            Metadata value (must be JSON-serializable)
        """
        self._custom_metadata[key] = value
    
    def save(self, suffix: Optional[str] = None) -> Dict[str, Path]:
        """
        Save all data to disk.
        
        Parameters
        ----------
        suffix : str, optional
            Optional suffix for filename
            
        Returns
        -------
        Dict[str, Path]
            Dictionary of format to saved file paths
        """
        saved_files = {}
        
        # Ensure output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        timestamp = self._start_time.strftime('%Y%m%d_%H%M%S')
        base = f"{self.config.base_filename}_{timestamp}"
        if suffix:
            base = f"{base}_{suffix}"
        
        # Build data structure
        data = self._build_data_structure()
        
        # Save JSON
        if self.config.save_json:
            json_path = self.config.output_dir / f"{base}.json"
            self._save_json(data, json_path)
            saved_files['json'] = json_path
        
        # Save CSV (one per controller)
        if self.config.save_csv:
            csv_paths = self._save_csv(base)
            saved_files['csv'] = csv_paths
        
        return saved_files
    
    def _build_data_structure(self) -> Dict[str, Any]:
        """Build complete data structure for serialization."""
        data = {
            'metadata': self._build_metadata(),
            'controllers': {}
        }
        
        for ctrl_type, freq_data in self._results.items():
            data['controllers'][ctrl_type.name] = self._serialize_freq_data(freq_data)
        
        return data
    
    def _build_metadata(self) -> Dict[str, Any]:
        """Build metadata section."""
        metadata = {
            'timestamp': self._start_time.isoformat(),
            'version': self.config.version,
            'analysis_duration_s': (datetime.now() - self._start_time).total_seconds(),
        }
        
        if self._sweep_config:
            metadata['sweep_config'] = self._sweep_config
        
        if self._custom_metadata:
            metadata['custom'] = self._custom_metadata
        
        if self.config.include_checksums:
            metadata['checksums'] = self._compute_checksums()
        
        return metadata
    
    def _serialize_freq_data(self, data: FrequencyResponseData) -> Dict[str, Any]:
        """Serialize FrequencyResponseData to dictionary."""
        return {
            'axis': data.axis,
            'frequencies_hz': data.frequencies_hz.tolist(),
            'frequencies_rad': data.frequencies_rad.tolist(),
            'closed_loop_gain_db': data.closed_loop_gain_db.tolist(),
            'closed_loop_phase_deg': data.closed_loop_phase_deg.tolist(),
            'sensitivity_gain_db': data.sensitivity_gain_db.tolist(),
            'sensitivity_phase_deg': data.sensitivity_phase_deg.tolist(),
            'control_effort_gain_db': data.control_effort_gain_db.tolist(),
            'coherence': data.coherence.tolist(),
            'metrics': {
                'bandwidth_hz': float(data.bandwidth_hz),
                'peak_sensitivity': float(data.peak_sensitivity),
                'peak_sensitivity_db': float(20 * np.log10(data.peak_sensitivity + 1e-12)),
                'gain_margin_db': float(data.gain_margin_db) if not np.isinf(data.gain_margin_db) else 'inf',
                'phase_margin_deg': float(data.phase_margin_deg),
            },
            'metadata': data.metadata,
        }
    
    def _compute_checksums(self) -> Dict[str, str]:
        """Compute checksums for data integrity verification."""
        checksums = {}
        
        for ctrl_type, freq_data in self._results.items():
            # Checksum based on frequency and gain arrays
            combined = np.concatenate([
                freq_data.frequencies_hz,
                freq_data.closed_loop_gain_db,
                freq_data.sensitivity_gain_db
            ])
            # Handle NaN values for hashing
            combined = np.nan_to_num(combined, nan=0.0)
            hash_bytes = combined.tobytes()
            checksums[ctrl_type.name] = hashlib.md5(hash_bytes).hexdigest()
        
        return checksums
    
    def _save_json(self, data: Dict, filepath: Path) -> None:
        """Save data as JSON."""
        indent = 2 if self.config.pretty_print else None
        
        with open(filepath, 'w') as f:
            json.dump(data, f, cls=NumpyEncoder, indent=indent)
        
        print(f"  [JSON] Saved: {filepath}")
    
    def _save_csv(self, base: str) -> List[Path]:
        """Save data as CSV files (one per controller)."""
        csv_paths = []
        
        for ctrl_type, freq_data in self._results.items():
            filepath = self.config.output_dir / f"{base}_{ctrl_type.name}.csv"
            
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Header
                writer.writerow([
                    'frequency_hz',
                    'frequency_rad',
                    'closed_loop_gain_db',
                    'closed_loop_phase_deg',
                    'sensitivity_gain_db',
                    'sensitivity_phase_deg',
                    'coherence'
                ])
                
                # Data rows
                for i in range(len(freq_data.frequencies_hz)):
                    writer.writerow([
                        freq_data.frequencies_hz[i],
                        freq_data.frequencies_rad[i],
                        freq_data.closed_loop_gain_db[i],
                        freq_data.closed_loop_phase_deg[i],
                        freq_data.sensitivity_gain_db[i],
                        freq_data.sensitivity_phase_deg[i],
                        freq_data.coherence[i]
                    ])
            
            csv_paths.append(filepath)
            print(f"  [CSV] Saved: {filepath}")
        
        return csv_paths
    
    @staticmethod
    def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Load frequency response data from JSON file.
        
        Parameters
        ----------
        filepath : Path or str
            Path to JSON file
            
        Returns
        -------
        Dict
            Loaded data structure
        """
        with open(filepath, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def verify_checksum(filepath: Union[str, Path]) -> bool:
        """
        Verify data integrity using stored checksums.
        
        Parameters
        ----------
        filepath : Path or str
            Path to JSON file
            
        Returns
        -------
        bool
            True if all checksums match
        """
        data = FrequencyResponseLogger.load_json(filepath)
        
        if 'checksums' not in data.get('metadata', {}):
            print("No checksums found in file")
            return True
        
        stored_checksums = data['metadata']['checksums']
        
        for ctrl_name, ctrl_data in data['controllers'].items():
            # Recompute checksum
            combined = np.concatenate([
                np.array(ctrl_data['frequencies_hz']),
                np.array(ctrl_data['closed_loop_gain_db']),
                np.array(ctrl_data['sensitivity_gain_db'])
            ])
            combined = np.nan_to_num(combined, nan=0.0)
            computed = hashlib.md5(combined.tobytes()).hexdigest()
            
            stored = stored_checksums.get(ctrl_name, '')
            if computed != stored:
                print(f"Checksum mismatch for {ctrl_name}: {computed} != {stored}")
                return False
        
        print("All checksums verified successfully")
        return True
