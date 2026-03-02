import numpy as np
import os
import matplotlib.pyplot as plt

print('Running full hybridsig 10s simulation to capture the oscillation data...')
from demo_feedback_linearization import SimulationConfig, DigitalTwinRunner

config_ndob = SimulationConfig(
    dt_sim=0.0001,
    dt_coarse=0.01,
    dt_fine=0.0001,
    log_period=0.0001,
    seed=42,
    target_az=np.deg2rad(0.0), 
    target_el=np.deg2rad(0.0),
    target_enabled=True,
    target_type='hybridsig',
    target_amplitude=8.0,
    target_period=4.0,
    target_reachangle=1.0, 
    use_feedback_linearization=True,
    environmental_disturbance_enabled=True,
    environmental_disturbance_config={
        'wind': {'enabled': True, 'mean_velocity': 8.0, 'turbulence_intensity': 0.15, 'start_time': 5.0},
        'vibration': {'enabled': False},  
        'structural_noise': {'enabled': True, 'std': 0.01, 'freq_low': 100, 'freq_high': 500}
    },
    ndob_config={
        'enable': True,
        'lambda_az': 50.0,
        'lambda_el': 50.0,
        'd_max': 5.0
    },
    enable_visualization=False,
    enable_plotting=False
)

runner = DigitalTwinRunner(config_ndob)
runner.run_simulation(duration=10.0)

t = np.array(runner.log_data['time'])
fsm_az = np.array(runner.log_data['fsm_tip']) * 2

mask = t > 8.0
t_ss = t[mask]
fsm_ss = fsm_az[mask]

dt = config_ndob.dt_fine
freqs = np.fft.rfftfreq(len(fsm_ss), dt)
fft_mag = np.abs(np.fft.rfft(fsm_ss - np.mean(fsm_ss)))

dom_freq = freqs[np.argmax(fft_mag)]
amp = np.max(fft_mag) / len(fsm_ss) * 2

print(f'\n*** OSCILLATION ANALYSIS [8-10s] ***')
print(f'Dominant Freq: {dom_freq:.2f} Hz')
print(f'Amplitude: {np.rad2deg(amp)*1000:.2f} mdeg')
