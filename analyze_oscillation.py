import numpy as np
import os
import matplotlib.pyplot as plt

def analyze_oscillation():
    # If the simulation saves data to npz or csv, we could load it here.
    # But since the user relies on the figure, let's look at the figure generation code
    # to find where the data is kept, or just write a script to run a 1s simulation
    # and plot the FFT of the output.
    
    print("Running a short 1s full simulation to capture the oscillation data...")
    from demo_feedback_linearization import SimulationConfig, DigitalTwinRunner
    
    config = SimulationConfig(
        dt_sim=0.0001,
        dt_coarse=0.01,
        dt_fine=0.0001,
        log_period=0.0001,
        seed=42,
        target_az=np.deg2rad(1.0),
        target_el=np.deg2rad(1.0),
        target_enabled=True,
        target_type='step',
        use_feedback_linearization=True,
        enable_visualization=False,
        enable_plotting=False
    )
    
    runner = DigitalTwinRunner(config)
    results = runner.run_simulation(duration=1.0)
    
    t = results['time']
    fsm_az = results['fsm_tip'] * 2  # optical multiplier
    
    # Analyze the last 0.5s for steady-state oscillation
    mask = t > 0.5
    t_ss = t[mask]
    fsm_ss = fsm_az[mask]
    
    # FFT
    dt = config.dt_fine
    freqs = np.fft.rfftfreq(len(fsm_ss), dt)
    fft_mag = np.abs(np.fft.rfft(fsm_ss - np.mean(fsm_ss)))
    
    dom_freq = freqs[np.argmax(fft_mag)]
    amp = np.max(fft_mag) / len(fsm_ss) * 2
    
    print(f"Dominant Oscillation Frequency: {dom_freq:.2f} Hz")
    print(f"Oscillation Amplitude: {np.rad2deg(amp)*1000:.2f} mdeg")
    
    # Save a debug plot
    plt.figure()
    plt.plot(t_ss, np.rad2deg(fsm_ss)*1000)
    plt.title(f"FSM Steady State (Dom Freq: {dom_freq:.1f} Hz)")
    plt.ylabel("Angle [mdeg]")
    plt.xlabel("Time [s]")
    plt.savefig('fsm_oscillation_debug.png')
    print("Saved plot to fsm_oscillation_debug.png")

if __name__ == '__main__':
    analyze_oscillation()
