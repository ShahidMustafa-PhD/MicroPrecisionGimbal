import numpy as np
import sys
from lasercom_digital_twin.core.dynamics.fsm_dynamics import create_fsm_from_vendor
from lasercom_digital_twin.core.controllers.fsm_pid_control import create_fsm_controller_from_design

def test_closed_loop():
    print("Testing FSM PIDF closed-loop with PI PZT...")
    # Initialize dynamics
    fsm = create_fsm_from_vendor("pi")
    info = fsm.config.get_info()
    print(f"Plant: f_n={info['f_n_Hz']} Hz, zeta={info['zeta']}, K={fsm.config.dc_sensitivity}")
    
    # Initialize controller
    pid = create_fsm_controller_from_design(bandwidth_hz=150.0)
    print(f"Gains: Kp={pid.tip_gains.Kp:.2f}, Ki={pid.tip_gains.Ki:.2f}, Kd={pid.tip_gains.Kd:.5f}, N={pid.tip_gains.N:.1f}")
    
    dt = 0.0001
    duration = 0.5
    steps = int(duration / dt)
    
    target = np.array([0.005, 0.0]) # 5 mrad target
    
    for i in range(steps):
        # Measure current output
        y = np.array([fsm.x[0], fsm.x[2]])  # Tip, tilt angles
        
        # Controller update (error = target - y, but in runner it passes setpoint=0, measurement=-error)
        u_cmd = pid.update(np.array([0.0, 0.0]), -(target - y), dt)
        
        u_sat = np.clip(u_cmd, -50.0, 50.0)
        
        # Step dynamics
        y_next = fsm.step(u_sat, dt)
        
        if np.any(np.isnan(fsm.x)) or np.any(np.abs(fsm.x) > 1e6):
            print(f"DIVERGED at step {i}, time {i*dt:.5f}s!")
            print(f"State: {fsm.x}")
            print(f"Command out: {u_sat}")
            return
            
    print(f"Success! Final state: {fsm.x}")
    print(f"Final output: {y_next}")
    
if __name__ == '__main__':
    test_closed_loop()
