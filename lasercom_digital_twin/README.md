# Laser Communication Digital Twin

This repository contains the foundational software architecture for a ground-based satellite laser communication terminal digital twin.

## Design Principles

The architecture is designed to enforce the following principles:

*   **Deterministic Execution:** The modular structure, orchestrated by the `simulation_runner`, supports a fixed-step execution loop. Each subsystem (dynamics, actuators, etc.) is called in a defined order at each time step, ensuring repeatable and deterministic simulation runs.

*   **Parameter-driven Configuration:** All simulation parameters, model configurations, and scenario settings are intended to be loaded from a central configuration file or a dedicated configuration class. This object is passed to each module during initialization, allowing for easy modification of the simulation without changing the source code.

*   **Independent Subsystem Testing:** The `ci_tests/` directory is dedicated to continuous integration and unit testing. The modular design allows each subsystem to be tested in isolation by providing it with mock inputs and verifying its outputs, ensuring correctness and facilitating rapid development.

*   **Replaceable Fidelity Levels:** Each module in the architecture (e.g., `actuators/`, `sensors/`) is designed to be swappable. For example, you can have multiple motor models like `MotorModel_L1` (simple, low-fidelity) and `MotorModel_L4` (complex, high-fidelity) within `actuators/motor_models.py`. The simulation can be configured to use a specific model, allowing for a trade-off between simulation speed and accuracy depending on the analysis requirements.
