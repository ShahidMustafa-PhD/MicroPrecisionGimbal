import numpy as np


class SimpleGimbalDynamics:
    """Simple diagonal 2-DOF gimbal dynamics model.

    Implements the same API as `GimbalDynamics` but assumes no coupling, no gravity,
    and no Coriolis terms:

        M ddq = tau

    This is intended as the default non-MuJoCo plant/controller model to keep the
    simulation runner deterministic and well-conditioned while still supporting
    Feedback Linearization and NDOB with a consistent dynamics interface.
    """

    def __init__(
        self,
        inertia_az: float = 2.0,
        inertia_el: float = 1.5,
    ) -> None:
        self.inertia_az = float(inertia_az)
        self.inertia_el = float(inertia_el)

    def get_mass_matrix(self, q: np.ndarray) -> np.ndarray:  # noqa: ARG002
        return np.array(
            [[self.inertia_az, 0.0], [0.0, self.inertia_el]],
            dtype=float,
        )

    def get_coriolis_matrix(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:  # noqa: ARG002
        return np.zeros((2, 2), dtype=float)

    def get_gravity_vector(self, q: np.ndarray) -> np.ndarray:  # noqa: ARG002
        return np.zeros(2, dtype=float)
