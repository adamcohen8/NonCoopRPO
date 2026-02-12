import numpy as np
from typing import Optional
from .constants import MU_EARTH_KM3_S2

# -------------------------
# Two-body derivative
# -------------------------
def two_body_deriv(x: np.ndarray, mu: float = MU_EARTH_KM3_S2, a_cmd_eci: Optional[np.ndarray] = None) -> np.ndarray:
    """
    x = [rx,ry,rz,vx,vy,vz] in km, km/s
    a_cmd_eci in km/s^2
    returns xdot
    """
    r = x[0:3]
    v = x[3:6]
    rnorm = np.linalg.norm(r)
    a_grav = -mu * r / (rnorm**3)
    if a_cmd_eci is None:
        a = a_grav
    else:
        a = a_grav + a_cmd_eci
    return np.hstack((v, a))
