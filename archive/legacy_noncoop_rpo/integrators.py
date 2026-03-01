import numpy as np


# -------------------------
# rk4 (one step)
# -------------------------
def rk4_step(f, x: np.ndarray, dt: float, *f_args, **f_kwargs) -> np.ndarray:
    """
    Generic RK4 one-step integrator.
    """
    k1 = f(x, *f_args, **f_kwargs)
    k2 = f(x + 0.5 * dt * k1, *f_args, **f_kwargs)
    k3 = f(x + 0.5 * dt * k2, *f_args, **f_kwargs)
    k4 = f(x + dt * k3, *f_args, **f_kwargs)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)