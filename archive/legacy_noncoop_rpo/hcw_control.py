import numpy as np
from scipy.linalg import expm, solve_discrete_are


# -----------------------------------
# HCW continuous-time matrices
# state = [x y z dx dy dz]
# control = [ux uy uz] accelerations
# -----------------------------------
def hcw_AB(n: float):
    A = np.array([
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [3*n**2, 0, 0, 0, 2*n, 0],
        [0, 0, 0, -2*n, 0, 0],
        [0, 0, -n**2, 0, 0, 0]
    ])

    B = np.array([
        [0,0,0],
        [0,0,0],
        [0,0,0],
        [1,0,0],
        [0,1,0],
        [0,0,1]
    ])

    return A, B


# -----------------------------------
# ZOH discretization
# -----------------------------------
def discretize(A, B, dt):
    n = A.shape[0]
    m = B.shape[1]

    M = np.zeros((n+m, n+m))
    M[:n,:n] = A
    M[:n,n:] = B

    Md = expm(M*dt)

    Ad = Md[:n,:n]
    Bd = Md[:n,n:]

    return Ad, Bd


# -----------------------------------
# Discrete LQR
# -----------------------------------
def dlqr(Ad, Bd, Q, R):
    P = solve_discrete_are(Ad, Bd, Q, R)
    K = np.linalg.inv(R + Bd.T @ P @ Bd) @ (Bd.T @ P @ Ad)
    return K


# -----------------------------------
# Convenience builder
# -----------------------------------
def build_hcw_lqr(n, dt, a_max):

    A, B = hcw_AB(n)
    Ad, Bd = discretize(A, B, dt)

    # Reasonable default weights
    Q = 1e3 * np.diag([
        8.66, 8.66, 8.66,   # position
        1.33, 1.33, 1.33    # velocity
    ])

    R = 1.94e14 * np.eye(3)  # penalize accel

    K = dlqr(Ad, Bd, Q, R)

    def controller(x_rel):
        u = -K @ x_rel

        # saturation
        norm = np.linalg.norm(u)
        if norm > a_max:
            u = u * (a_max / norm)

        return u

    return controller


def build_hcw_lqr_position_only(n, dt, a_max):
    """
    LQR builder with zero velocity-state penalties in Q.
    Useful when terminal relative velocity is not a priority.
    """
    A, B = hcw_AB(n)
    Ad, Bd = discretize(A, B, dt)

    Q = np.diag([
        1.0, 1.0, 1.0,   # position
        0.0, 0.0, 0.0      # velocity
    ])
    R = 1e11 * np.eye(3)
    try:
        K = dlqr(Ad, Bd, Q, R)
    except ValueError:
        # Zero velocity weights can be numerically ill-conditioned for DARE.
        # Add a tiny regularization while preserving position-only behavior.
        eps = 1e-12
        q_reg = np.diag([1.0, 1.0, 1.0, eps, eps, eps])
        K = dlqr(Ad, Bd, q_reg, R)

    def controller(x_rel):
        u = -K @ x_rel
        norm = np.linalg.norm(u)
        if norm > a_max:
            u = u * (a_max / norm)
        return u

    return controller
