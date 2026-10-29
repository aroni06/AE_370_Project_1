"""Right-hand side of the three-body IVP (Eqs. 2-4) and conserved quantities."""

import numpy as np

from .constants import G


def unpack(y):
    """Split state(s) of shape (..., 18) into positions and velocities (..., 3, 3)."""
    s = np.asarray(y).reshape(*np.shape(y)[:-1], 3, 6)
    return s[..., :3], s[..., 3:]


def accelerations(r, m):
    """Gravitational acceleration of each body, Eq. (3). r has shape (3, 3)."""
    a = np.zeros_like(r)
    for i, j in ((0, 1), (0, 2), (1, 2)):
        d = r[j] - r[i]
        inv_r3 = 1.0 / (d @ d) ** 1.5
        a[i] += G * m[j] * inv_r3 * d
        a[j] -= G * m[i] * inv_r3 * d
    return a


def rhs(t, y, m):
    """f(t, y) for the 18-component state y = [r_A, v_A, r_B, v_B, r_p, v_p]."""
    r, v = unpack(y)
    dy = np.empty((3, 6))
    dy[:, :3] = v
    dy[:, 3:] = accelerations(r, m)
    return dy.ravel()


def energy(Y, m):
    """Total energy (kinetic + potential) for state(s) Y of shape (..., 18)."""
    r, v = unpack(Y)
    kin = 0.5 * np.einsum("i,...ij,...ij->...", m, v, v)
    pot = 0.0
    for i, j in ((0, 1), (0, 2), (1, 2)):
        pot = pot - G * m[i] * m[j] / np.linalg.norm(r[..., i, :] - r[..., j, :], axis=-1)
    return kin + pot


def angular_momentum(Y, m):
    """Total angular momentum vector for state(s) Y of shape (..., 18)."""
    r, v = unpack(Y)
    return np.einsum("i,...ij->...j", m, np.cross(r, v))
