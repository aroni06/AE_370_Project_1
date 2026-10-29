"""Frame-independent diagnostics used to compare mass cases (Eqs. 10-11)."""

import numpy as np

from .dynamics import unpack


def stellar_barycenter(Y, m):
    r, _ = unpack(Y)
    return (m[0] * r[..., 0, :] + m[1] * r[..., 1, :]) / (m[0] + m[1])


def planet_relative(Y, m):
    """Planet position and velocity relative to the stellar barycenter, q_p."""
    r, v = unpack(Y)
    w = np.array([m[0], m[1]]) / (m[0] + m[1])
    q = r[..., 2, :] - (w[0] * r[..., 0, :] + w[1] * r[..., 1, :])
    qd = v[..., 2, :] - (w[0] * v[..., 0, :] + w[1] * v[..., 1, :])
    return q, qd


def binary_separation(Y):
    """r_B - r_A and its time derivative."""
    r, v = unpack(Y)
    return r[..., 1, :] - r[..., 0, :], v[..., 1, :] - v[..., 0, :]


def in_plane_phase(q, qd):
    """Unwrapped angle of q measured in the planet's initial orbital plane.

    The orbit is nearly edge-on in the inertial frame (i ~ 90 deg), so the
    angle is taken about the initial orbit normal q x qd rather than about z.
    """
    n = np.cross(q[0], qd[0])
    n /= np.linalg.norm(n)
    e1 = q[0] / np.linalg.norm(q[0])
    e2 = np.cross(n, e1)
    return np.unwrap(np.arctan2(q @ e2, q @ e1))


def revolution_times(t, theta):
    """Times at which the phase completes each full revolution (linear interp)."""
    k = np.arange(1, int((theta[-1] - theta[0]) // (2 * np.pi)) + 1)
    return np.interp(theta[0] + 2 * np.pi * k, theta, t)


def mean_revolution_time(t, theta):
    """Average revolution time over all complete revolutions."""
    tr = revolution_times(t, theta)
    return tr[-1] / len(tr)


def conjunction_times(t, Y, los=2):
    """Times of stellar conjunction along the line of sight (eclipse proxy).

    The binary orbit lies close to the x-z plane, so the line of sight is
    taken along the `los` axis (z by default). A conjunction is when the
    sky-plane x separation of B relative to A crosses zero with B on the +z
    side. Each crossing is refined with one Newton step using the velocity.
    """
    d, dd = binary_separation(Y)
    x, vx = d[:, 0], dd[:, 0]
    idx = np.where((np.sign(x[:-1]) != np.sign(x[1:])) & (d[:-1, los] > 0))[0]
    out = []
    for k in idx:
        j = k if abs(x[k]) < abs(x[k + 1]) else k + 1
        out.append(t[j] - x[j] / vx[j])
    return np.array(out)


def osculating_period(q, qd, mu):
    """Two-body period from a relative state through the vis-viva equation."""
    r = np.linalg.norm(q)
    a = 1.0 / (2.0 / r - qd @ qd / mu)
    return 2 * np.pi * np.sqrt(a**3 / mu)
