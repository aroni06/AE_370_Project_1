"""Two-body orbit tools: Kepler's equation, elements -> Cartesian state,
the analytic two-body solution, and the three-body initial state."""

import numpy as np

from . import constants as c


def solve_kepler(M, e, tol=1e-15, max_iter=50):
    """Solve M = E - e sin E for the eccentric anomaly E (Newton's method)."""
    M = np.asarray(M, dtype=float)
    E = M + e * np.sin(M)
    for _ in range(max_iter):
        dE = (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
        E = E - dE
        if np.all(np.abs(dE) < tol):
            break
    return E


def _rot_z(t):
    ct, st = np.cos(t), np.sin(t)
    return np.array([[ct, -st, 0.0], [st, ct, 0.0], [0.0, 0.0, 1.0]])


def _rot_x(t):
    ct, st = np.cos(t), np.sin(t)
    return np.array([[1.0, 0.0, 0.0], [0.0, ct, -st], [0.0, st, ct]])


def rotation(inc, Omega, omega):
    """R = Rz(Omega) Rx(i) Rz(omega), angles in degrees (Eq. 8)."""
    inc, Omega, omega = np.radians([inc, Omega, omega])
    return _rot_z(Omega) @ _rot_x(inc) @ _rot_z(omega)


def relative_state(el, mu, t=0.0):
    """Relative position and velocity of a two-body orbit at time t.

    el holds a, e, inc, Omega, omega, lam (degrees); mu is G times the total
    mass of the pair. At t = 0 this is Eqs. (6)-(8). For t != 0 the mean
    anomaly advances at n = sqrt(mu/a^3), which is the exact Kepler solution.
    t may be an array, in which case arrays of shape (len(t), 3) are returned.
    """
    a, e = el["a"], el["e"]
    n = np.sqrt(mu / a**3)
    M0 = np.radians(el["lam"] - el["omega"] - el["Omega"])
    t = np.asarray(t, dtype=float)
    E = solve_kepler(np.mod(M0 + n * t, 2 * np.pi), e)
    cE, sE = np.cos(E), np.sin(E)
    s = np.sqrt(1.0 - e * e)
    zero = np.zeros_like(E)
    r_pf = np.stack([a * (cE - e), a * s * sE, zero], axis=-1)
    v_pf = (n * a / (1.0 - e * cE))[..., None] * np.stack([-sE, s * cE, zero], axis=-1)
    R = rotation(el["inc"], el["Omega"], el["omega"])
    return r_pf @ R.T, v_pf @ R.T


def initial_state(m_p, velocity="fixed"):
    """Barycentric 18-component initial state for planet mass m_p (Msun).

    velocity="fixed": the planet's position and velocity relative to the
        stellar barycenter are built once with mu = G(M* + m_p0) and reused
        for every m_p (the draft's experiment, cases A and B).
    velocity="rebuilt": the planet's relative state is rebuilt from the same
        orbital elements with mu = G(M* + m_p), so the osculating orbit is
        the same for every m_p (case C).

    Returns (masses, y0) with masses = [m_A, m_B, m_p] and y0 ordered as
    [r_A, v_A, r_B, v_B, r_p, v_p] (Eq. 4).
    """
    r_ba, v_ba = relative_state(c.BINARY, c.G * c.M_STARS)
    mu_p = c.G * (c.M_STARS + (c.M_P0 if velocity == "fixed" else m_p))
    if velocity not in ("fixed", "rebuilt"):
        raise ValueError("velocity must be 'fixed' or 'rebuilt'")
    r_ps, v_ps = relative_state(c.PLANET, mu_p)

    ms, mt = c.M_STARS, c.M_STARS + m_p
    r = [-c.M_B / ms * r_ba - m_p / mt * r_ps,
         c.M_A / ms * r_ba - m_p / mt * r_ps,
         ms / mt * r_ps]
    v = [-c.M_B / ms * v_ba - m_p / mt * v_ps,
         c.M_A / ms * v_ba - m_p / mt * v_ps,
         ms / mt * v_ps]
    y0 = np.concatenate([np.concatenate([r[i], v[i]]) for i in range(3)])
    return np.array([c.M_A, c.M_B, m_p]), y0
